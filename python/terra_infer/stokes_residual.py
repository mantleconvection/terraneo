r"""Differentiable Stokes momentum residual for a physics-informed loss term.

\section why Why this form

The momentum residual TERRA-NG poses is

    f_u = A u + grad p,        A u = -div( 2 eta ( eps(u) - (1/3)(div u) I ) )

but the network predicts only velocity, so ``grad p`` is not available from the
prediction. It is available from the *data*: by construction ``grad p = f_u - A u_true``.
Substituting that,

    residual(u_pred) = A u_pred + grad p - f_u  =  A u_pred - A u_true

so penalising ``|| A u_pred - A u_true ||`` **is** the momentum residual evaluated with
the exact pressure gradient. No pressure field is needed and nothing is approximated
away -- the two unknown terms cancel identically.

This also explains why the term helps beyond the data loss: it is a penalty on the
*derivatives* of the error, an H1-type term. Plain L2 on u under-weights exactly the
high-frequency component that spectral bias leaves behind, which is the same argument
HANO makes for its H1 loss on multiscale elliptic problems.

\section how How the operator is evaluated

Each subdomain is a logically structured (nx, ny, nr) block with known Cartesian node
positions, so derivatives come from central differences in index space composed with the
inverse Jacobian of the index-to-physical map:

    d/dx_b = sum_a (J^-1)_{b a} d/dxi_a,      J_{a b} = dx_a / dxi_b

``J^-1`` depends only on the mesh, so it is computed once and reused for every sample.
Two derivative passes are needed (grad u, then div of the stress), so the result is
valid only where a node has neighbours on both sides in all three directions -- indices
two clear of every face, which is 125 of 729 nodes per subdomain at level 3 (the second pass consumes gradients that already needed a neighbour). That is a regulariser, not a
solver: it does not have to be exact everywhere to pull the prediction toward the PDE.

It is also **not** TERRA-NG's discrete operator -- it is the strong form of the same
continuum operator, on the same nodes. :func:`validate` checks it against a sympy-derived
analytic residual.
"""

from __future__ import annotations

import numpy as np
import torch


def inverse_jacobian(coords: np.ndarray) -> torch.Tensor:
    """(S, nx, ny, nr, 3) node positions -> (S, nx, ny, nr, 3, 3) inverse Jacobian.

    ``J[a][b] = dx_a/dxi_b`` by central differences in index space (unit spacing);
    edges fall back to one-sided differences so the array stays full-size, but those
    entries are masked out of the loss anyway.
    """
    c = torch.as_tensor(np.ascontiguousarray(coords), dtype=torch.float64)
    # J[a][b] = dx_a/dxi_b, so inv_J[b][a] = dxi_b/dx_a -- the factor the chain rule wants.
    J = torch.stack([_d_dxi(c, axis, first_spatial=1) for axis in range(3)], dim=-1)
    return torch.linalg.inv(J)


def _d_dxi(field: torch.Tensor, axis: int, first_spatial: int) -> torch.Tensor:
    """Central difference along one index axis, unit spacing.

    ``first_spatial`` is where the three grid axes start: 1 for a coordinate array
    ``(S, X, Y, R, 3)``, 2 for a batched field ``(B, S, X, Y, R, C)``.
    """
    ax = axis + first_spatial
    fwd = torch.roll(field, shifts=-1, dims=ax)
    bwd = torch.roll(field, shifts=1, dims=ax)
    out = (fwd - bwd) / 2.0
    # The ends need a one-sided stencil (roll would wrap). Use the THREE-point form,
    # which is second-order like the central difference; the two-point form is only
    # first-order, and with two derivative passes that error compounds -- measured, it
    # tripled the momentum floor from 0.058 to 0.171 once the near-boundary ring was
    # included, making those nodes unusable.
    idx = [slice(None)] * field.ndim
    idx[ax] = 0
    out[tuple(idx)] = (-3 * field.select(ax, 0) + 4 * field.select(ax, 1)
                       - field.select(ax, 2)) / 2.0
    idx[ax] = -1
    out[tuple(idx)] = (3 * field.select(ax, -1) - 4 * field.select(ax, -2)
                       + field.select(ax, -3)) / 2.0
    return out


def gradient(field: torch.Tensor, inv_J: torch.Tensor) -> torch.Tensor:
    """Physical gradient of a (B, S, nx, ny, nr, C) field -> (B, S, nx, ny, nr, C, 3).

    Chain rule: ``df/dx_a = sum_b (dxi_b/dx_a) df/dxi_b``, and ``inv_J[..., b, a]`` is
    exactly ``dxi_b/dx_a``. Subscripts: n batch, s subdomain, xyz grid, c component,
    a physical direction, b index direction.
    """
    d_xi = torch.stack([_d_dxi(field, a, first_spatial=2) for a in range(3)], dim=-1)
    return torch.einsum("sxyzba,nsxyzcb->nsxyzca", inv_J.to(field.dtype), d_xi)


def viscous_operator(u: torch.Tensor, eta: torch.Tensor, inv_J: torch.Tensor) -> torch.Tensor:
    """A u = -div( 2 eta ( eps(u) - (1/3)(div u) I ) ), TERRA-NG's deviatoric form."""
    grad_u = gradient(u, inv_J)                                  # (B,S,X,Y,R,3 i,3 j)
    eps = 0.5 * (grad_u + grad_u.transpose(-1, -2))
    div_u = grad_u.diagonal(dim1=-2, dim2=-1).sum(-1)            # (B,S,X,Y,R)

    eye = torch.eye(3, dtype=u.dtype, device=u.device)
    # eta is (B,S,X,Y,R); tau is a rank-2 tensor field, so two new axes, not one.
    tau = 2.0 * eta[..., None, None] * (eps - div_u[..., None, None] / 3.0 * eye)

    # -div(tau): differentiate each row, contract over the second index.
    flat = tau.reshape(*tau.shape[:5], 9)
    d_tau = gradient(flat, inv_J).reshape(*tau.shape[:5], 3, 3, 3)  # (..., i, j, direction)
    return -torch.einsum("nsxyzijj->nsxyzi", d_tau)


def momentum_residual(u, p, f_u, eta, inv_J, mask, eps=1e-12):
    """|| A u + grad p - f_u || / || f_u || over the interior -- the true momentum residual.

    Once the network predicts pressure this replaces the ``A u_pred - A u_true`` form:
    ``grad p`` is now available from the prediction, so nothing has to be substituted in
    from the data and the residual is the equation the solver actually poses.
    """
    res = viscous_operator(u, eta, inv_J) + gradient(p[..., None], inv_J)[..., 0, :] - f_u
    m = mask[None, None, ..., None]
    num = torch.sqrt((((res) * m) ** 2).sum(dim=(1, 2, 3, 4, 5)))
    den = torch.sqrt(((f_u * m) ** 2).sum(dim=(1, 2, 3, 4, 5))) + eps
    return (num / den).mean()


def continuity_residual(u, f_p, inv_J, mask, eps=1e-12, subsample=True):
    """|| div u - f_p || / || f_p ||.

    With ``subsample`` the divergence is taken to the coarse pressure grid (the pressure
    nodes are exactly every second velocity node, so this is a stride, not an
    interpolation) and ``f_p`` is the coarse field. Without it, both stay on the velocity
    grid and ``f_p`` must be the fine field ``f_p_fine``.

    The fine form sees 343 nodes per subdomain against 27 -- a 12x larger sample of the
    same constraint. Continuity was the laggard at 5.7x its finite-difference floor while
    momentum sat at 2.3x, and the amount of the domain each term actually looks at is the
    obvious asymmetry between them.
    """
    grad_u = gradient(u, inv_J)
    div_u = grad_u.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None]
    if subsample:
        div_u = div_u[:, :, ::2, ::2, ::2]
    m = mask[None, None, ..., None]
    num = torch.sqrt((((div_u - f_p) * m) ** 2).sum(dim=(1, 2, 3, 4, 5)))
    den = torch.sqrt(((f_p * m) ** 2).sum(dim=(1, 2, 3, 4, 5))) + eps
    return (num / den).mean()


def interior_mask(shape: tuple[int, int, int], passes: int = 2, device=None) -> torch.Tensor:
    """True where ``passes`` central-difference passes are valid.

    Each pass needs a neighbour on both sides, so the margin is one node per pass.
    ``passes=2`` for the momentum residual (grad u, then div of the stress);
    ``passes=1`` for continuity, which is a single divergence.

    Getting this wrong is expensive on a coarse grid: a 5^3 pressure grid with a
    two-node margin leaves exactly ONE node per subdomain -- 10 across the whole shell,
    which is neither a usable metric nor a usable training signal.

    For the coarse continuity mask the relevant question is whether the *fine* node it
    maps to is interior: coarse j sits at fine 2j, valid for 1 <= 2j <= n_fine - 2, which
    is exactly ``[1:-1]`` on the coarse grid.
    """
    m = torch.zeros(shape, dtype=torch.bool, device=device)
    k = max(1, passes)
    m[k:-k, k:-k, k:-k] = True
    return m


def residual_loss(u_pred, u_true, eta, inv_J, mask, eps=1e-12):
    """Relative L2 of ``A u_pred - A u_true`` over the interior -- the momentum residual
    with the exact pressure gradient substituted in (see the module docstring)."""
    a_pred = viscous_operator(u_pred, eta, inv_J)
    a_true = viscous_operator(u_true, eta, inv_J)
    m = mask[None, None, ..., None]
    num = torch.sqrt((((a_pred - a_true) * m) ** 2).sum(dim=(1, 2, 3, 4, 5)))
    den = torch.sqrt(((a_true * m) ** 2).sum(dim=(1, 2, 3, 4, 5))) + eps
    return (num / den).mean()


def validate(verbose=True):
    """Checks the finite-difference operator against a sympy-derived analytic residual.

    A random polynomial velocity and a positive polynomial viscosity are built on the
    real mesh; sympy differentiates them exactly, the FD operator differentiates them
    numerically, and the two are compared on interior nodes.
    """
    import json
    import os

    import sympy as sp
    from terra_data.stokes_symbolic import COORDS, viscous_operator as sym_operator
    from terra_data.generate import random_polynomial

    root = os.path.expanduser("~/stokes_dataset")
    mesh = json.load(open(os.path.join(root, "mesh.json")))
    shape = tuple(mesh["velocity_shape"])
    coords = np.fromfile(os.path.join(root, "coords_velocity.bin"),
                         dtype=np.float64).reshape(*shape, 3)

    rng = np.random.default_rng(0)
    x, y, z = COORDS
    u_sym = sp.Matrix([random_polynomial(rng, 3) for _ in range(3)])
    eta_sym = 1.0 + 3.0 * random_polynomial(rng, 2) ** 2
    a_sym = sym_operator(u_sym, eta_sym)

    pts = coords.reshape(-1, 3)
    ev = lambda e: np.asarray(sp.lambdify(COORDS, e, "numpy")(pts[:, 0], pts[:, 1], pts[:, 2]),
                              dtype=np.float64)
    u = np.stack([np.broadcast_to(ev(u_sym[i]), (len(pts),)) for i in range(3)], -1)
    eta = np.broadcast_to(ev(eta_sym), (len(pts),))
    a_ref = np.stack([np.broadcast_to(ev(a_sym[i]), (len(pts),)) for i in range(3)], -1)

    u_t = torch.as_tensor(np.array(u).reshape(1, *shape, 3), dtype=torch.float64)
    eta_t = torch.as_tensor(np.array(eta).reshape(1, *shape), dtype=torch.float64)
    inv_J = inverse_jacobian(coords)
    a_fd = viscous_operator(u_t, eta_t, inv_J)[0].numpy()

    mask = interior_mask(shape[1:]).numpy()
    ref = a_ref.reshape(*shape, 3)[:, mask]
    fd = a_fd[:, mask]
    rel = np.linalg.norm(fd - ref) / np.linalg.norm(ref)

    if verbose:
        print(f"  interior nodes           {mask.sum()} of {np.prod(shape[1:])} per subdomain")
        print(f"  |A_fd| rms               {np.sqrt((fd**2).mean()):.4g}")
        print(f"  |A_analytic| rms         {np.sqrt((ref**2).mean()):.4g}")
        print(f"  relative difference      {rel:.4f}")
        print("  ->", "FD operator tracks the analytic one"
              if rel < 0.25 else "FD operator does NOT match")
    return rel


def validate_full(verbose=True):
    """Checks the momentum and continuity residuals on real dataset samples.

    With the *true* u, p and f from the generator both residuals should be small -- what
    remains is the finite-difference error, nothing else. If either is O(1) the operator,
    the grid nesting or the scaling convention is wrong.
    """
    import json
    import os

    from terra_data.dataset import StokesDataset

    root = os.path.expanduser("~/stokes_dataset")
    mesh = json.load(open(os.path.join(root, "mesh.json")))
    shape = tuple(mesh["velocity_shape"])
    coords = np.fromfile(os.path.join(root, "coords_velocity.bin"),
                         dtype=np.float64).reshape(*shape, 3)
    inv_J = inverse_jacobian(coords)
    mask = interior_mask(shape[1:])
    mask_c = interior_mask(tuple(mesh["pressure_shape"][1:]), passes=1)

    ds = StokesDataset(root, "test")
    mom, con, con_f = [], [], []
    for i in range(8):
        smp = ds[i]
        t = lambda k: torch.as_tensor(np.array(smp[k]), dtype=torch.float64)[None]
        u, pr, eta = t("u"), t("p_fine")[..., 0], t("eta")[..., 0]
        f_u, f_p = t("f_u"), t("f_p")
        mom.append(float(momentum_residual(u, pr, f_u, eta, inv_J, mask)))
        con.append(float(continuity_residual(u, f_p, inv_J, mask_c)))
        con_f.append(float(continuity_residual(u, t("f_p_fine"), inv_J,
                                               interior_mask(shape[1:], passes=1),
                                               subsample=False)))

    if verbose:
        print(f"  momentum   || A u + grad p - f_u || / || f_u ||  = "
              f"{np.mean(mom):.4f}  (min {min(mom):.4f}, max {max(mom):.4f})")
        print(f"  continuity || div u - f_p || / || f_p ||         = "
              f"{np.mean(con):.4f}  (min {min(con):.4f}, max {max(con):.4f})")
        print(f"  continuity on the FINE grid                      = "
              f"{np.mean(con_f):.4f}  (343 nodes/subdomain vs 27)")
        print("  ->", "both residuals are at finite-difference level"
              if max(np.mean(mom), np.mean(con)) < 0.3 else "SOMETHING IS WRONG")
    return max(np.mean(mom), np.mean(con))


if __name__ == "__main__":
    print("operator vs sympy:")
    ok1 = validate() < 0.25
    print("\nresiduals on true dataset samples:")
    ok2 = validate_full() < 0.3
    raise SystemExit(0 if (ok1 and ok2) else 1)
