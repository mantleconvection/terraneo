r"""Trains the 3-D wavelet-attention operator on the manufactured Stokes dataset.

Learns the Stokes solution operator,

    ( f_u , f_p , log eta )  ->  ( u , p )

``f_p`` is not optional. The momentum equation alone, ``A u + grad p = f_u``, does not
determine the split between the two terms: different (u, p) pairs give the same f_u, and
only ``div u = f_p`` selects one. Training without it asked the network to learn a map
that is not a function -- and it responded exactly as it should, by predicting zero for
pressure (relative L2 exactly 1.0) and losing accuracy on velocity too.

with four output channels on the velocity grid. Pressure is predicted there rather than
on the coarser FE pressure grid for two reasons: the pressure nodes are exactly every
second velocity node (verified, 0.000e+00 coordinate difference), so the coarse values
are recovered by subsampling with no interpolation; and the momentum residual needs
``grad p`` wherever ``A u`` is evaluated, which is the fine grid.

Predicting pressure also upgrades the physics term. Without it the momentum residual had
to be written as ``A u_pred - A u_true``, substituting the pressure gradient in from the
data. With ``p`` predicted it becomes the equation the solver actually poses,
``A u + grad p - f_u``, and the continuity equation ``div u - f_p`` joins it.

Three normalisation choices, each for a measured reason:

*Input.* ``f_u`` already has ``rms = 1`` per sample from the generator, so only
``log eta`` needs standardising -- eta itself spans four decades by construction and
would swamp the lift layer.

*Output.* Velocity and pressure need **different** scale factors. ``u ~ f/eta`` so it
carries the mean-eta factor; but once the generator balances the momentum terms,
``|grad p| ~ |f_u| ~ 1`` makes ``p`` itself O(1), and applying eta to it as well
re-injects eta's four-decade range (measured: target spread 145x -> 17876x). So
``c_u = mean(eta)`` and ``c_p = 1``, and the residual divides each field by its own
factor -- multiplying through by ``c_u`` gives
``A(u_pred) + (c_u/c_p) grad(p_pred) - c_u f_u``.

``rms(u)`` spans 9605x across the set, because ``u ~ f/eta``. Measured on the
data, ``log rms(u)`` correlates -0.985 with ``log(mean eta)`` and dividing it out
collapses the spread to 9.4x. So the network predicts ``u * mean(eta)`` and the scaling
is undone at inference: an exactly invertible, physically motivated transform that
turns an unlearnable target range into an ordinary one. Pressure gets the same factor --
measured, ``log rms(p)`` correlates -0.951 with ``log(mean eta)`` and the same rescaling
takes its spread from 1.8e4x to 79x -- because ``p ~ eta * u / L`` scales the same way.

*Pressure is mean-free.* A constant added to ``p`` leaves ``grad p`` unchanged, so the
physics does not determine it; the dataset's polynomial happens to carry one (median
``|mean(p)|/rms(p)`` = 0.57). Both prediction and target are mean-centred per sample in
the pressure data term, so the network is not asked to memorise an arbitrary constant.

*Loss.* Relative L2 per sample rather than MSE. With a target spanning decades, MSE is
simply a weighting by ``|u|^2`` -- it would optimise the low-viscosity samples and
ignore everything else.

*Physics term.* Optionally added: the Stokes momentum residual, evaluated as
``|| A u_pred - A u_true ||`` (see :mod:`terra_infer.stokes_residual` for why that is the
momentum residual with the exact pressure gradient, and needs no predicted pressure).
It penalises the *derivatives* of the error, which plain L2 on u under-weights -- the
same reason HANO carries an H1 loss on multiscale elliptic problems. The relative form
makes it scale-free, so it composes with the data term without tuning units, and it is
invariant to the mean-eta target rescaling because A is linear in u.

A training example is the **whole shell** -- all ten diamonds in one forward pass, the
same tensor the solver hands to inference. The model's attention spans subdomains, so
splitting them into independent volumes would have thrown away lateral coupling for no
reason; at this size the entire domain fits on one GPU many times over.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import stokes_residual, symmetry
from .operator import Model, load_state


def load_coords(root):
    """Node positions from stokes_dataset_tool --dump-coords: (S, nx, ny, nr, 3)."""
    import json
    import os

    root = os.path.expanduser(root)
    mesh = json.load(open(os.path.join(root, "mesh.json")))
    shape = tuple(mesh["velocity_shape"])
    return np.fromfile(os.path.join(root, "coords_velocity.bin"),
                       dtype=np.float64).reshape(*shape, 3)


def load_coords_pressure(root):
    """Node positions of the coarse pressure grid: (S, nx, ny, nr, 3)."""
    import json
    import os

    root = os.path.expanduser(root)
    mesh = json.load(open(os.path.join(root, "mesh.json")))
    shape = tuple(mesh["pressure_shape"])
    return np.fromfile(os.path.join(root, "coords_pressure.bin"),
                       dtype=np.float64).reshape(*shape, 3)


def load_split(root, split, verbose=True, limit=None):
    """Loads a split as whole-shell samples: (n_samples, n_subdomains, nx, ny, nr, C).

    ``limit`` truncates the split. At level 5 a full split is ~40 GB of raw arrays
    before the derived channels, so being able to cap it is what makes the finer
    meshes trainable at all on one node.
    """
    from terra_data.dataset import StokesDataset

    ds = StokesDataset(root, split)
    if limit is not None:
        ds = _Truncated(ds, limit)
    f_u = np.empty((len(ds), *ds.shapes["u"]), dtype=np.float32)
    u = np.empty_like(f_u)
    p = np.empty((len(ds), *ds.shapes["p_fine"]), dtype=np.float32)
    f_p = np.empty((len(ds), *ds.shapes["f_p"]), dtype=np.float32)
    f_p_v = np.empty((len(ds), *ds.shapes["f_p_fine"]), dtype=np.float32)
    log_eta = np.empty((len(ds), *ds.shapes["eta"]), dtype=np.float32)
    eta_mean = np.empty(len(ds), dtype=np.float32)

    for i in range(len(ds)):
        smp = ds[i]
        f_u[i] = smp["f_u"]
        u[i] = smp["u"]
        p[i] = smp["p_fine"]
        f_p[i] = smp["f_p"]
        f_p_v[i] = smp["f_p_fine"]
        le = np.log(np.maximum(smp["eta"], 1e-12))
        log_eta[i] = le
        # One scale per sample -- this is the granularity at which the -0.985
        # correlation between log rms(u) and log(mean eta) was measured.
        eta_mean[i] = np.exp(le.mean())

    if verbose:
        print(f"  {split}: {len(ds)} whole-shell samples of {f_u.shape[1:]}")
    return dict(f_u=f_u, u=u, p=p, f_p=f_p, f_p_v=f_p_v,
                log_eta=log_eta, eta_mean=eta_mean)


class _Truncated:
    """A view of the first ``n`` samples of a dataset."""

    def __init__(self, ds, n):
        self._ds, self._n = ds, min(n, len(ds))
        self.shapes = ds.shapes

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return self._ds[i]


def mean_free(x):
    """Removes the per-sample constant. A constant in p leaves grad p unchanged, so the
    physics does not determine it and the network should not be scored on it."""
    return x - x.mean(dim=(1, 2, 3, 4, 5), keepdim=True)


def relative_l2(pred, target, eps=1e-12):
    """Per-sample ||pred - target|| / ||target||, averaged over the batch."""
    dims = tuple(range(1, pred.ndim))
    num = torch.sqrt(torch.sum((pred - target) ** 2, dim=dims))
    den = torch.sqrt(torch.sum(target**2, dim=dims)) + eps
    return (num / den).mean()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/wavelet3d_velocity.pt"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--physics-weight", type=float, default=0.0,
                    help="weight of the Stokes residual term; 0 disables it")
    ap.add_argument("--mean-free-target", action="store_true",
                    help="store the pressure target mean-free instead of removing the "
                         "mean inside the loss")
    ap.add_argument("--div-fu", action="store_true",
                    help="add div(f_u) as an input channel; pressure satisfies "
                         "laplacian(p) = div(f_u) - div(A u), so it depends on the "
                         "DIVERGENCE of the right-hand side, not on f_u itself")
    ap.add_argument("--head-mlp", action="store_true",
                    help="give velocity and pressure their own 2-layer readout heads")
    ap.add_argument("--mean-p-weight", type=float, default=0.0,
                    help="explicit penalty on the predicted pressure's per-sample mean, "
                         "normalised by the target rms so it is scale-free")
    ap.add_argument("--p-only", action="store_true",
                    help="diagnostic: train pressure alone, no velocity term, no physics")
    ap.add_argument("--grad-log-eta", action="store_true",
                    help="add grad(log eta) as three input channels. The viscosity "
                         "gradient appears EXPLICITLY in the operator: expanding "
                         "A u = -div(2 eta (...)) gives -2 eta div(...) - 2 grad(eta).(...), "
                         "so the model currently has to form it internally from eta")
    ap.add_argument("--momentum-margin", type=int, default=2,
                    help="mask margin for the momentum residual. With second-order end "
                         "stencils margin 1 gives 343 nodes at a LOWER floor (0.046) "
                         "than margin 2's 125 nodes (0.058)")
    ap.add_argument("--fine-continuity", action="store_true",
                    help="evaluate div u = f_p on the velocity grid (343 nodes per "
                         "subdomain) instead of subsampling to the pressure grid (27)")
    ap.add_argument("--band-tokens", action="store_true",
                    help="put the wavelet subbands on the TOKEN axis so attention has "
                         "explicit band-to-band weights, instead of on the channel axis "
                         "where scale dissolves into channel mixing")
    ap.add_argument("--n-levels", type=int, default=1,
                    help="wavelet pyramid depth (band-token layout only); 1 = no pyramid")
    ap.add_argument("--symmetry-aug", action="store_true",
                    help="augment each batch with a random element of the mesh's exact "
                         "symmetry group: the five 2pi/5 polar rotations (which permute "
                         "the nodes with zero interpolation error) times the sign flip "
                         "that linearity of Stokes provides. Group order 10.")
    ap.add_argument("--tta", action="store_true",
                    help="test-time augmentation: average the prediction over the whole "
                         "symmetry group, mapping each member back before averaging. "
                         "Exact for an equivariant model, a free ensemble otherwise.")
    ap.add_argument("--dump-predictions", default=None,
                    help="with --eval-only, write test-set predictions to this .npz")
    ap.add_argument("--eval-only", default=None,
                    help="load this checkpoint, evaluate once and exit (no training)")
    ap.add_argument("--bc-channel", action="store_true",
                    help="mark the Dirichlet boundary (first and last radial shell) with "
                         "an extra input channel")
    ap.add_argument("--bc-weight", type=float, default=0.0,
                    help="penalty on predicted velocity at the no-slip boundary, "
                         "relative to the velocity scale of the sample")
    ap.add_argument("--hard-bc", action="store_true",
                    help="enforce u = 0 on the boundary shells exactly, by masking the "
                         "velocity output instead of penalising it")
    ap.add_argument("--attention", default="linear", choices=("linear", "softmax"),
                    help="softmax restores the exponential the linear form gives up. "
                         "Measured: the linear weights have a max/median dynamic range "
                         "of 3.6 against softmax's 3200, and phi(q)'s magnitude cancels "
                         "exactly, so there is no temperature to sharpen with.")
    ap.add_argument("--spherical", type=int, default=0, metavar="LMAX",
                    help="add an SFNO-style spherical-harmonic branch with modes up to "
                         "this degree (0 = off). This is SAOT's Fourier branch with the "
                         "transform the geometry actually calls for: its weights carry "
                         "no mode index, so it is the resolution-independent half.")
    ap.add_argument("--radial-modes", type=int, default=0, metavar="KMAX",
                    help="Chebyshev modes in radius for the spherical branch. Spherical "
                         "harmonics span only the sphere; without this the branch does "
                         "no radial mixing at all.")
    ap.add_argument("--slices", type=int, default=0, metavar="K",
                    help="attention over K learned slices instead of over mesh "
                         "nodes (Transolver/LANO style). Each node is softly "
                         "assigned to a slice, attention runs among the K slices, "
                         "and the result is scattered back -- invariant, because "
                         "the node count enters only through a normalised mean.")
    ap.add_argument("--sph-couple", action="store_true",
                    help="attention among the (l,m) mode tokens of the spherical "
                         "branch. Laterally varying viscosity couples modes; the "
                         "diagonal (mode-shared or per-degree) weights cannot "
                         "express that. Zero-initialised, so training starts from "
                         "the exact diagonal model. Needs --radial-modes > 0.")
    ap.add_argument("--sph-couple-band", type=int, default=0, metavar="B",
                    help="restrict the mode-coupling attention to degree pairs with "
                         "|l - l'| <= B (the selection rule for viscosity of degree "
                         "<= B). 0 = unrestricted.")
    ap.add_argument("--sph-couple-shared", action="store_true",
                    help="one coupling module shared by all layers instead of one "
                         "per layer (n_layers-fold fewer coupling parameters).")
    ap.add_argument("--gno-radius", type=float, default=0.0, metavar="R",
                    help="add a local branch: attention over an importance sample of "
                         "the nodes within physical radius R (shell units), "
                         "quadrature-weighted so it discretises a fixed kernel "
                         "integral. 0 = off.")
    ap.add_argument("--gno-k", type=int, default=32, metavar="K",
                    help="neighbors sampled per node for --gno-radius.")
    ap.add_argument("--mass-slices", action="store_true",
                    help="pool slice tokens with per-node quadrature weights "
                         "(|det J|, trapezoid faces, seam multiplicity) so the "
                         "token means are continuum integrals rather than means "
                         "under the level-dependent storage measure.")
    ap.add_argument("--sph-per-degree", action="store_true",
                    help="give the spherical branch weights that depend on the "
                         "harmonic degree l. The mode count is fixed by the basis, "
                         "not the mesh, so this keeps invariance; and for isotropic "
                         "viscosity the transfer function depends on l alone.")
    ap.add_argument("--no-wavelet", action="store_true",
                    help="drop the wavelet/attention branch entirely. With the "
                         "spherical branch on, every remaining component is pointwise "
                         "or mode-based, so the operator is natively "
                         "discretisation-invariant -- no restriction, no interpolation.")
    ap.add_argument("--max-train", type=int, default=None,
                    help="cap the number of training samples (memory at level 4/5)")
    ap.add_argument("--max-test", type=int, default=None,
                    help="cap the number of test samples")
    ap.add_argument("--no-coords", action="store_true",
                    help="use index coordinates instead of physical node positions")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    dev = torch.device(args.device)

    print("loading")
    coords_arr = None if args.no_coords else load_coords(args.data)
    tr = load_split(args.data, "train", limit=args.max_train)
    te = load_split(args.data, "test", limit=args.max_test)

    # Standardise log eta on the training split only.
    le_mean, le_std = float(tr["log_eta"].mean()), float(tr["log_eta"].std() + 1e-8)
    print(f"  log eta: mean {le_mean:.4f}, std {le_std:.4f}")

    def to_tensors(split):
        f_u, u, p = split["f_u"], split["u"], split["p"]
        f_p, f_p_v = split["f_p"], split["f_p_v"]
        log_eta, eta_mean = split["log_eta"], split["eta_mean"]
        c = eta_mean[:, None, None, None, None, None]
        # Five input channels: momentum RHS, continuity RHS, and the viscosity.
        x = np.concatenate([f_u, f_p_v, (log_eta - le_mean) / le_std], axis=-1)
        # Four target channels: velocity, then pressure. Both carry the same eta factor
        # (p ~ eta u / L), so the pair stays consistent under the rescaling and the
        # right-hand sides scale with it too: A(cu) + grad(cp) = c f_u.
        # c_u = mean(eta) for velocity; c_p = 1 for pressure (see the module docstring).
        pc = p.copy()
        if args.mean_free_target:
            pc = pc - pc.mean(axis=(1, 2, 3, 4, 5), keepdims=True)
        y = np.concatenate([u * c, pc], axis=-1)
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.from_numpy(f_u * c),
                torch.from_numpy((f_p_v if args.fine_continuity else f_p) * c),
                torch.from_numpy(np.exp(log_eta)[..., 0]), torch.from_numpy(eta_mean))

    x_tr, y_tr, fu_tr, fp_tr, eta_tr, s_tr = to_tensors(tr)
    x_te, y_te, fu_te, fp_te, eta_te, s_te = to_tensors(te)
    # Hold on to only the two arrays the derived channels still need, then drop the
    # rest: the tensors above are a full duplicate of the split, and at level 5 keeping
    # both copies is the difference between fitting in memory and not.
    tr_le, te_le = tr["log_eta"], te["log_eta"]
    tr_fu, te_fu = tr["f_u"], te["f_u"]
    tr = {"eta_mean": tr["eta_mean"], "u": tr["u"], "p": tr["p"]}
    te = {"eta_mean": te["eta_mean"], "u": te["u"], "p": te["p"],
          "log_eta": te_le}          # --dump-predictions still reports the viscosity

    # How each input channel group behaves under the symmetry group: whether its
    # trailing axis is a 3-vector that rotates, and whether it flips sign with the
    # solution. The viscosity is a coefficient, not a solution field, so it is even;
    # so is grad(log eta), which rotates but does not flip.
    x_spec = [(3, True, True), (1, False, True), (1, False, False)]  # f_u, f_p, log eta
    y_spec = [(3, True, True), (1, False, True)]                     # u, p

    if args.grad_log_eta:
        if coords_arr is None:
            raise SystemExit("--grad-log-eta needs the geometry; drop --no-coords")
        iJe = stokes_residual.inverse_jacobian(coords_arr)

        def with_grad_eta(x, log_eta):
            g = stokes_residual.gradient(torch.from_numpy(log_eta), iJe)[..., 0, :]
            g = (g / (g.std() + 1e-8)).to(torch.float32)
            return torch.cat([x, g], dim=-1)

        x_tr = with_grad_eta(x_tr, tr_le)
        x_te = with_grad_eta(x_te, te_le)
        x_spec.append((3, True, False))
        print(f"  + grad(log eta) input channels -> {x_tr.shape[-1]} inputs")

    if args.div_fu:
        if coords_arr is None:
            raise SystemExit("--div-fu needs the geometry; drop --no-coords")
        iJ = stokes_residual.inverse_jacobian(coords_arr)
        def with_div(x, f_u):
            g = stokes_residual.gradient(torch.from_numpy(f_u), iJ)
            d = g.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None].to(torch.float32)
            d = d / (d.std() + 1e-8)
            return torch.cat([x, d], dim=-1)
        x_tr = with_div(x_tr, tr_fu)
        x_te = with_div(x_te, te_fu)
        x_spec.append((1, False, True))
        print(f"  + div(f_u) input channel -> {x_tr.shape[-1]} inputs")
    # The manufactured solutions are no-slip on both radial shells -- |u| rms there is
    # 1e-18, i.e. exactly zero -- so the boundary values are known, not learned.
    if args.bc_channel:
        def with_bc(x):
            b = torch.zeros(*x.shape[:-1], 1, dtype=x.dtype)
            b[:, :, :, :, 0] = 1.0
            b[:, :, :, :, -1] = 1.0
            return torch.cat([x, b], dim=-1)
        x_tr, x_te = with_bc(x_tr), with_bc(x_te)
        x_spec.append((1, False, False))
        print(f"  + Dirichlet boundary marker -> {x_tr.shape[-1]} inputs")

    print(f"  input {tuple(x_tr.shape)} -> target {tuple(y_tr.shape)}  (3 velocity + 1 pressure)")
    print(f"  target rms spread after rescaling: "
          f"{float(y_tr.flatten(1).pow(2).mean(1).sqrt().max() / y_tr.flatten(1).pow(2).mean(1).sqrt().min()):.1f}x")

    shape = tuple(x_tr.shape[2:5])
    coords = coords_arr
    print(f"  geometry: {'normalised index coordinates' if coords is None else 'physical node positions + depth'}")

    net = Model(in_channels=x_tr.shape[-1], out_channels=4, shape=shape,
                n_hidden=args.hidden, n_layers=args.layers, n_heads=args.heads,
                coords=coords, head_mlp=args.head_mlp,
                band_tokens=args.band_tokens, n_levels=args.n_levels,
                attention=args.attention,
                spherical=args.spherical, radial_modes=args.radial_modes,
                wavelet=not args.no_wavelet,
                per_degree=args.sph_per_degree, n_slices=args.slices,
                mass_slices=args.mass_slices, sph_couple=args.sph_couple,
                sph_couple_band=args.sph_couple_band,
                sph_couple_shared=args.sph_couple_shared,
                gno_radius=args.gno_radius, gno_k=args.gno_k).to(dev)
    print(f"  model {sum(p.numel() for p in net.parameters())/1e6:.2f}M params on {dev}")

    # u = 0 on the first and last radial shell. Masking the output enforces it exactly;
    # --bc-weight only pushes towards it.
    shell = torch.ones(1, 1, 1, 1, x_tr.shape[4], 1, device=dev)
    shell[..., 0, :] = 0.0
    shell[..., -1, :] = 0.0

    def forward(xb):
        out = net(xb)
        if args.hard_bc:
            out = torch.cat([out[..., :3] * shell, out[..., 3:]], dim=-1)
        return out

    def predict(xb):
        """The model, plus test-time augmentation when it is asked for.

        Every group member is mapped back before the average -- the output is a vector
        field, so the ten predictions come back in ten different frames.
        """
        pb = forward(xb)
        if not args.tta:
            return pb
        acc = pb
        for k in range(1, len(g_v)):
            pk = forward(by_spec(xb, x_spec, g_v, k))
            acc = acc + by_spec(pk, y_spec, g_v, g_v.inverse_index(k))
        return acc / len(g_v)

    def boundary_error(pred_u, true_u):
        """|| u_pred on the two Dirichlet shells || / || u_true ||, per sample.

        The true velocity vanishes there, so the numerator is the error outright.
        """
        b = torch.cat([pred_u[:, :, :, :, :1], pred_u[:, :, :, :, -1:]], dim=4)
        num = torch.sqrt(torch.sum(b ** 2, dim=tuple(range(1, b.ndim))))
        den = torch.sqrt(torch.sum(true_u ** 2, dim=tuple(range(1, true_u.ndim)))) + 1e-12
        return (num / den).mean()

    if args.hard_bc:
        print("  boundary: u = 0 enforced exactly by masking the velocity output")
    elif args.bc_weight > 0.0:
        print(f"  boundary: u = 0 penalised in the loss, weight {args.bc_weight:g}")

    def by_spec(t, spec, g, k):
        """Applies group element ``k`` channel group by channel group."""
        out, o = [], 0
        for w, vec, odd in spec:
            out.append(g.transform(t[..., o:o + w], k, vec, odd))
            o += w
        return torch.cat(out, dim=-1)

    g_v = g_p = None
    if args.symmetry_aug or args.tta:
        if coords is None:
            raise SystemExit("--symmetry-aug needs the geometry; drop --no-coords")
        g_v = symmetry.SymmetryGroup(coords, device=dev)
        g_p = (g_v if args.fine_continuity
               else symmetry.SymmetryGroup(load_coords_pressure(args.data), device=dev))
        if not args.symmetry_aug:
            print("  test-time augmentation over the same group")
        if g_p.n_rot != g_v.n_rot:
            raise SystemExit("velocity and pressure grids disagree on the symmetry group")
        print(f"  symmetry augmentation: {g_v.n_rot} exact polar rotations x the sign "
              f"flip = group of order {len(g_v)}; verified to leave the Stokes "
              f"residual of the true fields unchanged")

    def augment(xb, yb, fub, fpb, etab):
        """Applies one random element of the symmetry group to a whole batch."""
        k = int(torch.randint(len(g_v), (1,)).item())
        if k == 0:
            return xb, yb, fub, fpb, etab
        return (by_spec(xb, x_spec, g_v, k), by_spec(yb, y_spec, g_v, k),
                None if fub is None else g_v.transform(fub, k, True, True),
                None if fpb is None else g_p.transform(fpb, k, False, True),
                None if etab is None else g_v.transform(etab, k, False, False))

    phys_w = args.physics_weight
    if phys_w > 0.0 and coords is None:
        raise SystemExit("--physics-weight needs the real geometry; drop --no-coords")

    # Build the residual machinery whenever the geometry is available, even at weight 0:
    # watching the residual while it is *not* in the loss is what tells us whether
    # fitting u also fits its derivatives, which is the whole question the term answers.
    if coords is not None:
        inv_J = stokes_residual.inverse_jacobian(coords).to(dev)
        mask = stokes_residual.interior_mask(shape, passes=args.momentum_margin, device=dev)
        # continuity is a single divergence -> one-node margin, not two
        if args.fine_continuity:
            mask_c = stokes_residual.interior_mask(shape, passes=1, device=dev)
        else:
            mask_c = stokes_residual.interior_mask(tuple(fp_tr.shape[2:5]), passes=1, device=dev)
        print(f"  Stokes residual: {'in the loss, weight %g' % phys_w if phys_w > 0 else 'measured only'}"
              f"; momentum {int(mask.sum())}/{int(np.prod(shape))} nodes, "
              f"continuity {int(mask_c.sum())}/{int(np.prod(fp_tr.shape[2:5]))} nodes per subdomain")
    else:
        inv_J = mask = mask_c = None
        print("  Stokes residual: unavailable (no geometry)")

    # The mode-coupling attention diverges at the peak LR that suits the rest of the
    # model (the usual transformer failure at 1e-3); its parameters get a 10x lower
    # peak, everything else is unchanged.
    c_params = [p for n, p in net.named_parameters() if ".c_" in n]
    base_params = [p for n, p in net.named_parameters() if ".c_" not in n]
    if c_params:
        opt = torch.optim.AdamW([{"params": base_params},
                                 {"params": c_params, "lr": args.lr * 0.1}],
                                lr=args.lr, weight_decay=1e-4)
        max_lr = [args.lr, args.lr * 0.1]
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        max_lr = args.lr
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=args.epochs * ((len(x_tr) + args.batch_size - 1) // args.batch_size))

    x_te_d, y_te_d = x_te.to(dev), y_te.to(dev)

    def evaluate():
        """Test data error, and the Stokes residual on the same predictions.

        The residual is reported whether or not it is being trained on -- it is the
        physically meaningful number, and watching it while it is *not* in the loss
        says whether fitting u alone also fits its derivatives.
        """
        net.eval()
        losses, pres, mom, con, bnd = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(x_te_d), 16):
                xb, yb = x_te_d[i:i+16], y_te_d[i:i+16]
                pb = predict(xb)
                bnd.append(boundary_error(pb[..., :3], yb[..., :3]).item() * len(xb))
                losses.append(relative_l2(pb[..., :3], yb[..., :3]).item() * len(xb))
                pp, pt = ((pb[..., 3:], yb[..., 3:]) if args.mean_free_target
                          else (mean_free(pb[..., 3:]), mean_free(yb[..., 3:])))
                pres.append(relative_l2(pp, pt).item() * len(xb))
                if inv_J is not None:
                    eb = eta_te[i:i+16].to(dev)
                    cu = s_te[i:i+16].to(dev).view(-1, 1, 1, 1, 1)
                    mom.append(stokes_residual.momentum_residual(
                        pb[..., :3], pb[..., 3] * cu, fu_te[i:i+16].to(dev), eb,
                        inv_J, mask).item() * len(xb))
                    con.append(stokes_residual.continuity_residual(
                        pb[..., :3], fp_te[i:i+16].to(dev), inv_J, mask_c,
                        subsample=not args.fine_continuity).item() * len(xb))
        net.train()
        n = len(x_te_d)
        nan = float("nan")
        return (sum(losses)/n, sum(pres)/n,
                sum(mom)/n if mom else nan, sum(con)/n if con else nan,
                sum(bnd)/n)

    if args.eval_only:
        ck = torch.load(os.path.expanduser(args.eval_only), map_location=dev,
                        weights_only=False)
        load_state(net, ck["model"])
        u, pr, mo, co, bc = evaluate()
        print(f"\n  checkpoint {args.eval_only}"
              f"{' with test-time augmentation' if args.tta else ''}")
        print(f"  u {u:.4f}  p {pr:.4f}  momentum {mo:.4f}  continuity {co:.4f}  "
              f"boundary {bc:.4f}")

        if args.dump_predictions:
            with torch.no_grad():
                pb = torch.cat([predict(x_te_d[i:i + 16]).cpu()
                                for i in range(0, len(x_te_d), 16)]).numpy()
            c = te["eta_mean"][:, None, None, None, None, None]
            u_pred = pb[..., :3] / c          # undo the mean-eta target scaling
            p_pred = pb[..., 3:]              # pressure carries c_p = 1
            p_true = te["p"]
            p_pred = p_pred - p_pred.mean(axis=(1, 2, 3, 4, 5), keepdims=True)
            p_true = p_true - p_true.mean(axis=(1, 2, 3, 4, 5), keepdims=True)
            d = tuple(range(1, te["u"].ndim))
            np.savez_compressed(
                os.path.expanduser(args.dump_predictions),
                u_true=te["u"], u_pred=u_pred, p_true=p_true, p_pred=p_pred,
                eta=np.exp(te["log_eta"]),
                rel=np.sqrt(((u_pred - te["u"])**2).sum(d)) / np.sqrt((te["u"]**2).sum(d)),
                rel_p=np.sqrt(((p_pred - p_true)**2).sum(d)) / np.sqrt((p_true**2).sum(d)),
                contrast=np.exp(te["log_eta"].max(axis=(1, 2, 3, 4, 5))
                                - te["log_eta"].min(axis=(1, 2, 3, 4, 5))),
                coords=coords_arr)
            print(f"  wrote {args.dump_predictions}")
        return 0

    print(f"\n{'epoch':>6} {'tr data':>11} {'tr phys':>11} {'te u':>9} {'te p':>9} "
          f"{'te mom':>9} {'te cont':>9} {'te bc':>9} {'lr':>9} {'s':>5}")
    best = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        perm = torch.randperm(len(x_tr))
        running, seen = 0.0, 0
        running_data = running_phys = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = x_tr[idx].to(dev, non_blocking=True), y_tr[idx].to(dev, non_blocking=True)
            # The physics tensors are fetched before augmentation so that the whole
            # batch -- inputs, targets and right-hand sides -- moves under one and the
            # same group element.
            need_phys = phys_w > 0.0 and not args.p_only
            eb = fub = fpb = None
            if need_phys:
                eb = eta_tr[idx].to(dev, non_blocking=True)
                fub = fu_tr[idx].to(dev, non_blocking=True)
                fpb = fp_tr[idx].to(dev, non_blocking=True)
            if args.symmetry_aug:
                xb, yb, fub, fpb, eb = augment(xb, yb, fub, fpb, eb)
            pb = forward(xb)
            pp, pt = ((pb[..., 3:], yb[..., 3:]) if args.mean_free_target
                      else (mean_free(pb[..., 3:]), mean_free(yb[..., 3:])))
            data_term = relative_l2(pp, pt)
            if not args.p_only:
                data_term = data_term + relative_l2(pb[..., :3], yb[..., :3])
            if args.mean_p_weight > 0.0:
                dims = tuple(range(1, pp.ndim))
                m = pp.mean(dim=dims).abs()
                ref = torch.sqrt((pt**2).mean(dim=dims)) + 1e-12
                data_term = data_term + args.mean_p_weight * (m / ref).mean()
            if args.bc_weight > 0.0 and not args.p_only and not args.hard_bc:
                data_term = data_term + args.bc_weight * boundary_error(
                    pb[..., :3], yb[..., :3])
            if need_phys:
                # p is stored unscaled while u carries c_u, so the momentum residual
                # multiplied through by c_u wants c_u * p_pred alongside u_pred.
                cu = s_tr[idx].to(dev).view(-1, 1, 1, 1, 1)
                phys_term = (
                    stokes_residual.momentum_residual(
                        pb[..., :3], pb[..., 3] * cu, fub, eb, inv_J, mask)
                    + stokes_residual.continuity_residual(
                        pb[..., :3], fpb, inv_J, mask_c,
                        subsample=not args.fine_continuity))
                loss = data_term + phys_w * phys_term
                running_phys += phys_term.item() * len(idx)
            else:
                loss = data_term
            running_data += data_term.item() * len(idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * len(idx)
            seen += len(idx)

        test_u, test_p, test_mom, test_con, test_bc = evaluate()
        test_loss = test_u
        print(f"{epoch:>6} {running_data/seen:>11.4f} {running_phys/seen:>11.4f} "
              f"{test_u:>9.4f} {test_p:>9.4f} {test_mom:>9.4f} {test_con:>9.4f} "
              f"{test_bc:>9.4f} {sched.get_last_lr()[0]:>9.2e} {time.time()-t0:>5.1f}",
              flush=True)

        if test_loss < best:
            best = test_loss
            torch.save({"model": net.state_dict(),
                        "log_eta_mean": le_mean, "log_eta_std": le_std,
                        "shape": shape, "hidden": args.hidden,
                        "layers": args.layers, "heads": args.heads,
                        "attention": args.attention,
                        "spherical": args.spherical,
                        "radial_modes": args.radial_modes,
                        "wavelet": not args.no_wavelet,
                        "per_degree": args.sph_per_degree,
                        "n_slices": args.slices,
                        "mass_slices": args.mass_slices,
                        "sph_couple": args.sph_couple,
                        "sph_couple_band": args.sph_couple_band,
                        "sph_couple_shared": args.sph_couple_shared,
                        "gno_radius": args.gno_radius,
                        "gno_k": args.gno_k,
                        "test_rel_l2": test_loss, "out_channels": 4}, args.out)

    print(f"\nbest test relative L2 {best:.4f}, saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
