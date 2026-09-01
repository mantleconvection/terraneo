"""Exact discrete symmetries of the TERRA-NG spherical shell mesh.

Stokes is linear and isotropic, so a rotation of the sphere maps a solution to
another solution:

    u'(x) = R u(R^-1 x),  p'(x) = p(R^-1 x),  eta'(x) = eta(R^-1 x)
      =>  f_u'(x) = R f_u(R^-1 x),  f_p'(x) = f_p(R^-1 x)

and by linearity the sign flip (u, p, f_u, f_p) -> -(u, p, f_u, f_p) is a
symmetry too (eta is untouched: it is a coefficient, not a solution field).

Only rotations that map the node set *onto itself* are usable -- any other
rotation would need interpolation and would stop being exact.  The mesh is ten
diamonds arranged with five-fold symmetry about the polar axis, so the usable
rotations are the multiples of 2*pi/5 about z.  ``build_rotations`` checks this
numerically rather than assuming it.

Note on duplicates: nodes shared between diamonds are stored once per
subdomain, so the node -> node map is not injective on stored indices.  That is
harmless, because coincident stored nodes always carry the same value (the
fields are continuous and the generator evaluates them at node positions), so
gathering through a non-injective map still reproduces the rotated field.
"""

import numpy as np

__all__ = ["build_rotations", "apply_rotation", "SymmetryGroup"]


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def build_rotations(coords, n_fold=5, tol=1e-9):
    """Rotations about the polar axis that permute the nodes of ``coords`` exactly.

    ``coords`` is (S, nx, ny, nr, 3).  Returns ``(mats, perms)`` where
    ``mats[k]`` is a 3x3 rotation and ``perms[k]`` a flat index array over the
    S*nx*ny*nr nodes such that, for a field ``v`` flattened the same way,

        v_rotated[i] = R @ v[perm[i]]

    reproduces ``v'(x) = R v(R^-1 x)``.  The identity is always element 0.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    tree = cKDTree(pts)
    mats, perms = [], []
    for k in range(n_fold):
        R = _rot_z(2.0 * np.pi * k / n_fold)
        # x_perm[i] = R^-1 x_i = R^T x_i, and (R^T x)_a = sum_b x_b R_ba = (x @ R)_a
        dist, perm = tree.query(pts @ R)
        if dist.max() > tol:
            continue  # not a symmetry of this mesh
        mats.append(R)
        perms.append(perm.astype(np.int64))
    return np.stack(mats), np.stack(perms)


def apply_rotation(field, perm, R=None):
    """Rotates one field.  ``field`` is (..., S, nx, ny, nr[, C]).

    ``R`` is given for vector fields (C == 3) and omitted for scalars.  The
    leading axes (batch) and the trailing channel axis are preserved.
    """
    import torch

    is_torch = isinstance(field, torch.Tensor)
    n = perm.shape[0]
    # Find the node axes: the last axes whose product is n, allowing a trailing channel.
    if field.shape[-1] == 3 and R is not None:
        lead, chan = field.shape[:-5], field.shape[-1]
        flat = field.reshape(*lead, n, chan)
        out = flat[..., perm, :]
        Rt = R.T if not is_torch else R.transpose(-1, -2)
        out = out @ Rt
        return out.reshape(*lead, *field.shape[-5:-1], chan)
    # scalar, with or without a trailing length-1 channel
    if field.shape[-1] == 1:
        lead = field.shape[:-5]
        flat = field.reshape(*lead, n, 1)
        return flat[..., perm, :].reshape(*lead, *field.shape[-5:-1], 1)
    lead = field.shape[:-4]
    flat = field.reshape(*lead, n)
    return flat[..., perm].reshape(*lead, *field.shape[-4:])


class SymmetryGroup:
    """The symmetry group of one grid, ready to apply to batches on device.

    Combines the ``n_fold`` polar rotations with the sign flip, giving
    ``2 * n_rot`` distinct transformations (the identity among them).
    """

    def __init__(self, coords, n_fold=5, device=None, tol=1e-9):
        import torch

        mats, perms = build_rotations(coords, n_fold=n_fold, tol=tol)
        self.n_rot = len(mats)
        self.mats = torch.as_tensor(mats, dtype=torch.float32, device=device)
        self.perms = torch.as_tensor(perms, dtype=torch.long, device=device)
        self.n_nodes = perms.shape[1]

    def __len__(self):
        return 2 * self.n_rot

    def inverse_index(self, k):
        """The element that undoes ``k``.

        Rotation ``r`` about the pole is undone by ``n_rot - r``; the sign flip is
        its own inverse.  Used for test-time augmentation, where each prediction
        has to be mapped back before the average is taken.
        """
        rot, flip = k % self.n_rot, k // self.n_rot
        return flip * self.n_rot + (-rot) % self.n_rot

    def transform(self, field, k, vector, odd):
        """Applies group element ``k`` to one field.

        ``vector`` says the trailing channel axis is a 3-vector that rotates;
        ``odd`` says the field changes sign under the flip (solutions and
        right-hand sides do, the viscosity does not).
        """
        rot, flip = k % self.n_rot, k // self.n_rot
        out = apply_rotation(field, self.perms[rot],
                             self.mats[rot] if vector else None)
        if flip and odd:
            out = -out
        return out
