"""Spherical-harmonic transform on the TERRA shell, and an SFNO-style spectral branch.

SAOT's global branch is an AFNO block: FFT the field, mix channels with weights that
carry no mode index, inverse FFT. That is the only part of SAOT that can run at another
resolution -- its weights act on *modes*, and refining the mesh adds modes without
changing the weights. The wavelet branch has no such property (one DWT level means scale
2h, and h moves with the mesh).

An FFT is wrong here: the domain is ten curved diamonds on a spherical shell, not a
periodic box. The spherical harmonics are the right basis on that geometry, so this is
the same idea with the correct transform -- which is what SFNO does for the sphere.

The mesh is an exact radial extrusion (verified: lateral directions agree to 2e-16
across shells), so one lateral transform serves every radial shell.

Analysis is done by pseudo-inverse rather than quadrature. Nodes shared between diamonds
are stored once per subdomain, so a quadrature rule would silently double-count the
seams; the least-squares inverse handles duplicates and non-uniform sampling correctly
and is exact for band-limited fields.
"""

import numpy as np
import torch
import torch.nn as nn

__all__ = ["real_sph_harm", "build_transform", "node_quadrature", "SphericalBranch"]


def node_quadrature(coords: np.ndarray) -> np.ndarray:
    """Per-node quadrature weights for the stored nodes of a mesh (S, nx, ny, nr, 3).

    A uniform mean over stored nodes is a Monte-Carlo estimate under the *storage*
    measure, and that measure moves with the level: seam nodes shared between
    subdomains are stored once per subdomain (21% of lateral nodes at level 3, 6% at
    level 5), and thin regions like the boundary shells occupy a shrinking index
    fraction under refinement. Weighting each node by its volume element |det J| of
    the index->space map (trapezoid-halved on faces) and dividing by its storage
    multiplicity turns such a mean into a discretisation of the continuum integral,
    which converges under refinement instead of drifting with it.

    Returns weights shaped (S, nx, ny, nr), normalised to mean 1.
    """
    c = np.asarray(coords, dtype=np.float64)
    ji = np.gradient(c, axis=1)
    jj = np.gradient(c, axis=2)
    jk = np.gradient(c, axis=3)
    w = np.abs(np.einsum("...i,...i->...", np.cross(ji, jj), jk))
    for ax in (1, 2, 3):
        t = np.ones(c.shape[ax])
        t[0] = t[-1] = 0.5
        sh = [1, 1, 1, 1]
        sh[ax] = c.shape[ax]
        w = w * t.reshape(sh)
    _, inv, cnt = np.unique(np.round(c.reshape(-1, 3), 9), axis=0,
                            return_inverse=True, return_counts=True)
    w = w.reshape(-1) / cnt[inv]
    return (w / w.mean()).reshape(c.shape[:4])


def real_sph_harm(dirs: np.ndarray, lmax: int) -> np.ndarray:
    """Real spherical harmonics up to ``lmax`` at unit vectors ``dirs`` (n, 3).

    Returns (n, (lmax+1)^2), ordered l = 0..lmax and m = -l..l.
    """
    from scipy.special import lpmv

    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    ct = np.cos(theta)

    cols = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            # normalisation sqrt((2l+1)/(4pi) * (l-|m|)!/(l+|m|)!), built as a ratio so
            # the factorials never overflow
            ratio = 1.0
            for k in range(l - am + 1, l + am + 1):
                ratio /= k
            norm = np.sqrt((2 * l + 1) / (4 * np.pi) * ratio)
            p = lpmv(am, l, ct)
            if m == 0:
                cols.append(norm * p)
            elif m > 0:
                cols.append(np.sqrt(2.0) * norm * p * np.cos(m * phi))
            else:
                cols.append(np.sqrt(2.0) * norm * p * np.sin(am * phi))
    return np.stack(cols, axis=1)


def chebyshev(r: np.ndarray, kmax: int) -> np.ndarray:
    """Chebyshev polynomials T_0..T_kmax on radii mapped to [-1, 1]."""
    lo, hi = r.min(), r.max()
    x = 2.0 * (r - lo) / max(hi - lo, 1e-12) - 1.0
    t = np.arccos(np.clip(x, -1.0, 1.0))
    return np.stack([np.cos(k * t) for k in range(kmax + 1)], axis=1)


def build_transform(coords: np.ndarray, lmax: int, kmax: int = 0):
    """Transforms for a mesh (S, nx, ny, nr, 3): lateral, and optionally radial.

    Lateral: ``Y`` (n_lat, n_modes) synthesis and ``A`` (n_modes, n_lat) analysis in
    real spherical harmonics. The lateral directions come from the first radial shell,
    valid because the mesh is a radial extrusion.

    Radial: ``Yr`` (n_rad, kmax+1) and ``Ar`` its pseudo-inverse, in Chebyshev
    polynomials of the radius. Spherical harmonics only span the sphere; without a
    radial basis a branch built on them does no radial mixing at all, and the radial
    direction is where a thin shell has its structure. Together they give a fixed
    (n_modes x kmax+1) coefficient tensor whatever the mesh -- which is what makes the
    branch natively resolution-independent in BOTH directions.
    """
    d = coords[:, :, :, 0, :]
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    Y = real_sph_harm(d.reshape(-1, 3), lmax)
    A = np.linalg.pinv(Y)
    out = [torch.as_tensor(Y, dtype=torch.float32),
           torch.as_tensor(A, dtype=torch.float32)]
    if kmax:
        r = np.linalg.norm(coords[0, 0, 0, :, :], axis=-1)
        Yr = chebyshev(r, kmax)
        out += [torch.as_tensor(Yr, dtype=torch.float32),
                torch.as_tensor(np.linalg.pinv(Yr), dtype=torch.float32)]
    return tuple(out)


class SphericalBranch(nn.Module):
    """Channel mixing in the spherical-harmonic domain, shared across modes.

    The weights are block-diagonal over channels and carry no mode index, exactly as in
    AFNO -- which is what makes the branch resolution-independent: a finer mesh changes
    ``Y`` and ``A`` but not the number of modes the weights see.
    """

    def __init__(self, dim: int, n_blocks: int = 8, factor: int = 1,
                 lmax: int = 0, per_degree: bool = False,
                 couple: bool = False, n_radial: int = 0, couple_band: int = 0):
        super().__init__()
        if dim % n_blocks:
            raise ValueError(f"dim {dim} must be divisible by n_blocks {n_blocks}")
        self.n_blocks, self.bs = n_blocks, dim // n_blocks
        h = self.bs * factor
        # AFNO shares one weight across all modes because its mode count grows with the
        # mesh. Ours does not -- the basis truncation fixes it -- so the weights can
        # depend on the spherical-harmonic DEGREE l without costing invariance.
        #
        # Degree is also the physically correct grouping: for isotropic viscosity the
        # operator is rotationally invariant, so its spectral transfer function depends
        # on l alone and not on the order m (a rotation mixes the m within a fixed l).
        # This is the right symmetry class, not merely extra capacity -- and it is what
        # lets the branch express a per-degree response like the 1/(l(l+1)) of an
        # inverse Laplacian, which mode-shared weights cannot represent at all.
        # Shapes stay as they were without per_degree, so checkpoints trained with the
        # mode-shared branch keep loading.
        self.per_degree = per_degree
        pre = (lmax + 1,) if per_degree else ()
        self.w1 = nn.Parameter(0.02 * torch.randn(*pre, n_blocks, self.bs, h))
        self.b1 = nn.Parameter(torch.zeros(*pre, n_blocks, h))
        self.w2 = nn.Parameter(0.02 * torch.randn(*pre, n_blocks, h, self.bs))
        self.b2 = nn.Parameter(torch.zeros(*pre, n_blocks, self.bs))
        if per_degree:
            deg = torch.cat([torch.full((2 * l + 1,), l, dtype=torch.long)
                             for l in range(lmax + 1)])
            self.register_buffer("deg", deg, persistent=False)
        # Attention among the (l, m) mode tokens. Laterally varying viscosity couples
        # modes (a viscosity component of degree l_eta couples velocity degrees within
        # |l - l'| <= l_eta), and both the mode-shared and per-degree weights are
        # DIAGONAL in mode -- they cannot represent that coupling at all. The tokens
        # are the coefficients themselves, so the token count is fixed by the basis
        # truncation and the branch stays resolution-independent by construction.
        # The output projection is zero-initialised: the model starts as the exact
        # diagonal model and learns off-diagonal coupling only where the data asks.
        # Needs the fixed radial basis: with raw shells the token feature size would
        # move with the mesh.
        self.couple = couple
        if couple:
            if n_radial <= 0:
                raise ValueError("spectral coupling needs radial_modes > 0")
            tok = dim * (n_radial + 1)
            self.c_heads = n_blocks
            self.c_ln = nn.LayerNorm(tok)
            self.c_qkv = nn.Linear(tok, 3 * dim)
            self.c_out = nn.Linear(dim, tok)
            nn.init.zeros_(self.c_out.weight)
            nn.init.zeros_(self.c_out.bias)
            # Optional selection-rule band: a viscosity component of degree l_eta only
            # couples velocity degrees within |l - l'| <= l_eta, so restricting the
            # attention to a degree band removes exactly the couplings the physics
            # forbids -- a hypothesis-space cut, not a capacity cut.
            if couple_band > 0:
                lof = torch.cat([torch.full((2 * l + 1,), l, dtype=torch.long)
                                 for l in range(lmax + 1)])
                self.register_buffer(
                    "c_mask", (lof[:, None] - lof[None, :]).abs() <= couple_band,
                    persistent=False)
            else:
                self.c_mask = None

    def forward(self, f: torch.Tensor, Y: torch.Tensor, A: torch.Tensor,
                Yr: "torch.Tensor | None" = None, Ar: "torch.Tensor | None" = None):
        """``f``: (B, C, n_lat, n_rad) -> same shape.

        With ``Yr``/``Ar`` the radial axis is transformed too, so the coefficient tensor
        has a fixed shape (n_modes, kmax+1) at every resolution and the channel mixing
        -- which carries no mode index -- applies unchanged.
        """
        b, c, n_lat, n_rad = f.shape
        F = torch.einsum("mn,bcnr->bcmr", A.to(f.dtype), f)
        if Ar is not None:
            F = torch.einsum("bcmr,rk->bcmk", F, Ar.to(f.dtype).T)
        nk = F.shape[-1]
        if self.couple:
            n_modes = F.shape[2]
            t = F.permute(0, 2, 1, 3).reshape(b, n_modes, c * nk)
            qkv = self.c_qkv(self.c_ln(t)).reshape(b, n_modes, 3, self.c_heads, -1)
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            a = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=self.c_mask)
            a = a.transpose(1, 2).reshape(b, n_modes, -1)
            F = F + self.c_out(a).reshape(b, n_modes, c, nk).permute(0, 2, 1, 3)
        F = F.reshape(b, self.n_blocks, self.bs, -1, nk)
        if self.per_degree:
            w1, b1 = self.w1[self.deg], self.b1[self.deg]      # (M, K, I, H)
            w2, b2 = self.w2[self.deg], self.b2[self.deg]
            o = torch.relu(torch.einsum("bkinr,nkio->bkonr", F, w1)
                           + b1.permute(1, 2, 0)[None, :, :, :, None])
            o = (torch.einsum("bkinr,nkio->bkonr", o, w2)
                 + b2.permute(1, 2, 0)[None, :, :, :, None])
        else:
            o = torch.relu(torch.einsum("bkinr,kio->bkonr", F, self.w1)
                           + self.b1[None, :, :, None, None])
            o = (torch.einsum("bkinr,kio->bkonr", o, self.w2)
                 + self.b2[None, :, :, None, None])
        o = o.reshape(b, c, -1, nk)
        if Yr is not None:
            o = torch.einsum("bcmk,rk->bcmr", o, Yr.to(f.dtype))
        return torch.einsum("nm,bcmr->bcnr", Y.to(f.dtype), o)


class RadiusAttention(nn.Module):
    """Attention over a FIXED PHYSICAL neighborhood -- the local complement of the
    spectral branch.

    The spectral branch is band-limited: everything above l_max is handled pointwise,
    so local features have no way to communicate. Index-space locality (windows, the
    DWT, k-NN) is the wrong fix -- its physical footprint shrinks with the mesh, which
    is exactly why the wavelet branch broke under transfer. Here node i attends to a
    fixed-size importance sample of the nodes within physical radius r, drawn once per
    mesh with probability proportional to the quadrature weights: the attention output
    is then a self-normalised Monte-Carlo estimate of the kernel integral
    \\int_{|x-y|<r} kappa(f(x), f(y)) f(y) dy, whose meaning does not move with the
    discretisation. Neighbor tables are per-mesh buffers, swapped like the SHT
    matrices; parameters carry no node index. The output projection is
    zero-initialised, and the c_ parameter names put it in the low-LR group.
    """

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        d = dim // 2
        self.heads = n_heads
        self.c_ln = nn.LayerNorm(dim)
        self.c_q = nn.Linear(dim, d)
        self.c_kv = nn.Linear(dim, 2 * d)
        self.c_out = nn.Linear(d, dim)
        nn.init.zeros_(self.c_out.weight)
        nn.init.zeros_(self.c_out.bias)

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """``x``: (B, M, C); ``idx``: (M, K) neighbor sample -> (B, M, C)."""
        b, m, c = x.shape
        t = self.c_ln(x)
        q = self.c_q(t)
        k, v = self.c_kv(t).chunk(2, dim=-1)
        kn, vn = k[:, idx], v[:, idx]                       # (B, M, K, d)
        h = self.heads
        dh = q.shape[-1] // h
        qh = q.view(b, m, h, dh)
        kh = kn.view(b, m, -1, h, dh)
        vh = vn.view(b, m, -1, h, dh)
        att = torch.einsum("bmhd,bmkhd->bmkh", qh, kh) / dh ** 0.5
        att = att.softmax(dim=2)
        out = torch.einsum("bmkh,bmkhd->bmhd", att, vh).reshape(b, m, -1)
        return self.c_out(out)


def radius_neighbors(coords: np.ndarray, radius: float, k: int,
                     seed: int = 0) -> np.ndarray:
    """(M, k) neighbor sample for the stored nodes of ``coords`` (S, nx, ny, nr, 3).

    Candidates are the (up to 128) nearest nodes within ``radius``; from those, ``k``
    are importance-sampled proportionally to the quadrature weights, which also sends
    seam duplicates (weight ~ 1/multiplicity) and zero-weight pad copies to
    near-zero selection probability. Fixed seed: the sample is a frozen part of the
    mesh state, identical between training and evaluation.
    """
    from scipy.spatial import cKDTree

    c = np.asarray(coords, dtype=np.float64)
    q = node_quadrature(c).reshape(-1)
    pts = c.reshape(-1, 3)
    m = pts.shape[0]
    cand = min(512, m)
    dist, nb = cKDTree(pts).query(pts, k=cand, distance_upper_bound=radius, workers=-1)
    rng = np.random.default_rng(seed)
    idx = np.empty((m, k), dtype=np.int64)
    for i in range(m):
        good = nb[i][np.isfinite(dist[i])]
        w = q[good]
        s = w.sum()
        if s <= 0:                       # pad-copy query: fall back to uniform
            w = np.ones(len(good)) / len(good)
        else:
            w = w / s
        # with replacement: this is Monte-Carlo importance sampling of the kernel
        # integral, and the candidate set may hold fewer than k nonzero weights
        idx[i] = rng.choice(good, size=k, replace=True, p=w)
    return idx


class SliceAttention(nn.Module):
    """Attention over a fixed number of learned slices, not over mesh nodes.

    Every mesh node is softly assigned to K learnable slices by a pointwise linear map;
    the slices take a weighted MEAN of the node features, attention runs among the K
    slices, and the result is scattered back through the same assignment. This is
    Transolver's Physics-Attention / LANO's agent tokens.

    It is discretisation invariant for the same reason the spectral branch is: the node
    count enters only through a normalised mean, which converges under refinement, and
    every weight is indexed by slice or channel -- never by node. Cost is O(N*K).

    Attending over K tokens also makes softmax affordable, which matters here: the
    linear-attention form has a max/median weight ratio of 3.6 against softmax's ~3200
    on the same tokens, so it cannot produce selective weights at all.
    """

    def __init__(self, dim: int, n_slices: int = 32, n_heads: int = 8):
        super().__init__()
        self.n_slices, self.n_heads = n_slices, n_heads
        self.assign = nn.Linear(dim, n_slices)
        self.qkv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 3))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, q: "torch.Tensor | None" = None) -> torch.Tensor:
        """``x``: (B, N, C) -> (B, N, C). ``q``: optional (N,) quadrature weights.

        With ``q`` the slice pooling becomes a quadrature-weighted mean -- a continuum
        integral rather than a mean under the storage measure, so seam duplicates and
        the level-dependent index fraction of thin regions stop biasing the tokens.
        The scatter back to nodes stays pointwise in the un-weighted assignment.
        """
        import torch.nn.functional as F

        b, n, c = x.shape
        w = torch.softmax(self.assign(x), dim=-1)              # (B, N, K)
        # weighted mean, not sum: a sum would scale with the node count
        pw = w if q is None else w * q.to(x.dtype)[None, :, None]
        s = torch.einsum("bnk,bnc->bkc", pw, x) / (pw.sum(1)[..., None] + 1e-6)

        qkv = self.qkv(s).reshape(b, self.n_slices, 3, self.n_heads, c // self.n_heads)
        q, k, v = (t.transpose(1, 2) for t in (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]))
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, self.n_slices, c)

        return self.proj(torch.einsum("bnk,bkc->bnc", w, a))
