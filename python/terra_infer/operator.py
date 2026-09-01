r"""3-D wavelet-attention operator for TERRA-NG shell fields.

This is SAOT's wavelet-attention branch lifted to three dimensions, with the
Fourier branch and the gated fusion removed. Reference: Zhou, Chen & Yang,
"SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs"
(AAAI 2026), https://github.com/chenhong-zhou/SAOT. Nothing is imported from
that checkout -- the linear attention and the Haar transforms are reimplemented
here so this module stands alone.

\section why Why the Fourier branch is gone

SAOT's spectral branch is an AFNO-style complex MLP built on ``rfft2``, which
assumes a periodic rectangular grid. On a thick spherical shell that assumption
fails on both axes: laterally the domain is ten curved diamonds with pentagonal
corners, radially it is bounded rather than periodic. It is also 3% of the
module's parameters, and the soft-shrinkage that makes AFNO *adaptive* is not
implemented upstream. The principled replacement would be a spherical harmonic
transform, not an FFT; until that exists there is nothing to gate against, so
the fusion gate goes with it.

\section dwt The 3-D Haar transform

Along each axis a signal splits into lowpass and highpass halves, so in 3-D
there are 2^3 = 8 subbands (LLL, LLH, ... HHH), each at half the extent in every
direction. Channels go up 8x, volume goes down 8x: element count is preserved
exactly and the transform is invertible, which is the whole point -- unlike a
strided convolution it discards nothing.

Paired with a 1x1x1 convolution that first cuts C -> C/8, the composition is a
lossless **8x token reduction at constant channel width**. That is twice the
reduction SAOT gets in 2-D, and it matters more here because attention cost is
what limits volumetric operator learning.

\section subdomains Local transform, global attention

A TERRA-NG field arrives as ten curved diamonds, ``(n_subdomains, nx, ny, nr, C)``. The
wavelet transform has to stay *inside* a subdomain -- it is a strided convolution and
needs a regular grid, and the ten diamonds are not one contiguous block. Attention has
no such constraint: it is permutation-equivariant over tokens and does not care whether
they are laid out contiguously.

So the two are split. ``reduce``/DWT/``filter`` run per subdomain, then every subdomain's
subband tokens are pooled into one sequence and attention runs over all of them at once.
The model therefore sees the whole shell in a single forward pass and can couple across
diamond seams, which a per-subdomain model cannot do at any depth. Token count is
``n_subdomains * N/8`` -- 1250 at level 3, 359k at level 6, both comfortable for linear
attention.

\section radial What this buys over the per-shell adapter

``saot_adapter`` maps each (subdomain, radial shell) to an independent 2-D image,
so the model never sees a radial derivative. Here the whole subdomain volume
``(nx, ny, nr)`` is one sample and the DWT couples all three axes, so radial
structure -- plumes, boundary layers, the viscosity profile -- is inside the
model's receptive field rather than outside it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spherical import RadiusAttention, SliceAttention, SphericalBranch, radius_neighbors

# --------------------------------------------------------------------------- Haar 3-D


def _haar_filters_3d(dtype=torch.float32) -> torch.Tensor:
    """The eight separable Haar analysis filters, shaped ``(8, 1, 2, 2, 2)``.

    Haar is orthogonal, so the same filters serve for synthesis and the
    round trip is the identity -- asserted in :func:`self_test`.
    """
    lo = torch.tensor([1.0, 1.0], dtype=dtype) / np.sqrt(2.0)
    hi = torch.tensor([-1.0, 1.0], dtype=dtype) / np.sqrt(2.0)

    banks = []
    for fx in (lo, hi):
        for fy in (lo, hi):
            for fz in (lo, hi):
                # outer product over the three axes -> a 2x2x2 stencil
                w = fx.view(2, 1, 1) * fy.view(1, 2, 1) * fz.view(1, 1, 2)
                banks.append(w)
    return torch.stack(banks).unsqueeze(1)  # (8, 1, 2, 2, 2)


class DWT3D(nn.Module):
    """One-level 3-D Haar transform: ``(B, C, X, Y, R) -> (B, 8C, X/2, Y/2, R/2)``.

    Implemented as a grouped stride-2 convolution with fixed filters, so it is a
    single cuDNN call with no parameters. Subbands are concatenated on the
    channel axis in the order produced by :func:`_haar_filters_3d`.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("filters", _haar_filters_3d(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        w = self.filters.to(x.dtype).repeat(c, 1, 1, 1, 1)  # (8C, 1, 2, 2, 2)
        y = F.conv3d(x.contiguous(), w, stride=2, groups=c)
        # conv3d with groups=C emits [c0s0..c0s7, c1s0..], i.e. channel-major in c.
        return y.view(b, c, 8, *y.shape[2:]).transpose(1, 2).reshape(b, 8 * c, *y.shape[2:])


class IDWT3D(nn.Module):
    """Inverse of :class:`DWT3D`: ``(B, 8C, X, Y, R) -> (B, C, 2X, 2Y, 2R)``."""

    def __init__(self):
        super().__init__()
        self.register_buffer("filters", _haar_filters_3d(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c8 = x.shape[0], x.shape[1]
        c = c8 // 8
        # back to channel-major so grouped transposed conv sees each channel's 8 subbands
        x = x.view(b, 8, c, *x.shape[2:]).transpose(1, 2).reshape(b, c8, *x.shape[2:])
        w = self.filters.to(x.dtype).repeat(c, 1, 1, 1, 1)  # (8C, 1, 2, 2, 2)
        return F.conv_transpose3d(x.contiguous(), w, stride=2, groups=c)


# --------------------------------------------------------------------- linear attention


def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6):
    """Katharopoulos et al. linear attention with the ``elu(x)+1`` feature map.

    ``V' = phi(Q) (phi(K)^T V) / (phi(Q) . sum phi(K))`` -- the N x N matrix is never
    formed, so cost is O(N D^2) rather than O(N^2 D). Inputs are ``(B, N, H, D)``.
    """
    q = F.elu(q) + 1.0
    k = F.elu(k) + 1.0

    kv = torch.einsum("bnhd,bnhm->bhmd", k, v)
    z = 1.0 / (torch.einsum("bnhd,bhd->bnh", q, k.sum(dim=1)) + eps)
    return torch.einsum("bnhd,bhmd,bnh->bnhm", q, kv, z)


# ------------------------------------------------------------------ wavelet attention 3D


class SpectralMix(nn.Module):
    """Attention on the 3-D Haar subbands of a volumetric feature field.

    Two layouts, selected by ``band_tokens``:

    **channel layout (default).** Subbands live on the channel axis. A token is a spatial
    position carrying all eight bands in its channels, so band identity is dissolved by
    the dense ``qkv`` and cross-scale mixing is implicit in channel mixing. Cheap:
    ``reduce`` (C -> C/8) before the DWT (x8) makes the pair a lossless 8x token
    reduction at constant width.

    **band-token layout.** Each (position, subband) pair is its own token, so attention
    computes an explicit weight between band i at position p and band j at position q --
    genuine cross-scale attention rather than channel mixing. ``reduce`` is dropped so
    tokens keep full width, which costs 8x the token count (1250 -> 10000 per sample at
    level 3). A learned per-band embedding tells each token which scale it came from.

    With ``n_levels > 1`` the transform becomes a **pyramid**: the LLL band is fed back
    through the DWT, classically, and every level's detail bands plus the coarsest LLL
    join the same sequence. Each level costs 1/8 of the previous, so depth is nearly
    free in tokens -- two levels is 1091 tokens per subdomain against 1000 for one,
    a 9% increase for a second scale. A coarse token then attends to a fine token with
    its own weight, which is the property the channel layout cannot express at all.

    ``reduce`` cuts C -> C/8 and the DWT multiplies channels by 8, so the pair is a
    lossless 8x spatial reduction that lands back at C channels. Attention then runs
    on the N/8 subband tokens; the result is inverted back to full resolution,
    concatenated with the untouched input and projected.
    """

    def __init__(self, dim: int, n_heads: int = 8, use_filter: bool = True,
                 band_tokens: bool = False, n_levels: int = 1,
                 attention: str = "linear", spherical: int = 0,
                 wavelet: bool = True, per_degree: bool = False, n_slices: int = 0,
                 sph_couple: bool = False, radial_modes: int = 0,
                 sph_couple_band: int = 0, gno: bool = False):
        super().__init__()
        if attention not in ("linear", "softmax"):
            raise ValueError(f"attention must be linear|softmax, got {attention!r}")
        # Linear attention has no temperature -- scaling phi(q) cancels exactly between
        # numerator and denominator -- and the weights are an inner product of positive
        # bounded vectors, giving a max/median dynamic range of ~3.6 against softmax's
        # ~3200 on the same tokens. That is why the measured attention sits at 0.95-0.99
        # of maximum entropy with no spatial locality, and why neither more kv capacity
        # nor an explicit positional encoding moved it. softmax restores the exponential.
        self.attention = attention
        self.wavelet = wavelet
        self.band_tokens = band_tokens
        self.n_levels = max(1, n_levels)
        if dim % 8 != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by 8 (one channel per subband)")
        if dim % n_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by n_heads {n_heads}")

        self.n_heads = n_heads
        self.dwt = DWT3D()
        self.idwt = IDWT3D()

        self.reduce = nn.Sequential(
            nn.Conv3d(dim, dim // 8, kernel_size=1),
            nn.GroupNorm(1, dim // 8),  # BatchNorm would tie the operator to the batch
        ) if wavelet else None
        self.filter = (
            nn.Sequential(nn.Conv3d(dim, dim, kernel_size=3, padding=1), nn.GroupNorm(1, dim))
            if use_filter and wavelet
            else None
        )

        if band_tokens:
            # No `reduce`: tokens keep full width. The filter is depthwise, because a
            # dense 3x3x3 conv on 8C channels would be 7M parameters on its own.
            self.reduce = None
            self.filter = (nn.Sequential(
                nn.Conv3d(dim * 8, dim * 8, 3, padding=1, groups=dim * 8),
                nn.GroupNorm(1, dim * 8)) if use_filter else None)
            # one embedding per (level, band), plus one for the coarsest LLL
            self.band_emb = nn.Parameter(torch.zeros(self.n_levels * 7 + 1, dim))

        # SAOT's second branch was an FFT block; on a spherical shell the correct
        # transform is the spherical harmonics. Its weights carry no mode index, so it
        # is the resolution-independent half of the block -- the wavelet half is tied to
        # the grid through the DWT.
        self.sph = (SphericalBranch(dim, n_blocks=n_heads, lmax=spherical,
                            per_degree=per_degree, couple=sph_couple,
                            n_radial=radial_modes,
                            couple_band=sph_couple_band) if spherical else None)
        self.merge = nn.Linear(dim * 2, dim) if spherical else None
        # An additive branch: attention over a fixed set of learned slices, which is
        # invariant where node-token attention cannot be.
        self.slice_attn = SliceAttention(dim, n_slices, n_heads) if n_slices else None
        # Local branch: attention over a fixed physical neighborhood (see
        # RadiusAttention) -- communicates the features the band-limited spectral
        # branch cannot, without the index-space locality that broke the wavelets.
        self.gno = RadiusAttention(dim) if gno else None

        self.qkv = (nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 3))
                    if wavelet else None)
        # concat of the untouched input with the wavelet path
        self.proj = (nn.Linear(dim + (dim if band_tokens else dim // 8), dim)
                     if wavelet else None)

    def _spherical(self, x, shape, sht, b, s_dom, n, c):
        nx, ny, nr = shape
        lat = x.reshape(b, s_dom, nx, ny, nr, c).permute(0, 5, 1, 2, 3, 4)
        lat = lat.reshape(b, c, s_dom * nx * ny, nr)
        out = self.sph(lat, *sht)
        out = out.reshape(b, c, s_dom, nx, ny, nr).permute(0, 2, 3, 4, 5, 1)
        return out.reshape(b, s_dom, n, c)

    def forward(self, x: torch.Tensor, shape: tuple[int, int, int],
                sht=None, node_q: "torch.Tensor | None" = None,
                gno_idx: "torch.Tensor | None" = None) -> torch.Tensor:
        """``x``: (B, S, N, C) -- batch, subdomains, nodes, channels. ``shape``: (nx, ny, nr).

        The transform is per subdomain (S folds into the batch); the attention is over
        every subdomain's tokens at once (S folds into the sequence). ``node_q`` are
        optional per-node quadrature weights for the slice pooling.
        """
        b, s_dom, n, c = x.shape
        nx, ny, nr = shape

        slice_out = None
        if self.slice_attn is not None:
            flat = x.reshape(x.shape[0], -1, x.shape[-1])
            slice_out = self.slice_attn(flat, node_q).reshape(x.shape)
        if self.gno is not None:
            flat = x.reshape(x.shape[0], -1, x.shape[-1])
            gno_out = self.gno(flat, gno_idx).reshape(x.shape)
            slice_out = gno_out if slice_out is None else slice_out + gno_out

        if not self.wavelet and self.sph is None:
            # no branches at all: the block degenerates to a pointwise MLP, which is
            # the control for whether the spectral coupling does anything
            return slice_out if slice_out is not None else torch.zeros_like(x)

        if not self.wavelet:
            out = self.merge(torch.cat(
                [x, self._spherical(x, shape, sht, b, s_dom, n, c)], dim=-1))
            return out if slice_out is None else out + slice_out

        vol = x.reshape(b * s_dom, nx, ny, nr, c).permute(0, 4, 1, 2, 3)  # (B*S, C, X, Y, R)

        if self.band_tokens:
            bs = b * s_dom
            cur, details, geom, pads = vol, [], [], []
            for _ in range(self.n_levels):
                pad = tuple(d % 2 for d in cur.shape[2:])           # DWT halves each axis
                if any(pad):
                    cur = F.pad(cur, (0, pad[2], 0, pad[1], 0, pad[0]), mode="replicate")
                pads.append(pad)
                sub = self.dwt(cur)                                  # (BS, 8C, ...) LLL first
                if self.filter is not None:
                    sub = self.filter(sub)
                geom.append(sub.shape[2:])
                details.append(sub[:, c:])                           # the 7 detail bands
                cur = sub[:, :c]                                     # recurse on LLL

            # every level's detail bands, plus the coarsest LLL, in ONE sequence
            seq, e = [], 0
            for lv, det in enumerate(details):
                npos = int(np.prod(geom[lv]))
                t = det.reshape(bs, 7, c, npos).permute(0, 1, 3, 2)
                seq.append((t + self.band_emb[None, e:e + 7, None, :]).reshape(bs, 7 * npos, c))
                e += 7
            npos_c = int(np.prod(geom[-1]))
            seq.append((cur.reshape(bs, c, npos_c).transpose(1, 2)
                        + self.band_emb[None, e, None, :]))
            lens = [t.shape[1] for t in seq]
            tokens = torch.cat(seq, dim=1).reshape(b, -1, c)

            qkv = self.qkv(tokens).reshape(b, -1, 3, self.n_heads, c // self.n_heads)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            att = linear_attention(q, k, v).reshape(bs, sum(lens), c)

            parts, o = [], 0
            for L in lens:
                parts.append(att[:, o:o + L]); o += L
            cur = parts[-1].transpose(1, 2).reshape(bs, c, *geom[-1])
            for lv in range(self.n_levels - 1, -1, -1):
                npos = int(np.prod(geom[lv]))
                det = parts[lv].reshape(bs, 7, npos, c).permute(0, 1, 3, 2).reshape(bs, 7 * c, *geom[lv])
                cur = self.idwt(torch.cat([cur, det], dim=1))
                px, py, pr = pads[lv]
                if px: cur = cur[:, :, :-px]
                if py: cur = cur[:, :, :, :-py]
                if pr: cur = cur[:, :, :, :, :-pr]

            back = cur.permute(0, 2, 3, 4, 1).reshape(b, s_dom, n, c)
            return self.proj(torch.cat([x, back], dim=-1))

        sub = self.dwt(self.reduce(vol))
        if self.filter is not None:
            sub = self.filter(sub)

        bs, cs, sx, sy, sr = sub.shape
        # (B*S, C, n) -> (B, S*n, C): one sequence spanning the whole shell.
        tokens = sub.reshape(b, s_dom, cs, -1).permute(0, 1, 3, 2).reshape(b, -1, cs)

        qkv = self.qkv(tokens).reshape(b, -1, 3, self.n_heads, cs // self.n_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        if self.attention == "softmax":
            # fused SDPA: the N x N matrix is never materialised, so the cost is the
            # 1.68x per block that was estimated, not a memory blow-up
            att = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            attended = att.transpose(1, 2).reshape(b, s_dom, -1, cs)
        else:
            attended = linear_attention(q, k, v).reshape(b, s_dom, -1, cs)

        cur = attended.permute(0, 1, 3, 2).reshape(bs, cs, sx, sy, sr)
        cur = self.idwt(cur)                                    # (B*S, C/8, ...)
        back = cur.permute(0, 2, 3, 4, 1).reshape(b, s_dom, n, c // 8)
        wave = self.proj(torch.cat([x, back], dim=-1))
        if self.sph is None or sht is None:
            return wave

        # the shell is a radial extrusion, so the lateral index is (subdomain, i, j)
        # and the radial index rides along as an independent axis
        lat = x.reshape(b, s_dom, nx, ny, nr, c).permute(0, 5, 1, 2, 3, 4)
        lat = lat.reshape(b, c, s_dom * nx * ny, nr)
        sph = self._spherical(x, shape, sht, b, s_dom, n, c)
        out = self.merge(torch.cat([wave, sph], dim=-1))
        return out if slice_out is None else out + slice_out


class Block(nn.Module):
    """Pre-norm block: wavelet attention, then an MLP, each with a residual."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 2, use_filter: bool = True,
                 band_tokens: bool = False, n_levels: int = 1, attention: str = "linear",
                 spherical: int = 0, wavelet: bool = True,
                 per_degree: bool = False, n_slices: int = 0,
                 sph_couple: bool = False, radial_modes: int = 0,
                 sph_couple_band: int = 0, gno: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SpectralMix(dim, n_heads, use_filter, band_tokens, n_levels,
                                       attention, spherical, wavelet, per_degree,
                                       n_slices, sph_couple, radial_modes,
                                       sph_couple_band, gno)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x, shape, sht=None, node_q=None, gno_idx=None):
        x = x + self.attn(self.ln1(x), shape, sht, node_q, gno_idx)
        return x + self.mlp(self.ln2(x))


class Model(nn.Module):
    """Volumetric operator: ``(B, nx, ny, nr, in_ch) -> (B, nx, ny, nr, out_ch)``.

    Normalised (i, j, k) coordinates are concatenated onto the input, matching how
    SAOT feeds ``space_dim`` coordinates to its structured-mesh model. Odd extents --
    TERRA's are 2^L+1, so always odd -- are padded by one and cropped at the end,
    because the DWT halves each axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shape: tuple[int, int, int],
        n_hidden: int = 64,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: int = 2,
        use_filter: bool = True,
        band_tokens: bool = False,
        n_levels: int = 1,
        attention: str = "linear",
        multigrid: bool = False,
        spherical: int = 0,
        radial_modes: int = 0,
        wavelet: bool = True,
        per_degree: bool = False,
        n_slices: int = 0,
        mass_slices: bool = False,
        sph_couple: bool = False,
        sph_couple_band: int = 0,
        sph_couple_shared: bool = False,
        gno_radius: float = 0.0,
        gno_k: int = 32,
        head_mlp: bool = False,
        coords: "np.ndarray | None" = None,
        n_subdomains: int = 10,
    ):
        super().__init__()
        self.shape_in = tuple(shape)
        self.pad = tuple(s % 2 for s in shape)
        self.shape = tuple(s + p for s, p in zip(shape, self.pad))

        # Geometry channels. Without `coords` the model only gets normalised *index*
        # coordinates, which are identical in all ten diamonds -- a node at (4,4,4) looks
        # the same wherever it is on the sphere. That blindness shows up as subdomain-
        # shaped blocks in the prediction. Real Cartesian positions plus normalised depth
        # tell it where each node actually is.
        n_geom = 4 if coords is not None else 3
        self.lift = nn.Sequential(
            nn.Linear(in_channels + n_geom, n_hidden * 2), nn.GELU(),
            nn.Linear(n_hidden * 2, n_hidden)
        )
        self.multigrid = multigrid
        self.spherical = spherical
        self.radial_modes = radial_modes
        self.wavelet = wavelet
        self.blocks = nn.ModuleList(
            [Block(n_hidden, n_heads, mlp_ratio, use_filter, band_tokens, n_levels,
                   attention, spherical, wavelet,
                   per_degree, n_slices, sph_couple, radial_modes,
                   sph_couple_band, gno_radius > 0)
             for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(n_hidden)
        # A single Linear(H -> 4) already gives each output its own row, so "splitting"
        # it changes nothing -- the sharing is in the trunk, not the head. What does
        # differ is giving each field its own NONLINEAR readout.
        if head_mlp and out_channels == 4:
            self.head = None
            self.head_u = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU(),
                                        nn.Linear(n_hidden, 3))
            self.head_p = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.GELU(),
                                        nn.Linear(n_hidden, 1))
        else:
            self.head = nn.Linear(n_hidden, out_channels)

        if coords is None:
            self.register_buffer("geom", self._grid(self.shape), persistent=False)
        else:
            self.register_buffer("geom", self._physical(coords, self.pad), persistent=False)
        if spherical:
            if coords is None:
                raise ValueError("the spherical branch needs the mesh coordinates")
            t = self._build_sht(coords, self.pad)
            for nm, v in zip(("sht_Y", "sht_A", "sht_Yr", "sht_Ar"), t):
                self.register_buffer(nm, v, persistent=False)
        else:
            self.sht_Y = self.sht_A = None
        if not radial_modes:
            self.sht_Yr = self.sht_Ar = None
        # One coupling shared by every layer: the physical mode-coupling operator does
        # not change with depth, and sharing cuts its parameters by n_layers, which is
        # the overfitting margin the free-form version lost by.
        if sph_couple and sph_couple_shared:
            ref = self.blocks[0].attn.sph
            for blk in self.blocks[1:]:
                s = blk.attn.sph
                s.c_ln, s.c_qkv, s.c_out = ref.c_ln, ref.c_qkv, ref.c_out
        self.mass_slices = mass_slices
        if n_slices and mass_slices:
            if coords is None:
                raise ValueError("mass-weighted slices need the mesh coordinates")
            self.register_buffer("node_q", self._node_q(coords, self.pad),
                                 persistent=False)
        else:
            self.node_q = None
        self.gno_radius, self.gno_k = gno_radius, gno_k
        if gno_radius > 0:
            if coords is None:
                raise ValueError("radius attention needs the mesh coordinates")
            self.register_buffer("gno_idx",
                                 self._gno_idx(coords, self.pad, gno_radius, gno_k),
                                 persistent=False)
        else:
            self.gno_idx = None

    @staticmethod
    def _gno_idx(coords, pad, radius, k):
        """Frozen physical-radius neighbor sample on the padded grid."""
        c = np.asarray(coords, dtype=np.float64)
        px, py, pr = pad
        if px:
            c = np.concatenate([c, c[:, -1:]], axis=1)
        if py:
            c = np.concatenate([c, c[:, :, -1:]], axis=2)
        if pr:
            c = np.concatenate([c, c[:, :, :, -1:]], axis=3)
        return torch.as_tensor(radius_neighbors(c, radius, k), dtype=torch.long)

    @staticmethod
    def _node_q(coords, pad):
        """Quadrature weights for the slice pooling; zero on the replicated pad slices."""
        from .spherical import node_quadrature

        q = node_quadrature(coords)
        px, py, pr = pad
        q = np.pad(q, ((0, 0), (0, px), (0, py), (0, pr)))
        return torch.as_tensor(q.reshape(-1), dtype=torch.float32)

    def _build_sht(self, coords, pad):
        """Synthesis/analysis matrices for the padded lateral grid of this mesh."""
        from .spherical import build_transform

        c = np.asarray(coords, dtype=np.float64)
        px, py, pr = pad
        if px:
            c = np.concatenate([c, c[:, -1:]], axis=1)
        if py:
            c = np.concatenate([c, c[:, :, -1:]], axis=2)
        if pr:
            c = np.concatenate([c, c[:, :, :, -1:]], axis=3)
        return build_transform(c, self.spherical, self.radial_modes)

    def set_mesh(self, shape, coords):
        """Point the model at a different mesh, keeping every weight.

        Nothing learned depends on the discretisation -- only the padded extents, the
        geometry buffer and the DWT depth do. Swapping them between batches is what
        allows one model to be trained on several resolutions at once, which is how the
        variable-depth wavelet chain stops being extrapolation at the finer meshes.
        """
        dev = next(self.parameters()).device
        self.shape_in = tuple(shape)
        self.pad = tuple(s % 2 for s in shape)
        self.shape = tuple(s + p for s, p in zip(shape, self.pad))
        geom = (self._physical(coords, self.pad) if coords is not None
                else self._grid(self.shape))
        self.register_buffer("geom", geom.to(dev), persistent=False)
        if self.spherical:
            t = self._build_sht(coords, self.pad)
            for nm, v in zip(("sht_Y", "sht_A", "sht_Yr", "sht_Ar"), t):
                self.register_buffer(nm, v.to(dev), persistent=False)
        if self.node_q is not None:
            self.register_buffer("node_q", self._node_q(coords, self.pad).to(dev),
                                 persistent=False)
        if self.gno_idx is not None:
            self.register_buffer(
                "gno_idx",
                self._gno_idx(coords, self.pad, self.gno_radius, self.gno_k).to(dev),
                persistent=False)
        return self

    @staticmethod
    def _physical(coords, pad):
        """(S, nx, ny, nr, 3) Cartesian nodes -> (1, S, N, 4) geometry features.

        Channels are x, y, z and the normalised depth (r - r_min)/(r_max - r_min).
        Radius is redundant with the Cartesian triple but the network would have to
        learn a square root to recover it, and depth is the direction the physics
        actually varies along.
        """
        c = torch.as_tensor(np.ascontiguousarray(coords), dtype=torch.float32)
        px, py, pr = pad
        # Replicate the edge so the padded volume matches the field's padding. The
        # padded slice is cropped from the output; it only touches the DWT.
        if px:
            c = torch.cat([c, c[:, -1:]], dim=1)
        if py:
            c = torch.cat([c, c[:, :, -1:]], dim=2)
        if pr:
            c = torch.cat([c, c[:, :, :, -1:]], dim=3)

        r = c.norm(dim=-1, keepdim=True)
        r_min, r_max = float(r.min()), float(r.max())
        depth = (r - r_min) / max(r_max - r_min, 1e-12)
        feats = torch.cat([c, depth], dim=-1)                 # (S, X, Y, R, 4)
        s_dom = feats.shape[0]
        return feats.reshape(1, s_dom, -1, 4)

    @staticmethod
    def _grid(shape):
        nx, ny, nr = shape
        i = torch.linspace(0, 1, nx).view(nx, 1, 1, 1).expand(nx, ny, nr, 1)
        j = torch.linspace(0, 1, ny).view(1, ny, 1, 1).expand(nx, ny, nr, 1)
        k = torch.linspace(0, 1, nr).view(1, 1, nr, 1).expand(nx, ny, nr, 1)
        return torch.cat((i, j, k), dim=-1).reshape(1, 1, nx * ny * nr, 3)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        """``fx``: (B, S, nx, ny, nr, C_in) -> (B, S, nx, ny, nr, C_out).

        With ``multigrid`` set, any finer mesh is handled by restricting the input to
        the trained mesh, running the network there, and prolongating the prediction --
        both parameter-free, both applied ONCE at the model boundary. Restricting inside
        the blocks instead was tried four ways and all of them failed (0.62-3.81),
        because the residual stream then gets restricted and re-interpolated eight times
        while the pointwise layers stay at full resolution.

        The meshes are nested (2^L+1 per axis, coarse coordinates agree with the fine
        ones to 0.0), so restriction is a stride and the network sees exactly the
        discretisation it was trained on.

        A field with no batch axis, (S, nx, ny, nr, C) as the solver sends it, is
        accepted and returned in the same shape.
        """
        squeeze = fx.ndim == 5
        if squeeze:
            fx = fx.unsqueeze(0)

        if self.multigrid and tuple(fx.shape[2:5]) != self.shape_in:
            fine = tuple(fx.shape[2:5])
            st = [(f - 1) // (c - 1) for f, c in zip(fine, self.shape_in)]
            if all(s > 1 and (f - 1) % (c - 1) == 0
                   for s, f, c in zip(st, fine, self.shape_in)):
                coarse = fx[:, :, ::st[0], ::st[1], ::st[2]]
                out = self.forward(coarse if not squeeze else coarse[0])
                if squeeze:
                    out = out.unsqueeze(0)
                o = out.permute(0, 1, 5, 2, 3, 4).reshape(-1, out.shape[-1], *self.shape_in)
                o = F.interpolate(o, size=fine, mode="trilinear", align_corners=True)
                o = o.reshape(out.shape[0], out.shape[1], out.shape[-1], *fine)
                o = o.permute(0, 1, 3, 4, 5, 2)
                return o.squeeze(0) if squeeze else o

        b, s_dom = fx.shape[0], fx.shape[1]

        px, py, pr = self.pad
        fx = F.pad(fx, (0, 0, 0, pr, 0, py, 0, px))

        nx, ny, nr = self.shape
        fx = fx.reshape(b, s_dom, nx * ny * nr, -1)
        geom = self.geom.to(fx.dtype)
        geom = geom.expand(b, s_dom, -1, -1) if geom.shape[1] == 1 else geom.expand(b, -1, -1, -1)
        fx = torch.cat([geom, fx], dim=-1)
        fx = self.lift(fx)

        sht = ((self.sht_Y, self.sht_A, self.sht_Yr, self.sht_Ar)
               if self.spherical else None)
        for block in self.blocks:
            fx = block(fx, self.shape, sht, self.node_q, self.gno_idx)

        fx = self.ln_out(fx)
        fx = (self.head(fx) if self.head is not None
              else torch.cat([self.head_u(fx), self.head_p(fx)], dim=-1))
        fx = fx.reshape(b, s_dom, nx, ny, nr, -1)

        if px:
            fx = fx[:, :, :-px]
        if py:
            fx = fx[:, :, :, :-py]
        if pr:
            fx = fx[:, :, :, :, :-pr]
        return fx.squeeze(0) if squeeze else fx


# ------------------------------------------------------------------------ terra_infer glue

_MODELS: dict[tuple, Model] = {}


def load_state(model, state):
    """Loads weights, tolerating keys the model no longer has but never missing ones.

    Checkpoints written before the wavelet modules stopped being allocated for
    spherical-only models still carry those (never-trained) tensors. Extra keys are
    harmless; a MISSING key would silently leave part of the model at initialisation,
    so that stays an error.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} weights: {missing[:4]}")
    return unexpected


def _load_mesh_coords(shape, n_sd):
    """Reads node coordinates from $TERRA_MESH_COORDS, if it matches this field."""
    path = os.environ.get("TERRA_MESH_COORDS")
    if not path or not os.path.exists(path):
        return None
    c = np.fromfile(path, dtype=np.float64)
    want = n_sd * shape[0] * shape[1] * shape[2] * 3
    if c.size != want:
        print(f"terra_infer/wavelet3d: {path} has {c.size} values, this field wants {want}"
              " -- falling back to index coordinates", file=sys.stderr)
        return None
    return c.reshape(n_sd, *shape, 3)


def _build(name, shape, in_ch, out_ch, n_sd=10, coords=None):
    device = os.environ.get("TERRA_NEURAL_DEVICE", "cuda")
    net = Model(
        in_channels=in_ch,
        out_channels=out_ch,
        shape=shape,
        coords=coords if coords is not None else _load_mesh_coords(shape, n_sd),
        n_hidden=int(os.environ.get("TERRA_W3D_HIDDEN", 64)),
        n_layers=int(os.environ.get("TERRA_W3D_LAYERS", 4)),
        n_heads=int(os.environ.get("TERRA_W3D_HEADS", 8)),
    ).to(device)

    checkpoint = os.environ.get("TERRA_W3D_CHECKPOINT")
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        net.load_state_dict(state.get("model", state))
        provenance = f"weights from {checkpoint}"
    else:
        provenance = "RANDOMLY INITIALISED -- output is not physical"

    net.eval()
    print(
        f"terra_infer/wavelet3d: {name} {shape} {in_ch}->{out_ch} ch, "
        f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params on {device} ({provenance})",
        file=sys.stderr,
        flush=True,
    )
    return net


def apply(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Runs the model on the whole shell at once -- the solver already hands us all
    subdomains, and attention spans them."""
    device = os.environ.get("TERRA_NEURAL_DEVICE", "cuda")
    out = {}

    for name, array in fields.items():
        n_sd, nx, ny, nr, n_comp = array.shape
        x = torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(device)

        key = (name, n_sd, nx, ny, nr, n_comp)
        if key not in _MODELS:
            _MODELS[key] = _build(name, (nx, ny, nr), n_comp, n_comp, n_sd)

        with torch.no_grad():
            y = _MODELS[key](x)
        out[name] = y.cpu().numpy().astype(np.float32)

    return out


def self_test(device="cpu", verbose=True):
    """Checks the properties the module claims: invertibility, shapes, determinism."""
    torch.manual_seed(0)
    results = []

    # 1. The Haar pair must be the identity, or "lossless downsampling" is a lie.
    dwt, idwt = DWT3D().to(device), IDWT3D().to(device)
    z = torch.randn(2, 8, 16, 12, 10, device=device)
    sub = dwt(z)
    err = (idwt(sub) - z).abs().max().item()
    results.append(("DWT3D -> IDWT3D is the identity", err < 1e-5, f"max err {err:.2e}"))
    results.append(
        ("8 subbands, half extent per axis", tuple(sub.shape) == (2, 64, 8, 6, 5), str(tuple(sub.shape)))
    )

    # 2. Element count preserved -> the transform discards nothing.
    results.append(("element count preserved", sub.numel() == z.numel(), f"{sub.numel()} vs {z.numel()}"))

    # 3. Odd extents (TERRA's are 2^L+1) survive the pad/crop round trip.
    for shape, cin, cout in (((9, 9, 9), 3, 3), ((5, 5, 5), 1, 1), ((17, 17, 9), 3, 3)):
        net = Model(cin, cout, shape, n_hidden=32, n_layers=2, n_heads=4).to(device).eval()
        x = torch.randn(2, 10, *shape, cin, device=device)      # batch 2, ten diamonds
        with torch.no_grad():
            y = net(x)
        ok = tuple(y.shape) == (2, 10, *shape, cout)
        results.append((f"shape {shape} {cin}->{cout}", ok, str(tuple(y.shape))))

    # A field with no batch axis, exactly as the solver sends it.
    net = Model(3, 3, (9, 9, 9), n_hidden=32, n_layers=2, n_heads=4).to(device).eval()
    with torch.no_grad():
        y = net(torch.randn(10, 9, 9, 9, 3, device=device))
    results.append(("solver shape (10,9,9,9,3) round-trips", tuple(y.shape) == (10, 9, 9, 9, 3),
                    str(tuple(y.shape))))

    # The point of the change: a perturbation in one subdomain must reach the others.
    net = Model(1, 1, (9, 9, 9), n_hidden=32, n_layers=2, n_heads=4).to(device).eval()
    a = torch.zeros(10, 9, 9, 9, 1, device=device)
    bpt = a.clone(); bpt[0, 4, 4, 4, 0] = 1.0
    with torch.no_grad():
        d = (net(bpt) - net(a)).abs()
    cross = float(d[1:].max())
    results.append((f"perturbation crosses subdomains (max {cross:.2e})", cross > 1e-8, ""))

    # 4. Deterministic in eval mode (no BatchNorm running-stat drift).
    z = torch.randn(2, 10, 9, 9, 9, 1, device=device)
    with torch.no_grad():
        results.append(("eval is deterministic", torch.equal(net(z), net(z)), ""))

    if verbose:
        for label, ok, detail in results:
            print(f"  {'ok  ' if ok else 'FAIL'} : {label}" + (f"   [{detail}]" if detail else ""))
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    raise SystemExit(0 if self_test() else 1)
