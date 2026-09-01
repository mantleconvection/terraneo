r"""SAOT (Spectral-Attention Operator Transformer) as a TERRA-NG neural solver.

Upstream: https://github.com/chenhong-zhou/SAOT -- Zhou, Chen & Yang, "SAOT: An
Enhanced Locality-Aware Spectral Transformer for Solving PDEs" (arXiv 2511.18777).
Clone it and point $TERRA_SAOT_ROOT at the checkout.

\section mapping How a 3-D shell field is fed to a 2-D model

SAOT's structured-mesh model is 2-D: it takes ``(B, H, W, C)`` and returns
``(B, H, W, out)``. TERRA fields are ``(n_subdomains, nx, ny, nr, C)`` on a thick
spherical shell. The mapping used here treats **each (subdomain, radial shell)
pair as one 2-D image**::

    (n_sd, nx, ny, nr, C)  ->  (n_sd * nr, nx, ny, C)

so the lateral diamond patch is the image and the batch runs over subdomains and
radii. This is the only embedding that leaves SAOT unmodified, and it mirrors how
the solar-wind SFNO surrogates treat a spherical shell -- 2-D per shell, with the
radial direction carried outside the model.

What that costs, stated plainly: the model sees no radial coupling at all. Each
shell is predicted independently, so radial derivatives -- which is where most of
the Stokes physics lives -- are invisible to it. The normalised radius is appended
as an input channel so the model at least knows which depth it is looking at, but
that is a label, not a coupling. Making this a real 3-D operator means either
marching radially (autoregressive, as the solar-wind work does) or replacing the
2-D DWT with a 3-D one.

\section state Status

The architecture is instantiated and called; there are **no trained weights**.
Without ``$TERRA_SAOT_CHECKPOINT`` the output is a randomly initialised network's
response and is physically meaningless -- this path exercises the plumbing, not a
solver. Training it is a separate exercise.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

_MODELS: dict[tuple, "object"] = {}


def _install_timm_shim():
    """Provides the three helpers SAOT imports from ``timm.models.layers``.

    timm pulls in torchvision, and the torchvision on this cluster is built for
    torch 2.5 while the installed torch is 2.10 -- importing it dies with
    ``operator torchvision::nms does not exist``. SAOT only uses ``trunc_normal_``
    (which torch ships natively), ``to_2tuple`` and ``DropPath``, so a stub is
    both sufficient and less invasive than forcing a torchvision downgrade on a
    venv that already has a working torch.
    """
    if "timm.models.layers" in sys.modules:
        return

    import torch
    import torch.nn as nn
    from torch.nn.init import trunc_normal_

    def to_2tuple(x):
        return x if isinstance(x, tuple) else (x, x)

    class DropPath(nn.Module):
        """Stochastic depth, per sample. Identity at eval time and for p=0."""

        def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
            super().__init__()
            self.drop_prob = drop_prob
            self.scale_by_keep = scale_by_keep

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            mask = x.new_empty(shape).bernoulli_(keep)
            if keep > 0.0 and self.scale_by_keep:
                mask.div_(keep)
            return x * mask

    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    layers.trunc_normal_ = trunc_normal_
    layers.to_2tuple = to_2tuple
    layers.DropPath = DropPath
    models.layers = layers
    timm.models = models

    sys.modules["timm"] = timm
    sys.modules["timm.models"] = models
    sys.modules["timm.models.layers"] = layers


def _saot_root() -> str:
    root = os.environ.get("TERRA_SAOT_ROOT")
    if not root:
        raise RuntimeError(
            "model 'saot' needs $TERRA_SAOT_ROOT pointing at a clone of "
            "https://github.com/chenhong-zhou/SAOT"
        )
    if not os.path.isdir(os.path.join(root, "model")):
        raise RuntimeError(f"$TERRA_SAOT_ROOT={root!r} does not look like a SAOT checkout")
    return root


def _build(name: str, nx: int, ny: int, in_channels: int, out_channels: int):
    """One SAOT model per (field, lateral shape, channel count)."""
    import torch

    _install_timm_shim()
    root = _saot_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    from model.SAOT_Structured_Mesh_2D import Model  # noqa: E402

    device = os.environ.get("TERRA_NEURAL_DEVICE", "cuda")
    net = Model(
        space_dim=2,          # normalised lateral (i, j)
        fun_dim=in_channels,  # field components + normalised radius
        out_dim=out_channels,
        n_layers=int(os.environ.get("TERRA_SAOT_LAYERS", 4)),
        n_hidden=int(os.environ.get("TERRA_SAOT_HIDDEN", 64)),
        n_head=int(os.environ.get("TERRA_SAOT_HEADS", 4)),
        mlp_ratio=2,
        slice_num=32,
        ref=8,
        unified_pos=False,    # get_grid() calls .cuda() unconditionally; avoid it
        H=nx,
        W=ny,
    ).to(device)

    checkpoint = os.environ.get("TERRA_SAOT_CHECKPOINT")
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        net.load_state_dict(state.get("model", state))
        trained = f"weights from {checkpoint}"
    else:
        trained = "RANDOMLY INITIALISED -- output is not physical"

    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"terra_infer/saot: built {name} {nx}x{ny}, {in_channels}->{out_channels} ch, "
        f"{n_params / 1e6:.2f}M params on {device} ({trained})",
        file=sys.stderr,
        flush=True,
    )
    return net


def _lateral_grid(nx: int, ny: int, batch: int, device, dtype):
    """(B, nx, ny, 2) of normalised lateral indices."""
    import torch

    i = torch.linspace(0, 1, nx, device=device, dtype=dtype).view(1, nx, 1, 1).expand(batch, nx, ny, 1)
    j = torch.linspace(0, 1, ny, device=device, dtype=dtype).view(1, 1, ny, 1).expand(batch, nx, ny, 1)
    return torch.cat((i, j), dim=-1)


def apply(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Runs SAOT on every field, shell by shell. See the module docstring."""
    import torch

    device = os.environ.get("TERRA_NEURAL_DEVICE", "cuda")
    out: dict[str, np.ndarray] = {}

    for name, array in fields.items():
        n_sd, nx, ny, nr, n_comp = array.shape

        # The solver hands us a read-only view of its own buffer; copy so torch
        # never holds a non-writable tensor (it warns, and would be UB if written).
        x = torch.from_numpy(np.array(array, dtype=np.float32, copy=True)).to(device)
        # (n_sd, nx, ny, nr, C) -> (n_sd * nr, nx, ny, C): one image per shell.
        x = x.permute(0, 3, 1, 2, 4).reshape(n_sd * nr, nx, ny, n_comp)

        # Tell the model which depth each image came from. A label, not a coupling.
        radius = torch.linspace(0, 1, nr, device=device, dtype=x.dtype)
        radius = radius.repeat(n_sd).view(n_sd * nr, 1, 1, 1).expand(n_sd * nr, nx, ny, 1)
        fx = torch.cat((x, radius), dim=-1)

        key = (name, nx, ny, n_comp + 1, n_comp)
        if key not in _MODELS:
            _MODELS[key] = _build(name, nx, ny, n_comp + 1, n_comp)
        net = _MODELS[key]

        coords = _lateral_grid(nx, ny, n_sd * nr, device, x.dtype)
        with torch.no_grad():
            y = net(coords, fx)

        y = y.view(n_sd, nr, nx, ny, n_comp).permute(0, 2, 3, 1, 4).contiguous()
        out[name] = y.cpu().numpy().astype(np.float32)

    return out
