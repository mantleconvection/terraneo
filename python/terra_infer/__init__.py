"""Python side of terra::ml::NeuralSolver (embedded CPython).

TERRA-NG imports this module inside its own process and calls :func:`call` once
per solve. Fields arrive as ``memoryview``s over the solver's own host buffers,
so ``np.frombuffer`` wraps them without copying; whatever you return is copied
straight back into those buffers.

A field is shaped ``(n_subdomains, nx, ny, nr, n_components)`` -- the Kokkos view
shape on the C++ side -- with the components interleaved last. The Stokes system
sends two: ``u`` (3 components, velocity level) and ``p`` (1 component, one
refinement level coarser). Subdomains are independent blocks: nodes shared
between them are duplicated, and the C++ side repairs the disagreement after
unpacking, so a model may treat each subdomain on its own.

To plug in a model, register a function taking and returning
``dict[str, np.ndarray]``::

    @register("my_model")
    def my_model(fields):
        u = fields["u"]              # (n_sd, nx, ny, nr, 3), float32, read-only
        return {"u": ..., "p": ...}  # same shapes, float32

Returned arrays must be float32 and the same shape as the input; the solver
checks and raises if not.
"""

from __future__ import annotations

import os

import numpy as np

MODELS: dict[str, callable] = {}


def register(name):
    """Decorator: makes a function reachable as the solver's ``--neural-solver <name>``."""

    def wrap(fn):
        MODELS[name] = fn
        return fn

    return wrap


def call(model, buffers, shapes):
    """Entry point invoked by terra::ml::NeuralSolver.

    :param model: registered model name.
    :param buffers: ``{name: memoryview}`` over the solver's host buffers.
    :param shapes: ``{name: (n_subdomains, nx, ny, nr, n_components)}``.
    :returns: ``{name: np.ndarray}``, float32, matching shapes.
    """
    if model not in MODELS:
        raise KeyError(f"unknown model {model!r}; registered: {sorted(MODELS)}")

    fields = {
        name: np.frombuffer(buf, dtype=np.float32).reshape(shapes[name])
        for name, buf in buffers.items()
    }

    out = MODELS[model](fields)

    if not isinstance(out, dict):
        raise TypeError(f"model {model!r} returned {type(out).__name__}, expected dict")

    # The solver reads these through the buffer protocol, so they have to be
    # contiguous float32 before they go back.
    return {
        name: np.ascontiguousarray(array, dtype=np.float32)
        for name, array in out.items()
    }


# ---------------------------------------------------------------- built-in models


@register("zero")
def zero_model(fields):
    """Zeros. The neutral answer, and the one that proves the plumbing is honest."""
    return {name: np.zeros_like(array) for name, array in fields.items()}


@register("echo")
def echo_model(fields):
    """The right-hand side, unchanged. Useless as a solver, ideal as a round-trip test."""
    return {name: array.copy() for name, array in fields.items()}


@register("scale")
def scale_model(fields):
    """0.5 * rhs. Deterministic and non-trivial: proves the values coming back are
    the ones Python computed, not a copy that never left."""
    return {name: 0.5 * array for name, array in fields.items()}


@register("wavelet3d")
def wavelet3d_operator(fields):
    """3-D wavelet attention: SAOT's wavelet branch lifted to 3-D, Fourier
    branch and fusion gate removed. Couples all three axes, so unlike 'saot'
    it sees radial structure. See wavelet3d for the details."""
    from . import operator as _w3d

    return _w3d.apply(fields)


@register("saot")
def saot(fields):
    """SAOT spectral-attention operator transformer, applied shell by shell.

    See saot_adapter for the 3-D-to-2-D mapping and its limits. Needs
    $TERRA_SAOT_ROOT; without $TERRA_SAOT_CHECKPOINT the weights are random.
    """
    from . import saot_adapter

    return saot_adapter.apply(fields)


@register("torch")
def torch_model(fields):
    """Runs the TorchScript module named by ``$TERRA_NEURAL_CHECKPOINT``.

    The module receives ``(n_subdomains, n_components, nx, ny, nr)`` -- channels
    first, transposed from the wire layout -- with subdomains as the batch
    dimension, which is what makes a per-subdomain model parallel for free.
    """
    import torch

    global _MODULE, _DEVICE
    if _MODULE is None:
        path = os.environ.get("TERRA_NEURAL_CHECKPOINT")
        if not path:
            raise RuntimeError("model 'torch' needs $TERRA_NEURAL_CHECKPOINT")
        _DEVICE = os.environ.get("TERRA_NEURAL_DEVICE", "cuda")
        _MODULE = torch.jit.load(path, map_location=_DEVICE).eval()

    device = _DEVICE
    out = {}
    for name, array in fields.items():
        x = torch.from_numpy(np.ascontiguousarray(array)).to(device)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        with torch.no_grad():
            y = _MODULE(x)
        out[name] = y.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy()
    return out


_MODULE = None
_DEVICE = "cuda"
