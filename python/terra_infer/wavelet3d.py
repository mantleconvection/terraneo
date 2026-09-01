"""Compatibility shim: the model moved to :mod:`terra_infer.operator`.

The file was named for its ancestry -- SAOT's wavelet attention, ported to 3-D --
but every model since the spectral branch landed runs ``--no-wavelet``, and the
current architecture (per-degree spherical mixing plus banded mode-coupling
attention) contains no wavelet at all. Old scripts and the registered solver
name keep working through this module; new code imports from ``operator``.
"""

from .operator import *  # noqa: F401,F403
from .operator import Model, load_state, apply, Block, SpectralMix  # noqa: F401

WaveletAttention3D = SpectralMix
