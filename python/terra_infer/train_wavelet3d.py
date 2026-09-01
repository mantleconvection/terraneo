"""Compatibility shim: the training driver moved to :mod:`terra_infer.train_operator`.

``python -m terra_infer.train_wavelet3d`` keeps working for existing batch scripts.
"""

from .train_operator import *  # noqa: F401,F403
from .train_operator import main, load_split, mean_free, relative_l2  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
