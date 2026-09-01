r"""Reader for the manufactured-solution Stokes dataset.

Each sample file is the five fields concatenated, in the order given by
``meta.json``, each ``[n_subdomains, nx, ny, nr, n_components]`` C-contiguous
float32 -- the same layout ``terra::ml::NeuralSolver`` ships to ``terra_infer``, so a
model trained here meets the identical arrangement at inference time.

    from terra_data.dataset import StokesDataset
    ds = StokesDataset("~/stokes_dataset", split="train")
    s = ds[0]
    s["f_u"], s["eta"]   ->  inputs        (velocity grid)
    s["u"], s["p"]       ->  targets       (velocity / pressure grids)
"""

from __future__ import annotations

import json
import os

import numpy as np


class StokesDataset:
    def __init__(self, root: str, split: str = "train", mmap: bool = True):
        self.root = os.path.expanduser(root)
        self.split = split
        with open(os.path.join(self.root, "meta.json")) as fh:
            self.meta = json.load(fh)

        self.shapes, self.offsets, offset = {}, {}, 0
        for field in self.meta["fields"]:
            grid = self.meta[f"{field['grid']}_shape"]
            shape = (*grid, field["components"])
            self.shapes[field["name"]] = shape
            self.offsets[field["name"]] = offset
            offset += int(np.prod(shape))
        self.n_values = offset

        self.files = sorted(
            os.path.join(self.root, split, f)
            for f in os.listdir(os.path.join(self.root, split))
            if f.endswith(".bin")
        )
        self.mmap = mmap

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        raw = (
            np.memmap(path, dtype=np.float32, mode="r")
            if self.mmap
            else np.fromfile(path, dtype=np.float32)
        )
        if raw.size != self.n_values:
            raise ValueError(f"{path}: {raw.size} values, expected {self.n_values}")
        return {
            name: np.asarray(raw[off : off + int(np.prod(self.shapes[name]))]).reshape(self.shapes[name])
            for name, off in self.offsets.items()
        }

    def stats(self, limit=None):
        """Per-field min / max / RMS over the split -- a cheap sanity read on the data."""
        n = len(self) if limit is None else min(limit, len(self))
        acc = {k: [np.inf, -np.inf, 0.0, 0] for k in self.shapes}
        for i in range(n):
            for k, v in self[i].items():
                lo, hi, sq, cnt = acc[k]
                acc[k] = [min(lo, float(v.min())), max(hi, float(v.max())),
                          sq + float(np.sum(np.float64(v) ** 2)), cnt + v.size]
        return {k: {"min": lo, "max": hi, "rms": np.sqrt(sq / cnt)} for k, (lo, hi, sq, cnt) in acc.items()}


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "~/stokes_dataset"
    for split in ("train", "test"):
        ds = StokesDataset(root, split)
        print(f"\n{split}: {len(ds)} samples, {ds.n_values} values each "
              f"({ds.n_values * 4 / 1e6:.2f} MB)")
        for name, shape in ds.shapes.items():
            print(f"    {name:5s} {shape}")
        print("  statistics over the first 100 samples:")
        for name, s in ds.stats(limit=100).items():
            print(f"    {name:5s} min {s['min']:12.4g}  max {s['max']:12.4g}  rms {s['rms']:12.4g}")
