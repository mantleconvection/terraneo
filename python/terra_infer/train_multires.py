r"""Trains one operator on several mesh resolutions at once.

Fixing the token grid to a constant *physical* scale (``--token-grid``) makes the
attention resolution-independent: 1250 tokens whether the mesh is 9^3, 17^3 or 33^3,
because the DWT depth grows with the mesh (1, 2, 3 levels). That removes the largest
obstacle to discretisation invariance but leaves one: at level 5 the reduce/DWT chain
is applied three times while a level-3-trained model only ever learned to apply it
once, so the two extra levels are extrapolation.

Training on more than one resolution at a time removes that too -- the model sees the
chain applied at depth 1 and 2 and has to make the same weights work for both, which is
what should let depth 3 follow. Nothing in the network depends on the discretisation;
only the padded extents, the geometry buffer and the DWT depth do, and ``set_mesh``
swaps those between batches.

Reports the error at every resolution each epoch, since "the same accuracy at level 5
as at level 3" is the actual objective, not the average.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from . import stokes_residual, symmetry
from .train_operator import load_split, mean_free, relative_l2
from .operator import Model


def mesh_of(root):
    mesh = json.load(open(os.path.join(root, "mesh.json")))
    shape = tuple(mesh["velocity_shape"])
    coords = np.fromfile(os.path.join(root, "coords_velocity.bin"),
                         dtype=np.float64).reshape(*shape, 3)
    return shape, coords


def bundle(root, split, le_mean, le_std, limit, dev, aug):
    """Everything needed to train or evaluate on one mesh."""
    shape, coords = mesh_of(root)
    d = load_split(root, split, verbose=False, limit=limit)
    iJ = stokes_residual.inverse_jacobian(coords)
    c = d["eta_mean"][:, None, None, None, None, None]

    x = torch.from_numpy(np.concatenate(
        [d["f_u"], d["f_p_v"], (d["log_eta"] - le_mean) / le_std], -1))
    g = stokes_residual.gradient(torch.from_numpy(d["log_eta"]), iJ)[..., 0, :]
    x = torch.cat([x, (g / (g.std() + 1e-8)).to(torch.float32)], -1)
    dv = stokes_residual.gradient(torch.from_numpy(d["f_u"]), iJ)
    dv = dv.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None]
    x = torch.cat([x, (dv / (dv.std() + 1e-8)).to(torch.float32)], -1)
    del g, dv

    p = d["p"] - d["p"].mean(axis=(1, 2, 3, 4, 5), keepdims=True)
    y = torch.from_numpy(np.concatenate([d["u"] * c, p], -1))
    b = dict(name=os.path.basename(root.rstrip("/")), shape=shape, coords=coords,
             x=x, y=y, fu=torch.from_numpy(d["f_u"] * c),
             fp=torch.from_numpy(d["f_p_v"] * c),
             eta=torch.from_numpy(np.exp(d["log_eta"])[..., 0]),
             s=torch.from_numpy(d["eta_mean"]),
             inv_J=iJ.to(dev),
             mask=stokes_residual.interior_mask(shape[1:], passes=2, device=dev),
             mask_c=stokes_residual.interior_mask(shape[1:], passes=1, device=dev))
    b["group"] = symmetry.SymmetryGroup(coords, device=dev) if aug else None
    return b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", nargs="+", required=True,
                    help="one or more dataset roots; the first sets the log-eta statistics")
    ap.add_argument("--eval-data", nargs="*", default=None,
                    help="roots to report test error on (default: the training roots)")
    ap.add_argument("--out", default=os.path.expanduser("~/w3d_multires.pt"))
    ap.add_argument("--epochs", type=int, default=360)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    
    ap.add_argument("--attention", default="softmax", choices=("linear", "softmax"))
    ap.add_argument("--physics-weight", type=float, default=2.0)
    ap.add_argument("--mean-p-weight", type=float, default=1.0)
    ap.add_argument("--symmetry-aug", action="store_true")
    ap.add_argument("--spherical", type=int, default=0, metavar="LMAX")
    ap.add_argument("--radial-modes", type=int, default=0, metavar="KMAX")
    ap.add_argument("--sph-per-degree", action="store_true")
    ap.add_argument("--sph-couple", action="store_true")
    ap.add_argument("--sph-couple-band", type=int, default=0)
    ap.add_argument("--sph-couple-shared", action="store_true")
    ap.add_argument("--no-wavelet", action="store_true")
    ap.add_argument("--max-train", type=int, nargs="*", default=None,
                    help="per-root cap on training samples")
    ap.add_argument("--max-test", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                       else "cpu")
    caps = args.max_train or []
    caps = list(caps) + [None] * (len(args.data) - len(caps))

    print("loading")
    prim = load_split(args.data[0], "train", verbose=False, limit=caps[0])
    le_mean, le_std = float(prim["log_eta"].mean()), float(prim["log_eta"].std() + 1e-8)
    del prim
    print(f"  log eta (from {os.path.basename(args.data[0])}): "
          f"mean {le_mean:.4f}, std {le_std:.4f}")

    train = [bundle(r, "train", le_mean, le_std, c, dev, args.symmetry_aug)
             for r, c in zip(args.data, caps)]
    ev_roots = args.eval_data if args.eval_data is not None else args.data
    test = [bundle(r, "test", le_mean, le_std, args.max_test, dev, False)
            for r in ev_roots]
    for b in train:
        print(f"  train {b['name']:>12}: {len(b['x']):>4} samples, mesh {b['shape']}")
    for b in test:
        print(f"  test  {b['name']:>12}: {len(b['x']):>4} samples, mesh {b['shape']}")

    net = Model(train[0]["x"].shape[-1], 4, train[0]["shape"][1:], n_hidden=args.hidden,
                n_layers=args.layers, n_heads=args.heads, coords=train[0]["coords"],
                head_mlp=True, attention=args.attention,
                spherical=args.spherical, radial_modes=args.radial_modes,
                per_degree=args.sph_per_degree, sph_couple=args.sph_couple,
                sph_couple_band=args.sph_couple_band,
                sph_couple_shared=args.sph_couple_shared,
                wavelet=not args.no_wavelet).to(dev)
    print(f"  model {sum(p.numel() for p in net.parameters())/1e6:.2f}M params on {dev}")
    # Every mesh-dependent buffer is captured per mesh and swapped between batches:
    # the padded extents, the geometry channels, and for the spherical branch BOTH
    # transforms -- lateral (Y, A) and radial (Yr, Ar); the Chebyshev matrices move
    # with the shell count just as the harmonics move with the lateral nodes.
    for b in train + test:
        net.set_mesh(b["shape"][1:], b["coords"])
        b["mesh_state"] = (net.shape_in, net.pad, net.shape, net.geom,
                           getattr(net, "sht_Y", None), getattr(net, "sht_A", None),
                           getattr(net, "sht_Yr", None), getattr(net, "sht_Ar", None))
        print(f"  mesh {b['name']:>12}: {int(np.prod(b['shape'])):,} nodes")

    def use(b):
        (net.shape_in, net.pad, net.shape, net.geom, Y, A, Yr, Ar) = b["mesh_state"]
        if net.spherical:
            net.sht_Y, net.sht_A = Y, A
            if net.radial_modes:
                net.sht_Yr, net.sht_Ar = Yr, Ar

    x_spec = [(3, True, True), (1, False, True), (1, False, False),
              (3, True, False), (1, False, True)]
    y_spec = [(3, True, True), (1, False, True)]

    def by_spec(t, spec, g, k):
        out, o = [], 0
        for w, vec, odd in spec:
            out.append(g.transform(t[..., o:o + w], k, vec, odd))
            o += w
        return torch.cat(out, dim=-1)

    steps = sum((len(b["x"]) + args.batch_size - 1) // args.batch_size for b in train)
    # Same LR split as train_operator: the coupling attention diverges at the peak
    # LR the rest of the model wants, so it runs at a 10x lower peak.
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
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr,
                                                total_steps=args.epochs * steps)

    def evaluate():
        net.eval()
        out = []
        with torch.no_grad():
            for b in test:
                use(b)
                ru = rp = rm = 0.0
                n = len(b["x"])
                for i in range(0, n, 4):
                    xb = b["x"][i:i+4].to(dev)
                    yb = b["y"][i:i+4].to(dev)
                    pb = net(xb)
                    w = len(xb)
                    ru += relative_l2(pb[..., :3], yb[..., :3]).item() * w
                    rp += relative_l2(mean_free(pb[..., 3:]),
                                      mean_free(yb[..., 3:])).item() * w
                    cu = b["s"][i:i+4].to(dev).view(-1, 1, 1, 1, 1)
                    rm += stokes_residual.momentum_residual(
                        pb[..., :3], pb[..., 3] * cu, b["fu"][i:i+4].to(dev),
                        b["eta"][i:i+4].to(dev), b["inv_J"], b["mask"]).item() * w
                out.append((ru / n, rp / n, rm / n))
        net.train()
        return out

    hdr = "".join(f"{b['name'][-2:]+' u':>9}{b['name'][-2:]+' m':>9}" for b in test)
    print(f"\n{'epoch':>6} {'tr loss':>10}{hdr} {'lr':>9} {'s':>6}")
    best = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        order = []
        for bi, b in enumerate(train):
            perm = torch.randperm(len(b["x"]))
            order += [(bi, perm[i:i + args.batch_size])
                      for i in range(0, len(perm), args.batch_size)]
        np.random.shuffle(order)
        run = seen = 0.0
        for bi, idx in order:
            b = train[bi]
            use(b)
            xb, yb = b["x"][idx].to(dev), b["y"][idx].to(dev)
            eb, fub = b["eta"][idx].to(dev), b["fu"][idx].to(dev)
            fpb = b["fp"][idx].to(dev)
            if b["group"] is not None:
                g = b["group"]
                k = int(torch.randint(len(g), (1,)).item())
                if k:
                    xb, yb = by_spec(xb, x_spec, g, k), by_spec(yb, y_spec, g, k)
                    fub = g.transform(fub, k, True, True)
                    fpb = g.transform(fpb, k, False, True)
                    eb = g.transform(eb, k, False, False)
            pb = net(xb)
            loss = relative_l2(pb[..., :3], yb[..., :3]) + relative_l2(
                mean_free(pb[..., 3:]), mean_free(yb[..., 3:]))
            if args.mean_p_weight:
                pp = mean_free(pb[..., 3:])
                dims = tuple(range(1, pp.ndim))
                loss = loss + args.mean_p_weight * (
                    pp.mean(dim=dims).abs()
                    / (torch.sqrt((mean_free(yb[..., 3:]) ** 2).mean(dim=dims)) + 1e-12)
                ).mean()
            if args.physics_weight:
                cu = b["s"][idx].to(dev).view(-1, 1, 1, 1, 1)
                loss = loss + args.physics_weight * (
                    stokes_residual.momentum_residual(
                        pb[..., :3], pb[..., 3] * cu, fub, eb, b["inv_J"], b["mask"])
                    + stokes_residual.continuity_residual(
                        pb[..., :3], fpb, b["inv_J"], b["mask_c"], subsample=False))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item() * len(idx)
            seen += len(idx)

        res = evaluate()
        row = "".join(f"{u:>9.4f}{m:>9.4f}" for u, _, m in res)
        print(f"{ep:>6} {run/seen:>10.4f}{row} {sched.get_last_lr()[0]:>9.2e} "
              f"{time.time()-t0:>6.1f}", flush=True)
        worst = max(u for u, _, _ in res)      # the objective is the WORST resolution
        if worst < best:
            best = worst
            torch.save({"model": net.state_dict(), "log_eta_mean": le_mean,
                        "log_eta_std": le_std, "hidden": args.hidden,
                        "layers": args.layers, "heads": args.heads,
                        "shape": train[0]["shape"],
                        "attention": args.attention,
                        "spherical": args.spherical,
                        "radial_modes": args.radial_modes,
                        "per_degree": args.sph_per_degree,
                        "sph_couple": args.sph_couple,
                        "sph_couple_band": args.sph_couple_band,
                        "sph_couple_shared": args.sph_couple_shared,
                        "wavelet": not args.no_wavelet,
                        "n_slices": 0,
                        "test_rel_l2": best, "out_channels": 4}, args.out)
    print(f"\nbest worst-resolution u {best:.4f}, saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
