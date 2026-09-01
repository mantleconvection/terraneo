r"""Generates the manufactured-solution Stokes dataset.

Each sample draws a random polynomial velocity, pressure and viscosity, hands them to
:mod:`terra_data.stokes_symbolic` -- which carries TERRA-NG's operator, validated
against the analytic case in ``tests/test_epsilon_divdiv_stokes.cpp`` -- and gets

    f_u = -div( 2 eta ( eps(u) - (1/3)(div u) I ) ) + grad p
    f_p = div u

**analytically**, by differentiation rather than by applying the discrete operator. The
expressions are then evaluated at the node coordinates exported by
``stokes_dataset_tool --dump-coords``, so the samples land exactly on the solver's mesh.

Design choices worth knowing:

*Degree.* Each sample draws its degree uniformly from ``[1, max_degree]``, so the set
spans smooth-and-easy through wiggly-and-hard rather than sitting at one difficulty.

*Viscosity.* ``eta = eta0 + q(x)^2`` with ``q`` polynomial: positive everywhere by
construction, still a polynomial so ``f`` stays exact and closed-form, and the contrast
is set by rescaling ``q``. A bare polynomial would go negative and a bare ``exp`` would
leave the polynomial class.

*Boundaries.* The velocity is multiplied by ``(r - r_min)(r_max - r)``, so it vanishes
on the CMB and the surface and satisfies the homogeneous Dirichlet condition the solver
imposes. Nothing is left to a boundary lifting.

*What ``f`` is.* This is the pointwise right-hand side. The solver's assembled load
vector is ``M f`` with ``M`` the velocity mass matrix -- see ``stokes_dataset_tool
--validate``, which uses exactly that pairing. Apply the mass matrix if you want to
train against the load vector rather than the strong-form RHS.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import sympy as sp

import sympy as _sp

from .stokes_symbolic import (COORDS, continuity_rhs, momentum_rhs, viscous_operator,
                              x, y, z)


def random_polynomial(rng, degree, scale=1.0):
    """A random polynomial in x, y, z of total degree <= ``degree``.

    Coefficients are Gaussian, scaled by 1/sqrt(#terms) so that the field's magnitude
    does not grow with the degree -- otherwise high-degree samples would dominate.
    """
    terms = []
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            for c in range(degree + 1 - a - b):
                terms.append((a, b, c))
    coeffs = rng.normal(0.0, 1.0, len(terms)) * (scale / np.sqrt(len(terms)))
    return sum(float(k) * x**a * y**b * z**c for k, (a, b, c) in zip(coeffs, terms))


def build_sample(rng, max_degree, r_min, r_max, contrast_range, pts_v):
    """Symbolic (u, p, eta, f_u, f_p) for one sample, plus its metadata.

    ``pts_v`` is needed before the derivation, not after: the viscosity polynomial is
    rescaled by its own maximum over those nodes so that eta_max/eta_min is exactly the
    requested contrast, and ``f`` must be derived from the rescaled eta.
    """
    degree = int(rng.integers(1, max_degree + 1))

    # Velocity: polynomial x radial bubble -> zero on both boundaries.
    # The bubble is written in r^2 rather than r: (r^2 - r_min^2)(r_max^2 - r^2) has the
    # same zeros on the two spheres but stays a polynomial, so u, f_u and f_p remain
    # polynomials and sympy never has to carry a sqrt through the derivatives.
    r2 = x**2 + y**2 + z**2
    bubble = (r2 - r_min**2) * (r_max**2 - r2)
    u = sp.Matrix([bubble * random_polynomial(rng, degree) for _ in range(3)])

    p_raw = random_polynomial(rng, degree)

    # Viscosity: 1 + (c-1) (q/max|q|)^2 is positive, polynomial, and hits the requested
    # contrast exactly -- eta = 1 where q = 0 and eta = c at the node where |q| peaks.
    contrast = float(np.exp(rng.uniform(*np.log(contrast_range))))
    q = random_polynomial(rng, max(1, degree - 1))
    q_max = float(np.abs(lambdify_at(q, pts_v)).max())
    if q_max > 0.0:
        q = q / q_max
    eta = 1.0 + (contrast - 1.0) * q**2

    # Balance the two momentum terms.
    #
    # Drawn independently at the same coefficient scale, u and p enter through operators
    # of very different strength: A u carries TWO derivatives times eta (up to 1e4),
    # grad p carries one at O(1). Measured on the first dataset that left |grad p| at
    # 2.3% of |f_u| and |A u| at 105% -- a 46x imbalance. Pressure was then buried an
    # order of magnitude below the finite-difference floor of the operator itself, so it
    # was not recoverable from f_u at all, and a network trained on it correctly learned
    # to output zero. Real Stokes flows have the two terms in balance.
    #
    # Both terms are measured on a subsample of the nodes and p is rescaled so their
    # ratio is O(1). Everything is linear in p, so scaling p scales grad p exactly.
    a_u = viscous_operator(u, eta)
    grad_p = _sp.Matrix([_sp.diff(p_raw, c) for c in COORDS])

    probe = pts_v[:: max(1, len(pts_v) // 2000)]
    def _rms(vec):
        v = np.stack([lambdify_at(vec[i], probe) for i in range(3)], axis=-1)
        return float(np.sqrt((np.float64(v) ** 2).mean()))

    a_rms, g_rms = _rms(a_u), _rms(grad_p)
    target = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
    scale_p = target * a_rms / g_rms if g_rms > 0 else 1.0

    p = scale_p * p_raw
    f_u = _sp.Matrix([a_u[i] + scale_p * grad_p[i] for i in range(3)])
    f_p = continuity_rhs(u)

    return {
        "u": u, "p": p, "eta": eta, "f_u": f_u, "f_p": f_p,
        "degree": degree, "contrast": contrast, "grad_p_over_Au": target,
    }


def lambdify_at(expr, pts):
    """Evaluates a scalar expression at an (N, 3) array of Cartesian points."""
    fn = sp.lambdify(COORDS, expr, "numpy")
    out = fn(pts[:, 0], pts[:, 1], pts[:, 2])
    return np.broadcast_to(np.asarray(out, dtype=np.float64), (pts.shape[0],)).astype(np.float32)


def evaluate(sample, pts_v, pts_p, normalise=True):
    """Symbolic sample -> the five float32 field arrays, in dataset order.

    With ``normalise`` the sample is rescaled so that ``rms(f_u) = 1``. Because the
    operator is linear in (u, p) at fixed eta, multiplying u, p, f_u and f_p by one
    common factor keeps ``f = A u + grad p`` exactly satisfied -- it only moves where the
    sample sits on the scale. Without it the right-hand side spans four orders of
    magnitude across the set (it tracks eta, correlation 0.92), which no network will
    fit. The factor is returned so the original magnitudes can be recovered.
    """
    u = np.stack([lambdify_at(sample["u"][i], pts_v) for i in range(3)], axis=-1)
    f_u = np.stack([lambdify_at(sample["f_u"][i], pts_v) for i in range(3)], axis=-1)
    eta = lambdify_at(sample["eta"], pts_v)[:, None]
    p = lambdify_at(sample["p"], pts_p)[:, None]
    f_p = lambdify_at(sample["f_p"], pts_p)[:, None]
    # p on the velocity grid as well. The FE pressure space is the coarse one, but the
    # momentum residual needs grad p wherever A u is evaluated, and a model that predicts
    # pressure predicts it there. p is a continuum polynomial, so this is just the same
    # function sampled on the finer node set -- no interpolation involved.
    p_fine = lambdify_at(sample["p"], pts_v)[:, None]
    # f_p on the velocity grid too. It is an INPUT, not just a label: given f_u and eta
    # alone the split between A u and grad p is undetermined -- two different (u, p)
    # pairs give the same f_u, and only the continuity equation picks one. Without it
    # the map the network is asked to learn is not a function.
    f_p_fine = lambdify_at(sample["f_p"], pts_v)[:, None]

    scale = 1.0
    if normalise:
        rms = float(np.sqrt((np.float64(f_u) ** 2).mean()))
        if rms > 0.0:
            scale = 1.0 / rms
            u, p, f_u, f_p, p_fine, f_p_fine = (
                np.float32(a * scale) for a in (u, p, f_u, f_p, p_fine, f_p_fine))

    return (u, p, eta, f_u, f_p, p_fine, f_p_fine), scale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="directory holding mesh.json and coords_*.bin")
    ap.add_argument("--num-train", type=int, default=1000)
    ap.add_argument("--num-test", type=int, default=200)
    ap.add_argument("--max-degree", type=int, default=4)
    ap.add_argument("--contrast-min", type=float, default=1.0)
    ap.add_argument("--contrast-max", type=float, default=1.0e4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0, help="index of this shard")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split both splits across this many independent jobs")
    args = ap.parse_args(argv)

    with open(os.path.join(args.dir, "mesh.json")) as fh:
        mesh = json.load(fh)
    shape_v, shape_p = mesh["velocity_shape"], mesh["pressure_shape"]

    pts_v = np.fromfile(os.path.join(args.dir, "coords_velocity.bin"), dtype=np.float64).reshape(-1, 3)
    pts_p = np.fromfile(os.path.join(args.dir, "coords_pressure.bin"), dtype=np.float64).reshape(-1, 3)
    assert pts_v.shape[0] == int(np.prod(shape_v)), "velocity coords do not match mesh.json"
    assert pts_p.shape[0] == int(np.prod(shape_p)), "pressure coords do not match mesh.json"

    print(f"velocity nodes {pts_v.shape[0]} {shape_v} | pressure nodes {pts_p.shape[0]} {shape_p}")
    if args.num_shards > 1:
        print(f"shard {args.shard} of {args.num_shards}")

    for split_id, (split, count) in enumerate((("train", args.num_train),
                                               ("test", args.num_test))):
        out_dir = os.path.join(args.dir, split)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        degrees, contrasts, scales, ratios = [], [], [], []

        # Every sample gets its own seed derived from (seed, split, index). That makes a
        # sample reproducible from its index alone, so the work shards across independent
        # jobs and an interrupted run can be resumed instead of restarted -- which the
        # previous sequential RNG made impossible.
        indices = range(args.shard, count, args.num_shards)
        for i in indices:
            rng = np.random.default_rng([args.seed, split_id, i])
            sample = build_sample(rng, args.max_degree, mesh["r_min"], mesh["r_max"],
                                  (args.contrast_min, args.contrast_max), pts_v)
            fields, scale = evaluate(sample, pts_v, pts_p)
            scales.append(scale)
            with open(os.path.join(out_dir, f"sample_{i:06d}.bin"), "wb") as fh:
                for arr in fields:
                    fh.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
            degrees.append(sample["degree"])
            contrasts.append(sample["contrast"])
            ratios.append(sample["grad_p_over_Au"])

            if len(scales) % 25 == 0 or i + args.num_shards >= count:
                rate = len(scales) / (time.time() - t0)
                print(f"  {split}: {len(scales)}/{len(range(args.shard, count, args.num_shards))}"
                      f" of this shard  ({rate:.2f} samples/s)", flush=True)

        print(f"  {split}: degrees {np.bincount(degrees, minlength=args.max_degree+1)[1:].tolist()}, "
              f"contrast {min(contrasts):.3g}..{max(contrasts):.3g}, "
              f"norm factor {min(scales):.3g}..{max(scales):.3g}, "
              f"|grad p|/|A u| {min(ratios):.2f}..{max(ratios):.2f}")
        np.save(os.path.join(args.dir, f"{split}_scale_shard{args.shard:03d}.npy"),
                np.array(scales, dtype=np.float64))

    meta = {
        "format": "terra-stokes-manufactured-v1",
        "dtype": "float32",
        "rhs": "analytic (sympy), pointwise; the solver's load vector is M f",
        "normalised": "u, p, f_u, f_p scaled per sample so rms(f_u) = 1; the common factor "
                      "leaves f = A u + grad p exact and is saved in <split>_scale.npy",
        "operator": "f_u = -div(2 eta (eps(u) - (1/3)(div u) I)) + grad p ; f_p = div u",
        "velocity_shape": shape_v,
        "pressure_shape": shape_p,
        "level": mesh["level"],
        "r_min": mesh["r_min"], "r_max": mesh["r_max"],
        "max_degree": args.max_degree,
        "contrast_range": [args.contrast_min, args.contrast_max],
        "grad_p_over_Au": "drawn log-uniform in [0.5, 2.0]; p is rescaled so the two "
                          "momentum terms balance, as they do in real Stokes flow",
        "seed": args.seed,
        "num_train": args.num_train, "num_test": args.num_test,
        "fields": [
            {"name": "u", "grid": "velocity", "components": 3},
            {"name": "p", "grid": "pressure", "components": 1},
            {"name": "eta", "grid": "velocity", "components": 1},
            {"name": "f_u", "grid": "velocity", "components": 3},
            {"name": "f_p", "grid": "pressure", "components": 1},
            {"name": "p_fine", "grid": "velocity", "components": 1},
            {"name": "f_p_fine", "grid": "velocity", "components": 1},
        ],
        "layout": "each field is [n_subdomains, nx, ny, nr, n_components] C-contiguous, "
                  "concatenated per sample in the order above -- the same layout "
                  "terra::ml::NeuralSolver ships to terra_infer",
    }
    # Only one shard writes the manifest, and it writes atomically -- meta.json is what
    # the loader gates on, so a half-written one would be worse than none.
    if args.shard == 0:
        tmp = os.path.join(args.dir, "meta.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(meta, fh, indent=2)
        os.replace(tmp, os.path.join(args.dir, "meta.json"))
        print(f"wrote {os.path.join(args.dir, 'meta.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
