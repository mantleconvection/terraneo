r"""Symbolic form of TERRA-NG's Stokes operator, and its manufactured right-hand side.

The operator is read off the kernel in
``src/terra/fe/wedge/operators/shell/epsilon_divdiv.hpp``, which accumulates

    k * |J| * w * ( 2 * sym_grad_j : grad_u  -  2/3 * div(v) * div(u) )

i.e. the bilinear form

    a(u, v) = \int eta * [ 2 eps(u) : eps(v) - (2/3) (div u)(div v) ] dx

whose strong form is the deviatoric viscous operator

    A u = -div( 2 eta ( eps(u) - (1/3) (div u) I ) ),      eps(u) = (grad u + grad u^T)/2

and the off-diagonal blocks contribute grad(p), so the momentum residual TERRA-NG
poses is

    f = A u + grad(p).

Nothing here is assumed: :func:`validate_against_terra_testcase` re-derives ``f`` for
the manufactured solution hardcoded in ``tests/test_epsilon_divdiv_stokes.cpp`` and
checks it against that file's ``RHSVelocityInterpolator``, symbolically and exactly.
If the operator in the C++ ever changes, that check fails.
"""

from __future__ import annotations

import sympy as sp

x, y, z = sp.symbols("x y z", real=True)
COORDS = (x, y, z)


def sym_grad(u):
    """eps(u) = (grad u + grad u^T) / 2."""
    grad = sp.Matrix(3, 3, lambda i, j: sp.diff(u[i], COORDS[j]))
    return (grad + grad.T) / 2


def divergence(u):
    return sum(sp.diff(u[i], COORDS[i]) for i in range(3))


def deviatoric_stress(u, eta):
    """2 eta ( eps(u) - (1/3)(div u) I ) -- the tensor whose weak form is TERRA-NG's kernel."""
    return 2 * eta * (sym_grad(u) - sp.Rational(1, 3) * divergence(u) * sp.eye(3))


def viscous_operator(u, eta):
    """A u = -div( 2 eta ( eps(u) - (1/3)(div u) I ) ), returned componentwise."""
    tau = deviatoric_stress(u, eta)
    return sp.Matrix([-sum(sp.diff(tau[i, j], COORDS[j]) for j in range(3)) for i in range(3)])


def momentum_rhs(u, p, eta, simplify=False):
    """f_u = A u + grad p.

    ``simplify`` is off by default and the result is left unexpanded. Both are purely
    cosmetic: measured on a degree-4 sample, ``expand`` costs 9.8 s against 0.02 s for
    the raw derivative tree and yields bit-identical values once lambdified. At 1200
    samples that is the difference between three hours and half a minute.
    """
    f = viscous_operator(u, eta) + sp.Matrix([sp.diff(p, c) for c in COORDS])
    return sp.simplify(f) if simplify else f


def continuity_rhs(u, simplify=False):
    """f_p = div u. Zero for a solenoidal field; nonzero manufactured solutions are fine,
    they just make the sample a general operator pair rather than a physical flow."""
    d = divergence(u)
    return sp.simplify(d) if simplify else d


# --------------------------------------------------------------------------- validation

#: The manufactured solution hardcoded in tests/test_epsilon_divdiv_stokes.cpp.
TERRA_TESTCASE = {
    "u": sp.Matrix([-4 * sp.cos(4 * z), 8 * sp.cos(8 * x), -2 * sp.cos(2 * y)]),
    "p": sp.sin(4 * x) * sp.sin(8 * y) * sp.sin(2 * z),
    "eta": 2 + sp.sin(z),
}

#: RHSVelocityInterpolator from the same file, transcribed verbatim.
TERRA_TESTCASE_RHS = sp.Matrix(
    [
        -64 * (sp.sin(z) + 2) * sp.cos(4 * z)
        - 16 * sp.sin(4 * z) * sp.cos(z)
        + 4 * sp.sin(8 * y) * sp.sin(2 * z) * sp.cos(4 * x),
        512 * (sp.sin(z) + 2) * sp.cos(8 * x)
        + 8 * sp.sin(4 * x) * sp.sin(2 * z) * sp.cos(8 * y)
        - 4 * sp.sin(2 * y) * sp.cos(z),
        -8 * (sp.sin(z) + 2) * sp.cos(2 * y)
        + 2 * sp.sin(4 * x) * sp.sin(8 * y) * sp.cos(2 * z),
    ]
)


def validate_against_terra_testcase(verbose=True):
    """Re-derives the test's RHS from the operator and compares, term by term."""
    case = TERRA_TESTCASE
    derived = momentum_rhs(case["u"], case["p"], case["eta"], simplify=True)
    diff = sp.simplify(derived - TERRA_TESTCASE_RHS)

    ok = all(sp.simplify(d) == 0 for d in diff)
    if verbose:
        print("validating the symbolic operator against tests/test_epsilon_divdiv_stokes.cpp")
        print(f"  u   = {case['u'].T}")
        print(f"  p   = {case['p']}")
        print(f"  eta = {case['eta']}")
        print(f"  div u = {continuity_rhs(case['u'], simplify=True)}   "
              "(zero, so this case does NOT exercise the -2/3 div-div term)")
        for i, c in enumerate("xyz"):
            print(f"  f_{c}: derived - reference = {diff[i]}")
        print("  ->", "MATCH: the symbolic operator is TERRA-NG's" if ok else "MISMATCH")
    return ok


def _selftest():
    results = [("reproduces the TERRA-NG test RHS", validate_against_terra_testcase())]

    # A solenoidal field must leave the div-div term inert...
    u_sol = sp.Matrix([sp.sin(y), sp.sin(z), sp.sin(x)])
    results.append(("solenoidal field has div u = 0", continuity_rhs(u_sol, simplify=True) == 0))

    # ...while a compressive one must not, or the 2/3 term is not wired in.
    u_com = sp.Matrix([x**2, sp.Integer(0), sp.Integer(0)])
    with_term = viscous_operator(u_com, sp.Integer(1))
    without = sp.Matrix(
        [-sum(sp.diff((2 * sym_grad(u_com))[i, j], COORDS[j]) for j in range(3)) for i in range(3)]
    )
    results.append(("div-div term changes a compressive field",
                    sp.simplify(with_term - without) != sp.zeros(3, 1)))

    # Constant viscosity, solenoidal u: A u must collapse to -eta * laplacian(u).
    u_l = sp.Matrix([sp.sin(y), sp.sin(z), sp.sin(x)])
    lap = sp.Matrix([-sum(sp.diff(u_l[i], c, 2) for c in COORDS) for i in range(3)])
    results.append(("constant eta, div-free u  ->  -laplacian",
                    sp.simplify(viscous_operator(u_l, sp.Integer(1)) - lap) == sp.zeros(3, 1)))

    print()
    for label, ok in results:
        print(f"  {'ok  ' if ok else 'FAIL'} : {label}")
    return all(ok for _, ok in results)


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
