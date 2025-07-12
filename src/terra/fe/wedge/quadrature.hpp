
#pragma once

namespace terra::fe::wedge {

/// Quadrature rules for wedge.
/// Taken from https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_wedge/quadrature_rules_wedge.html
///
/// Reference:
///   Carlos Felippa,
///   A compendium of FEM integration formulas for symbolic work,
///   Engineering Computation,
///   Volume 21, Number 8, 2004, pages 867-890.
///
/// Reference wedge:
///
///   0 <= X
///   0 <= Y
///   X + Y <= 1
///   -1 <= Z <= 1
///

constexpr int                     quad_felippa_1x1_nq                      = 1;
constexpr dense::Vec< double, 3 > quad_felippa_1x1_qp[quad_felippa_1x1_nq] = { { 1.0 / 3.0, 1.0 / 3.0, 0.0 } };
constexpr double                  quad_felippa_1x1_qw[quad_felippa_1x1_nq] = { 1.0 };

constexpr int                     quad_felippa_3x2_nq                      = 6;
constexpr dense::Vec< double, 3 > quad_felippa_3x2_qp[quad_felippa_3x2_nq] = {
    { { 0.6666666666666666, 0.1666666666666667, -0.5773502691896257 } },
    { { 0.1666666666666667, 0.6666666666666666, -0.5773502691896257 } },
    { { 0.1666666666666667, 0.1666666666666667, -0.5773502691896257 } },
    { { 0.6666666666666666, 0.1666666666666667, 0.5773502691896257 } },
    { { 0.1666666666666667, 0.6666666666666666, 0.5773502691896257 } },
    { { 0.1666666666666667, 0.1666666666666667, 0.5773502691896257 } } };
constexpr double quad_felippa_3x2_qw[quad_felippa_3x2_nq] = {
    0.1666666666666667,
    0.1666666666666667,
    0.1666666666666667,
    0.1666666666666667,
    0.1666666666666667,
    0.1666666666666667 };
} // namespace terra::fe::wedge