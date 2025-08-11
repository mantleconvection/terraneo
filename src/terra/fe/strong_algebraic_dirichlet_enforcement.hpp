
#pragma once

#include <linalg/vector_q1.hpp>

#include "terra/linalg/operator.hpp"

namespace terra::fe {

/// @brief Helper function to enforce Dirichlet conditions strongly by symmetric elimination for Poisson-like systems.
///
/// Consider the linear (Poisson-like) problem
///
///     Lu = f
///
/// with Dirichlet boundary conditions.
///
/// We approach the elimination as follows (assuming interpolating FE spaces).
///
/// Let A be the "Neumann" operator matrix of L, i.e., we do not treat the boundaries any differently and just execute
/// the volume integrals.
///
/// 1. Interpolate Dirichlet boundary conditions into a vector g.
/// 2. Compute g_A <- A * g.
/// 3. Compute g_D <- diag(A) * g.
/// 4. Set the rhs to b_elim = b - g_A,
///    where b is the assembled rhs vector for the homogeneous problem
///    (the result of evaluating the linear form into a vector or of the matrix-vector product of a vector f_vec where
///    the rhs function f has been interpolated into, and then b = M * f (M being the mass matrix))
/// 5. Set the rhs b_elim at the boundary nodes to g_D, i.e.
///    b_elim <- g_D on the Dirichlet boundary
/// 6. Solve
///         A_elim x = b_elim
///    where A_elim is A, but with all off-diagonal entries in the same row/col as a boundary node set to zero.
///    This feature has to be supplied by the operator implementation.
///    In a matrix-free context, we have to adapt the element matrix A_local accordingly by (symmetrically) zeroing
///    out all the off-diagonals (row and col) that correspond to a boundary node. But we keep the diagonal intact.
///    We still have diag(A) == diag(A_elim).
/// 7. x is the solution of the original problem. No boundary correction should be necessary.
///
template < typename ScalarType, linalg::OperatorLike OperatorType >
void strong_algebraic_dirichlet_enforcement_poisson_like(
    OperatorType&                                   A_neumann,
    OperatorType&                                   A_neumann_diag,
    const linalg::VectorQ1Scalar< ScalarType >&     g,
    linalg::VectorQ1Scalar< ScalarType >&           tmp,
    linalg::VectorQ1Scalar< ScalarType >&           b,
    const grid::Grid4DDataScalar< util::MaskType >& mask_data,
    const util::MaskAndValue&                       dirichlet_boundary_mask )
{
    // g_A <- A * g
    linalg::apply( A_neumann, g, tmp );

    // b_elim <- b - g_A
    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp } );

    // g_D <- diag(A) * g
    linalg::apply( A_neumann_diag, g, tmp );

    // b_elim <- g_D on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old( b.grid_data(), tmp.grid_data(), mask_data, dirichlet_boundary_mask );
}

/// @brief Same as strong_algebraic_dirichlet_enforcement_poisson_like() for homogenous boundary conditions (g = 0).
///
/// Does not require most of the steps since g = g_A = g_D = 0. Still requires solving A_elim x = b_elim after this.
template < typename ScalarType >
void strong_algebraic_homogeneous_dirichlet_enforcement_poisson_like(
    linalg::VectorQ1Scalar< ScalarType >&           b,
    const grid::Grid4DDataScalar< util::MaskType >& mask_data,
    const util::MaskAndValue&                       dirichlet_boundary_mask )
{
    // b_elim <- 0 on the Dirichlet boundary
    kernels::common::assign_masked_else_keep_old( b.grid_data(), 0.0, mask_data, dirichlet_boundary_mask );
}

} // namespace terra::fe