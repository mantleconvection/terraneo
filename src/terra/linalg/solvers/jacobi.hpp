#pragma once

#include "solver.hpp"

namespace terra::linalg::solvers {

/// @brief Jacobi iterative solver for linear systems.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Uses a diagonal preconditioner and supports relaxation.
/// The update rule is:
/// \f[ x^{(k+1)} = x^{(k)} + \omega D^{-1} (b - Ax^{(k)}) \f]
/// where \f$ D \f$ is the diagonal of \f$ A \f$ and \f$ \omega \f$ is the relaxation parameter.
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
template < OperatorLike OperatorT >
class Jacobi
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType = DstOf< OperatorType >;

    /// @brief Scalar type for computations.
    using ScalarType = SolutionVectorType::ScalarType;

    /// @brief Construct a Jacobi solver.
    /// @param inverse_diagonal Inverse of the diagonal of the operator.
    /// @param iterations Number of Jacobi iterations to perform.
    /// @param tmp Temporary vector for workspace.
    /// @param omega Relaxation parameter (default 1.0).
    Jacobi(
        const SolutionVectorType& inverse_diagonal,
        const int                 iterations,
        const SolutionVectorType& tmp,
        const ScalarType          omega = 1.0 )
    : inverse_diagonal_( inverse_diagonal )
    , iterations_( iterations )
    , tmp_( tmp )
    , omega_( omega )
    {}

    /// @brief Solve the linear system using Jacobi iteration.
    /// Applies the update rule for the specified number of iterations.
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        for ( int iteration = 0; iteration < iterations_; ++iteration )
        {
            apply( A, x, tmp_ );
            lincomb( tmp_, { 1.0, -1.0 }, { b, tmp_ } );
            scale_in_place( tmp_, inverse_diagonal_ );
            lincomb( x, { 1.0, omega_ }, { x, tmp_ } );
        }
    }

  private:
    SolutionVectorType inverse_diagonal_; ///< Inverse diagonal vector.
    int                iterations_;       ///< Number of iterations.
    SolutionVectorType tmp_;              ///< Temporary workspace vector.
    ScalarType         omega_;            ///< Relaxation parameter.
};

/// @brief Static assertion: Jacobi satisfies SolverLike concept.
static_assert( SolverLike< Jacobi< linalg::detail::DummyConcreteOperator > > );

} // namespace terra::linalg::solvers