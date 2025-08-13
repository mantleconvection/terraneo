#pragma once

#include "solver.hpp"

namespace terra::linalg::solvers {

/// @brief Richardson iterative solver for linear systems.
/// Satisfies the SolverLike concept (see solver.hpp).
/// Implements the update rule:
/// \f[ x^{(k+1)} = x^{(k)} + \omega (b - Ax^{(k)}) \f]
/// where \f$ \omega \f$ is the relaxation parameter.
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
template < OperatorLike OperatorT >
class Richardson
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType       = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType      = DstOf< OperatorType >;

    /// @brief Construct a Richardson solver.
    /// @param iterations Number of Richardson iterations to perform.
    /// @param omega Relaxation parameter.
    /// @param r_tmp Temporary vector for workspace.
    Richardson( const int iterations, const double omega, const RHSVectorType& r_tmp )
    : iterations_( iterations )
    , omega_( omega )
    , r_( r_tmp )
    {}

    /// @brief Solve the linear system using Richardson iteration.
    /// Applies the update rule for the specified number of iterations.
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        for ( int iteration = 0; iteration < iterations_; ++iteration )
        {
            apply( A, x, r_ );
            lincomb( x, { 1.0, omega_, -omega_ }, { x, b, r_ } );
        }
    }

  private:
    int           iterations_; ///< Number of iterations.
    double        omega_;      ///< Relaxation parameter.
    RHSVectorType r_;          ///< Temporary workspace vector.
};

/// @brief Static assertion: Richardson satisfies SolverLike concept.
static_assert( SolverLike< Richardson< linalg::detail::DummyConcreteOperator > > );

} // namespace terra::linalg::solvers