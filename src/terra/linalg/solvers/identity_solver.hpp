#pragma once

#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::linalg::solvers {

/// @brief "Identity solver" for linear systems.
///
/// Implements a "no-op" solve operation by directly assigning 
/// the right-hand side to the solution vector: \f$ x \gets b \f$.
/// Satisfies the SolverLike concept (see solver.hpp).
/// Can be used as a placeholder for "no preconditioner".
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
template < OperatorLike OperatorT >
class IdentitySolver
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType       = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType      = DstOf< OperatorType >;

    /// @brief Solve the linear system by assigning the right-hand side to the solution.
    /// Implements \f$ x = b \f$.
    /// @param A Operator (matrix), unused.
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        assign( x, b );
        (void) A;
    }
};

/// @brief Static assertion: IdentitySolver satisfies SolverLike concept.
static_assert( SolverLike< IdentitySolver< linalg::detail::DummyOperator<
                   linalg::detail::DummyVector< double >,
                   linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers