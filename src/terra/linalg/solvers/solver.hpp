#pragma once

#include <optional>

#include "iterative_solver_info.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::util {
class Table;
}

namespace terra::linalg::solvers {

/// @brief Concept for types that behave like linear solvers.
/// Requires exposing OperatorType and a solve_impl method.
/// See OperatorLike in operator.hpp for operator requirements.
template < typename T >
concept SolverLike = requires(
    // TODO: Cannot make solver const since we may have temporaries as members.
    T& self,
    // TODO: See OperatorLike for why A is not const.
    typename T::OperatorType&                      A,
    typename T::OperatorType::SrcVectorType&       x,
    const typename T::OperatorType::DstVectorType& b ) {
    /// @brief Operator type to be solved.
    typename T::OperatorType;

    /// @brief Operator type must satisfy OperatorLike concept.
    requires OperatorLike< typename T::OperatorType >;

    /// @brief Required solve implementation.
    { self.solve_impl( A, x, b ) } -> std::same_as< void >;
};

/// @brief Alias for the solution vector type of a solver.
template < SolverLike Solver >
using SolutionOf = SrcOf< typename Solver::OperatorType >;

/// @brief Alias for the right-hand side vector type of a solver.
template < SolverLike Solver >
using RHSOf = DstOf< typename Solver::OperatorType >;

/// @brief Solve a linear system using the given solver and operator.
/// Calls the solver's solve_impl method.
/// @param solver The solver instance.
/// @param A The operator (matrix).
/// @param x Solution vector (output).
/// @param b Right-hand side vector (input).
template < SolverLike Solver, OperatorLike Operator, VectorLike SolutionVector, VectorLike RHSVector >
void solve( Solver& solver, Operator& A, SolutionVector& x, const RHSVector& b )
{
    solver.solve_impl( A, x, b );
}

namespace detail {

/// @brief Dummy solver for concept checks and testing.
/// Implements solve_impl as a no-op.
template < OperatorLike OperatorT >
class DummySolver
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;

    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType      = DstOf< OperatorType >;

    /// @brief Dummy solve_impl, does nothing.
    /// @param A Operator.
    /// @param x Solution vector.
    /// @param b Right-hand side vector.
    void solve_impl( const OperatorType& A, SolutionVectorType& x, const RHSVectorType& b ) const
    {
        (void) A;
        (void) x;
        (void) b;
    }
};

/// @brief Static assertion to check SolverLike concept for DummySolver.
static_assert( SolverLike< DummySolver< linalg::detail::DummyConcreteOperator > > );

} // namespace detail

} // namespace terra::linalg::solvers