

#pragma once

#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::linalg::solvers {

template < OperatorLike OperatorT >
class IdentitySolver
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    void solve_impl(
        const OperatorType&                                                  A,
        SolutionVectorType&                                                  x,
        const RHSVectorType&                                                 b,
        const int                                                            level,
        std::optional< std::reference_wrapper< IterativeSolverStatistics > > params ) const
    {
        assign( x, b, level );
        (void) A;
        (void) params;
    }
};

static_assert( SolverLike< IdentitySolver< linalg::detail::DummyOperator<
                   linalg::detail::DummyVector< double >,
                   linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers