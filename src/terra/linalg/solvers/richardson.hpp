#pragma once

#include <iostream>

#include "solver.hpp"

namespace terra::linalg::solvers {

template < OperatorLike OperatorT >
class Richardson
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    Richardson( const IterativeSolverParameters& params, const double omega, const RHSVectorType& r_tmp )
    : params_( params )
    , omega_( omega )
    , r_( r_tmp )
    {}

    void solve_impl(
        OperatorType&                                                        A,
        SolutionVectorType&                                                  x,
        const RHSVectorType&                                                 b,
        int                                                                  level,
        std::optional< std::reference_wrapper< IterativeSolverStatistics > > params )
    {
        for ( int iteration = 0; iteration < params_.max_iterations(); ++iteration )
        {
            assign( r_, 0, level );
            apply( A, x, r_, level );
            lincomb( x, { 1.0, omega_, -omega_ }, { x, b, r_ }, level );
        }
    }

  private:
    IterativeSolverParameters params_;
    double                    omega_;
    RHSVectorType             r_;
};

static_assert( SolverLike< Richardson< linalg::detail::DummyConcreteOperator > > );

} // namespace terra::linalg::solvers