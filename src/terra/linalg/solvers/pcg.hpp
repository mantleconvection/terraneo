

#pragma once

#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::linalg::solvers {

template < OperatorLike OperatorT, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class PCG
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    using ScalarType = typename SolutionVectorType::ScalarType;

    PCG( const IterativeSolverParameters& params,
         const RHSVectorType&             r_tmp,
         const SolutionVectorType&        p_tmp,
         const RHSVectorType&             ap_tmp,
         const SolutionVectorType&        z_tmp )
    : PCG( params, r_tmp, p_tmp, ap_tmp, z_tmp, IdentitySolver< OperatorT >() )
    {}

    PCG( const IterativeSolverParameters& params,
         const RHSVectorType&             r_tmp,
         const SolutionVectorType&        p_tmp,
         const RHSVectorType&             ap_tmp,
         const SolutionVectorType&        z_tmp,
         const PreconditionerT            preconditioner )
    : params_( params )
    , r_( r_tmp )
    , p_( p_tmp )
    , ap_( ap_tmp )
    , z_( z_tmp )
    , preconditioner_( preconditioner )
    {}

    void solve_impl(
        OperatorType&                                                        A,
        SolutionVectorType&                                                  x,
        const RHSVectorType&                                                 b,
        int                                                                  level,
        std::optional< std::reference_wrapper< IterativeSolverStatistics > > params )
    {
        apply( A, x, r_, level );

        lincomb( r_, { 1.0, -1.0 }, { b, r_ }, level );

        solve( preconditioner_, A, z_, r_, level );

        assign( p_, z_, level );

        // TODO: should this be dot(z, z) instead or dot(r, r)?
        const ScalarType initial_residual = dot( r_, r_, level );

        if ( std::sqrt( initial_residual ) < params_.absolute_residual_tolerance() )
        {
            return;
        }

        for ( int iteration = 0; iteration < params_.max_iterations(); ++iteration )
        {
            const ScalarType alpha_num = dot( z_, r_, level );

            apply( A, p_, ap_, level );
            const ScalarType alpha_den = dot( ap_, p_, level );

            const ScalarType alpha = alpha_num / alpha_den;

            lincomb( x, { 1.0, alpha }, { x, p_ }, level );
            lincomb( r_, { 1.0, -alpha }, { r_, ap_ }, level );

            // TODO: is this the correct term for the residual check?
            const ScalarType absolute_residual = dot( r_, r_, level );

            const ScalarType relative_residual = std::sqrt( absolute_residual ) / std::sqrt( initial_residual );

            if ( relative_residual <= params_.relative_residual_tolerance() )
            {
                return;
            }

            if ( std::sqrt( absolute_residual ) < params_.absolute_residual_tolerance() )
            {
                return;
            }

            solve( preconditioner_, A, z_, r_, level );

            const ScalarType beta_num = dot( z_, r_, level );
            const ScalarType beta     = beta_num / alpha_num;

            lincomb( p_, { 1.0, beta }, { z_, p_ }, level );
        }
    }

  private:
    IterativeSolverParameters params_;

    RHSVectorType      r_;
    SolutionVectorType p_;
    RHSVectorType      ap_;
    SolutionVectorType z_;

    PreconditionerT preconditioner_;
};

static_assert(
    SolverLike<
        PCG< linalg::detail::
                 DummyOperator< linalg::detail::DummyVector< double >, linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers