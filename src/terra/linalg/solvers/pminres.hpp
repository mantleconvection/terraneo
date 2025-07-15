

#pragma once

#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"
#include "util/table.hpp"

namespace terra::linalg::solvers {

template < OperatorLike OperatorT, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class PMINRES
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    using ScalarType = typename SolutionVectorType::ScalarType;

    PMINRES(
        const IterativeSolverParameters& params,
        const RHSVectorType&             az_tmp,
        const RHSVectorType&             v_j_minus_1_tmp,
        const RHSVectorType&             v_j_tmp,
        const SolutionVectorType&        w_j_minus_1_tmp,
        const SolutionVectorType&        w_j_tmp,
        const SolutionVectorType&        z_tmp,
        const SolutionVectorType&        z_j_plus_1_tmp )
    : PMINRES(
          params,
          az_tmp,
          v_j_minus_1_tmp,
          v_j_tmp,
          w_j_minus_1_tmp,
          w_j_tmp,
          z_tmp,
          z_j_plus_1_tmp,
          IdentitySolver< OperatorT >() )
    {}

    PMINRES(
        const IterativeSolverParameters& params,
        const RHSVectorType&             az_tmp,
        const RHSVectorType&             v_j_minus_1_tmp,
        const RHSVectorType&             v_j_tmp,
        const SolutionVectorType&        w_j_minus_1_tmp,
        const SolutionVectorType&        w_j_tmp,
        const SolutionVectorType&        z_tmp,
        const SolutionVectorType&        z_j_plus_1_tmp,
        const PreconditionerT            preconditioner )
    : tag_( "pminres_solver" )
    , params_( params )
    , az_( az_tmp )
    , v_j_minus_1_( v_j_minus_1_tmp )
    , v_j_( v_j_tmp )
    , w_j_minus_1_( w_j_minus_1_tmp )
    , w_j_( w_j_tmp )
    , z_j_plus_1_( z_j_plus_1_tmp )
    , z_( z_tmp )
    , preconditioner_( preconditioner )
    {}

    void set_tag( const std::string& tag ) { tag_ = tag; }

    void solve_impl(
        OperatorType&                                          A,
        SolutionVectorType&                                    x,
        const RHSVectorType&                                   b,
        int                                                    level,
        std::optional< std::reference_wrapper< util::Table > > statistics )
    {
        assign( v_j_minus_1_, 0, level );
        assign( w_j_, 0, level );
        assign( w_j_minus_1_, 0, level );

        apply( A, x, v_j_, level );
        lincomb( v_j_, { 1.0, -1.0 }, { b, v_j_ }, level );

        solve( preconditioner_, A, z_, v_j_, level );

        ScalarType gamma_j_minus_1 = 1.0;
        ScalarType gamma_j         = std::sqrt( dot( z_, v_j_, level ) );

        ScalarType eta         = gamma_j;
        ScalarType s_j_minus_1 = 0;
        ScalarType s_j         = 0;
        ScalarType c_j_minus_1 = 1;
        ScalarType c_j         = 1;

        const ScalarType initial_residual = gamma_j;

        if ( statistics.has_value() )
        {
            statistics->get().add_row(
                { { "tag", tag_ },
                  { "iteration", 0 },
                  { "relative_residual", 1.0 },
                  { "absolute_residual", initial_residual } } );
        }

        if ( std::sqrt( initial_residual ) < params_.absolute_residual_tolerance() )
        {
            return;
        }

        for ( int iteration = 1; iteration <= params_.max_iterations(); ++iteration )
        {
            lincomb( z_, { 1.0 / gamma_j }, { z_ }, level );

            apply( A, z_, az_, level );

            const ScalarType delta = dot( az_, z_, level );

            lincomb(
                v_j_minus_1_,
                { 1.0, -delta / gamma_j, -gamma_j / gamma_j_minus_1 },
                { az_, v_j_, v_j_minus_1_ },
                level );
            swap( v_j_minus_1_, v_j_ );

            assign( z_j_plus_1_, 0.0, level );
            solve( preconditioner_, A, z_j_plus_1_, v_j_, level );

            const ScalarType gamma_j_plus_1 = std::sqrt( dot( z_j_plus_1_, v_j_, level ) );

            const ScalarType alpha_0 = c_j * delta - c_j_minus_1 * s_j * gamma_j;
            const ScalarType alpha_1 = std::sqrt( alpha_0 * alpha_0 + gamma_j_plus_1 * gamma_j_plus_1 );
            const ScalarType alpha_2 = s_j * delta + c_j_minus_1 * c_j * gamma_j;
            const ScalarType alpha_3 = s_j_minus_1 * gamma_j;

            const ScalarType c_j_plus_1 = alpha_0 / alpha_1;
            const ScalarType s_j_plus_1 = gamma_j_plus_1 / alpha_1;

            lincomb(
                w_j_minus_1_,
                { 1.0 / alpha_1, -alpha_3 / alpha_1, -alpha_2 / alpha_1 },
                { z_, w_j_minus_1_, w_j_ },
                level );
            swap( w_j_minus_1_, w_j_ );

            lincomb( x, { 1.0, c_j_plus_1 * eta }, { x, w_j_ }, level );

            eta = -s_j_plus_1 * eta;

            apply( A, x, az_, level );
            lincomb( az_, { 1.0, -1.0 }, { b, az_ }, level );

            const ScalarType absolute_residual = std::sqrt( dot( az_, az_, level ) );
            const ScalarType relative_residual = absolute_residual / initial_residual;

            if ( statistics.has_value() )
            {
                statistics->get().add_row(
                    { { "tag", tag_ },
                      { "iteration", iteration },
                      { "relative_residual", relative_residual },
                      { "absolute_residual", absolute_residual } } );
            }

            if ( relative_residual <= params_.relative_residual_tolerance() )
            {
                return;
            }

            if ( std::sqrt( absolute_residual ) < params_.absolute_residual_tolerance() )
            {
                return;
            }

            swap( z_, z_j_plus_1_ );

            gamma_j_minus_1 = gamma_j;
            gamma_j         = gamma_j_plus_1;

            c_j_minus_1 = c_j;
            c_j         = c_j_plus_1;

            s_j_minus_1 = s_j;
            s_j         = s_j_plus_1;
        }
    }

  private:
    std::string tag_;

    IterativeSolverParameters params_;

    RHSVectorType      az_;
    RHSVectorType      v_j_minus_1_;
    RHSVectorType      v_j_;
    SolutionVectorType w_j_minus_1_;
    SolutionVectorType w_j_;
    SolutionVectorType z_j_plus_1_;
    SolutionVectorType z_;

    PreconditionerT preconditioner_;
};

static_assert( SolverLike< PMINRES< linalg::detail::DummyOperator<
                   linalg::detail::DummyVector< double >,
                   linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers