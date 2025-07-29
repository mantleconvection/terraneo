

#pragma once

#include "eigen/eigen_wrapper.hpp"
#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"
#include "util/table.hpp"

namespace terra::linalg::solvers {

template < OperatorLike OperatorT, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class PBiCGStab
{
  public:
    using OperatorType       = OperatorT;
    using SolutionVectorType = SrcOf< OperatorType >;
    using RHSVectorType      = DstOf< OperatorType >;

    using ScalarType = typename SolutionVectorType::ScalarType;

    PBiCGStab( const int l, const IterativeSolverParameters& params, const std::vector< SolutionVectorType >& tmp )
    : l_( l )
    , params_( params )
    , tmp_( tmp )
    , tag_( "pbicgstab_solver" )
    , preconditioner_( IdentitySolver< OperatorT >() )
    {
        const int num_required_tmp_vectors = 2 * ( l + 1 ) + 2;
        if ( tmp.size() < num_required_tmp_vectors )
        {
            throw std::runtime_error(
                "PBiCGStab: tmp.size() != 2 * (l+1) + 2 = " + std::to_string( num_required_tmp_vectors ) );
        }
    }

    void set_tag( const std::string& tag ) { tag_ = tag; }

    void solve_impl(
        OperatorType&                                          A,
        SolutionVectorType&                                    x,
        const RHSVectorType&                                   b,
        int                                                    level,
        std::optional< std::reference_wrapper< util::Table > > statistics )
    {
        linalg::randomize( r_shadow(), level );

        for ( int j = 0; j < l_ + 1; ++j )
        {
            assign( us( j ), 0, level );
        }

        apply( A, x, residual(), level );
        lincomb( residual(), { 1.0, -1.0 }, { b, residual() }, level );

        if constexpr ( !std::is_same_v< PreconditionerT, IdentitySolver< OperatorT > > )
        {
            assign( tmp_prec(), residual(), level );
            solve( A, tmp_prec(), residual(), level );
            assign( residual(), tmp_prec(), level );
        }

        const ScalarType initial_residual = std::sqrt( dot( residual(), residual(), level ) );

        ScalarType absolute_residual = initial_residual;
        ScalarType relative_residual = 1.0;
        int        iteration         = 0;

        auto add_table_row = [&]( bool final_iteration ) {
            if ( statistics.has_value() )
            {
                if ( final_iteration )
                {
                    statistics->get().add_row(
                        { { "tag", tag_ },
                          { "final_iteration", iteration },
                          { "relative_residual", relative_residual },
                          { "absolute_residual", absolute_residual } } );
                }
                else
                {
                    statistics->get().add_row(
                        { { "tag", tag_ },
                          { "iteration", iteration },
                          { "relative_residual", relative_residual },
                          { "absolute_residual", absolute_residual } } );
                }
            }
        };

        add_table_row( false );

        if ( initial_residual < params_.absolute_residual_tolerance() )
        {
            add_table_row( true );
            return;
        }

        ScalarType omega = 1;
        ScalarType sigma = 1;

        Eigen::Matrix< ScalarType, Eigen::Dynamic, Eigen::Dynamic > M( l_, l_ );

        Eigen::Matrix< ScalarType, Eigen::Dynamic, 1 > gamma( l_ );

        for ( ; iteration < params_.max_iterations(); ++iteration )
        {
            sigma = -omega * sigma;

            // BiCG part

            for ( int j = 0; j < l_; ++j )
            {
                auto rho  = dot( r_shadow(), rs( j ), level );
                auto beta = rho / sigma;

                for ( int i = 0; i <= j; ++i )
                {
                    lincomb( us( i ), { 1.0, -beta }, { rs( i ), us( i ) }, level );
                }

                apply( A, us( j ), us( j + 1 ), level );

                if constexpr ( !std::is_same_v< PreconditionerT, IdentitySolver< OperatorT > > )
                {
                    assign( tmp_prec(), us( j + 1 ), level );
                    solve( A, tmp_prec(), us( j + 1 ), level );
                    assign( us( j + 1 ), tmp_prec(), level );
                }

                sigma      = dot( r_shadow(), us( j + 1 ), level );
                auto alpha = rho / sigma;

                for ( int i = 0; i <= j; ++i )
                {
                    lincomb( rs( i ), { 1.0, -alpha }, { rs( i ), us( i + 1 ) }, level );
                }

                apply( A, rs( j ), rs( j + 1 ), level );

                if constexpr ( !std::is_same_v< PreconditionerT, IdentitySolver< OperatorT > > )
                {
                    assign( tmp_prec(), rs( j + 1 ), level );
                    solve( A, tmp_prec(), rs( j + 1 ), level );
                    assign( rs( j + 1 ), tmp_prec(), level );
                }

                lincomb( x, { 1.0, alpha }, { x, us( 0 ) }, level );
            }

            // MR part

            Eigen::Matrix< ScalarType, Eigen::Dynamic, 1 > M0( l_ );

            for ( int j = 1; j < l_ + 1; j++ )
            {
                M0( j - 1 ) = dot( rs( 0 ), rs( j ), level );
            }

            for ( int i = 0; i < l_; i++ )
            {
                for ( int j = 0; j < l_; j++ )
                {
                    M( i, j ) = dot( rs( i + 1 ), rs( j + 1 ), level );
                }
            }

            gamma = M.fullPivLu().solve( M0 );

            for ( int j = 1; j < l_ + 1; ++j )
            {
                lincomb( us( 0 ), { 1.0, -gamma( j - 1 ) }, { us( 0 ), us( j ) }, level );
            }

            for ( int j = 0; j < l_; ++j )
            {
                lincomb( x, { 1.0, gamma( j ) }, { x, rs( j ) }, level );
            }

            for ( int j = 1; j < l_ + 1; ++j )
            {
                lincomb( rs( 0 ), { 1.0, -gamma( j - 1 ) }, { rs( 0 ), rs( j ) }, level );
            }

            omega = gamma( l_ - 1 );

            absolute_residual = std::sqrt( dot( residual(), residual(), level ) );

            relative_residual = absolute_residual / initial_residual;

            add_table_row( false );

            if ( relative_residual <= params_.relative_residual_tolerance() )
            {
                add_table_row( true );
                return;
            }

            if ( absolute_residual < params_.absolute_residual_tolerance() )
            {
                add_table_row( true );
                return;
            }
        }

        add_table_row( true );
    }

  private:
    SolutionVectorType& r_shadow() { return tmp_[0]; }
    SolutionVectorType& rs( int index ) { return tmp_[1 + index]; }
    SolutionVectorType& us( int index ) { return tmp_[1 + l_ + 1 + index]; }
    SolutionVectorType& tmp_prec() { return tmp_[1 + 2 * ( l_ + 1 )]; }
    SolutionVectorType& residual() { return rs( 0 ); }

    int l_;

    IterativeSolverParameters params_;

    std::vector< SolutionVectorType > tmp_;

    std::string tag_;

    PreconditionerT preconditioner_;
};

static_assert( SolverLike< PBiCGStab< linalg::detail::DummyOperator<
                   linalg::detail::DummyVector< double >,
                   linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers