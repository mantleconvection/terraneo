#pragma once

#include "eigen/eigen_wrapper.hpp"
#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"
#include "util/table.hpp"

namespace terra::linalg::solvers {

/// @brief BiCGStab(l) iterative solver for general (possibly unsymmetric) linear systems.
///
/// See
/// @code
/// Sleijpen, G. L., & Fokkema, D. R. (1993).
/// BiCGstab (ell) for linear equations involving unsymmetric matrices with complex spectrum.
/// Electronic Transactions on Numerical Analysis., 1, 11-32.
/// @endcode
/// for details.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Supports optional preconditioning.
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam PreconditionerT Preconditioner type (must satisfy SolverLike, defaults to IdentitySolver).
template < OperatorLike OperatorT, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class PBiCGStab
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType = DstOf< OperatorType >;
    /// @brief Scalar type for computations.
    using ScalarType = typename SolutionVectorType::ScalarType;

    /// @brief Construct a PBiCGStab solver.
    /// @param l Number of BiCG iterations per "minimal residual" (MR) step.
    /// @param params Iterative solver parameters.
    /// @param statistics Shared pointer to statistics table.
    /// @param tmp Temporary vectors for workspace. (At least 2 * (l + 1) + 2 vectors are required.)
    /// @param preconditioner Preconditioner solver (optional).
    PBiCGStab(
        const int                                l,
        const IterativeSolverParameters&         params,
        const std::shared_ptr< util::Table >&    statistics,
        const std::vector< SolutionVectorType >& tmp,
        const PreconditionerT&                   preconditioner = IdentitySolver< OperatorT >() )
    : l_( l )
    , params_( params )
    , statistics_( statistics )
    , tmp_( tmp )
    , tag_( "pbicgstab_solver" )
    , preconditioner_( preconditioner )
    {
        const int num_required_tmp_vectors = 2 * ( l + 1 ) + 2;
        if ( tmp.size() < num_required_tmp_vectors )
        {
            throw std::runtime_error(
                "PBiCGStab: tmp.size() < 2 * (l+1) + 2 = " + std::to_string( num_required_tmp_vectors ) );
        }

        if ( tmp.size() > num_required_tmp_vectors )
        {
            std::cout << "Note: You are using more tmp vectors that required in PBiCGStab. Required: "
                      << num_required_tmp_vectors << ", passed: " << tmp.size() << std::endl;
        }
    }

    /// @brief Set a tag string for statistics output.
    /// @param tag Tag string.
    void set_tag( const std::string& tag ) { tag_ = tag; }

    /// @brief Solve the linear system \( Ax = b \) using PBiCGStab.
    /// Calls the iterative solver and updates statistics.
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        linalg::randomize( r_shadow() );

        for ( int j = 0; j < l_ + 1; ++j )
        {
            assign( us( j ), 0 );
        }

        apply( A, x, residual() );
        lincomb( residual(), { 1.0, -1.0 }, { b, residual() } );

        if constexpr ( !std::is_same_v< PreconditionerT, IdentitySolver< OperatorT > > )
        {
            assign( tmp_prec(), residual() );
            solve( preconditioner_, A, tmp_prec(), residual() );
            assign( residual(), tmp_prec() );
        }

        const ScalarType initial_residual = std::sqrt( dot( residual(), residual() ) );

        ScalarType absolute_residual = initial_residual;
        ScalarType relative_residual = 1.0;
        int        iteration         = 0;

        /// @brief Lambda to add a row to the statistics table.
        auto add_table_row = [&]( bool final_iteration ) {
            if ( statistics_ )
            {
                if ( final_iteration )
                {
                    statistics_->add_row(
                        { { "tag", tag_ },
                          { "final_iteration", iteration },
                          { "relative_residual", relative_residual },
                          { "absolute_residual", absolute_residual } } );
                }
                else
                {
                    statistics_->add_row(
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

        // This has to be double regardless of the template parameter for robustness.
        Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > M( l_, l_ );
        Eigen::Matrix< double, Eigen::Dynamic, 1 >              gamma( l_ );

        iteration++;

        for ( ; iteration < params_.max_iterations(); ++iteration )
        {
            sigma = -omega * sigma;

            // BiCG part

            for ( int j = 0; j < l_; ++j )
            {
                auto rho  = dot( r_shadow(), rs( j ) );
                auto beta = rho / sigma;

                for ( int i = 0; i <= j; ++i )
                {
                    lincomb( us( i ), { 1.0, -beta }, { rs( i ), us( i ) } );
                }

                apply( A, us( j ), us( j + 1 ) );

                if constexpr ( !std::is_same_v< PreconditionerT, IdentitySolver< OperatorT > > )
                {
                    assign( tmp_prec(), us( j + 1 ) );
                    solve( preconditioner_, A, tmp_prec(), us( j + 1 ) );
                    assign( us( j + 1 ), tmp_prec() );
                }

                sigma      = dot( r_shadow(), us( j + 1 ) );
                auto alpha = rho / sigma;

                for ( int i = 0; i <= j; ++i )
                {
                    lincomb( rs( i ), { 1.0, -alpha }, { rs( i ), us( i + 1 ) } );
                }

                apply( A, rs( j ), rs( j + 1 ) );

                if constexpr ( !std::is_same_v< PreconditionerT, IdentitySolver< OperatorT > > )
                {
                    assign( tmp_prec(), rs( j + 1 ) );
                    solve( preconditioner_, A, tmp_prec(), rs( j + 1 ) );
                    assign( rs( j + 1 ), tmp_prec() );
                }

                lincomb( x, { 1.0, alpha }, { x, us( 0 ) } );
            }

            // MR part

            // This has to be double regardless of the template parameter for robustness.
            Eigen::Matrix< double, Eigen::Dynamic, 1 > M0( l_ );

            for ( int j = 1; j < l_ + 1; j++ )
            {
                M0( j - 1 ) = dot( rs( 0 ), rs( j ) );
            }

            for ( int i = 0; i < l_; i++ )
            {
                for ( int j = 0; j < l_; j++ )
                {
                    M( i, j ) = dot( rs( i + 1 ), rs( j + 1 ) );
                }
            }

            gamma = M.fullPivLu().solve( M0 );

            for ( int j = 1; j < l_ + 1; ++j )
            {
                lincomb( us( 0 ), { 1.0, ScalarType( -gamma( j - 1 ) ) }, { us( 0 ), us( j ) } );
            }

            for ( int j = 0; j < l_; ++j )
            {
                lincomb( x, { 1.0, ScalarType( gamma( j ) ) }, { x, rs( j ) } );
            }

            for ( int j = 1; j < l_ + 1; ++j )
            {
                lincomb( rs( 0 ), { 1.0, ScalarType( -gamma( j - 1 ) ) }, { rs( 0 ), rs( j ) } );
            }

            omega = gamma( l_ - 1 );

            absolute_residual = std::sqrt( dot( residual(), residual() ) );
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
    /// @brief Accessor for the shadow residual vector.
    SolutionVectorType& r_shadow() { return tmp_[0]; }
    /// @brief Accessor for the j-th residual vector.
    SolutionVectorType& rs( int index ) { return tmp_[1 + index]; }
    /// @brief Accessor for the j-th search direction vector.
    SolutionVectorType& us( int index ) { return tmp_[1 + l_ + 1 + index]; }
    /// @brief Accessor for the temporary preconditioner vector.
    SolutionVectorType& tmp_prec() { return tmp_[1 + 2 * ( l_ + 1 )]; }
    /// @brief Accessor for the main residual vector.
    SolutionVectorType& residual() { return rs( 0 ); }

    int l_; ///< Number of BiCG iterations per MR step.

    IterativeSolverParameters params_; ///< Solver parameters.

    std::shared_ptr< util::Table > statistics_; ///< Statistics table.

    std::vector< SolutionVectorType > tmp_; ///< Temporary workspace vectors.

    std::string tag_; ///< Tag for statistics output.

    PreconditionerT preconditioner_; ///< Preconditioner solver.
};

/// @brief Static assertion: PBiCGStab satisfies SolverLike concept.
static_assert( SolverLike< PBiCGStab< linalg::detail::DummyOperator<
                   linalg::detail::DummyVector< double >,
                   linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers