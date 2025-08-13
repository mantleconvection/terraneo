#pragma once

#include "identity_solver.hpp"
#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/iterative_solver_info.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"
#include "util/table.hpp"

namespace terra::linalg::solvers {

/// @brief Preconditioned MINRES (PMINRES) iterative solver for symmetric indefinite linear systems.
///
/// See, e.g., 
/// @code
/// Elman, H. C., Silvester, D. J., & Wathen, A. J. (2014). 
/// Finite elements and fast iterative solvers: with applications in incompressible fluid dynamics. 
/// Oxford university press. 
/// @endcode
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Supports optional preconditioning.
/// @tparam OperatorT Operator type (must satisfy OperatorLike).
/// @tparam PreconditionerT Preconditioner type (must satisfy SolverLike, defaults to IdentitySolver).
template < OperatorLike OperatorT, SolverLike PreconditionerT = IdentitySolver< OperatorT > >
class PMINRES
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType       = OperatorT;
    /// @brief Solution vector type.
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type.
    using RHSVectorType      = DstOf< OperatorType >;
    /// @brief Scalar type for computations.
    using ScalarType         = typename SolutionVectorType::ScalarType;

    /// @brief Construct a PMINRES solver with default identity preconditioner.
    /// @param params Iterative solver parameters.
    /// @param statistics Shared pointer to statistics table.
    /// @param tmp Temporary vectors for workspace.
    PMINRES(
        const IterativeSolverParameters&         params,
        const std::shared_ptr< util::Table >&    statistics,
        const std::vector< SolutionVectorType >& tmp )
    : PMINRES( params, statistics, tmp, IdentitySolver< OperatorT >() )
    {}

    /// @brief Construct a PMINRES solver with a custom preconditioner.
    /// @param params Iterative solver parameters.
    /// @param statistics Shared pointer to statistics table.
    /// @param tmp Temporary vectors for workspace.
    /// @param preconditioner Preconditioner solver.
    PMINRES(
        const IterativeSolverParameters&         params,
        const std::shared_ptr< util::Table >&    statistics,
        const std::vector< SolutionVectorType >& tmp,
        const PreconditionerT                    preconditioner )
    : tag_( "pminres_solver" )
    , params_( params )
    , statistics_( statistics )
    , tmp_( tmp )
    , preconditioner_( preconditioner )
    {}

    /// @brief Set a tag string for statistics output.
    /// @param tag Tag string.
    void set_tag( const std::string& tag ) { tag_ = tag; }

    /// @brief Solve the linear system \f$ Ax = b \f$ using PMINRES.
    /// Calls the iterative solver and updates statistics.
    /// @param A Operator (matrix).
    /// @param x Solution vector (output).
    /// @param b Right-hand side vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        auto& az_          = tmp_[0]; ///< Temporary vector for A*z.
        auto& v_j_minus_1_ = tmp_[1]; ///< Temporary vector for v_{j-1}.
        auto& v_j_         = tmp_[2]; ///< Temporary vector for v_j.
        auto& w_j_minus_1_ = tmp_[3]; ///< Temporary vector for w_{j-1}.
        auto& w_j_         = tmp_[4]; ///< Temporary vector for w_j.
        auto& z_j_plus_1_  = tmp_[5]; ///< Temporary vector for z_{j+1}.
        auto& z_           = tmp_[6]; ///< Temporary vector for z.

        assign( v_j_minus_1_, 0 );
        assign( w_j_, 0 );
        assign( w_j_minus_1_, 0 );

        apply( A, x, v_j_ );
        lincomb( v_j_, { 1.0, -1.0 }, { b, v_j_ } );

        solve( preconditioner_, A, z_, v_j_ );

        ScalarType gamma_j_minus_1 = 1.0;
        ScalarType gamma_j         = std::sqrt( dot( z_, v_j_ ) );

        ScalarType eta         = gamma_j;
        ScalarType s_j_minus_1 = 0;
        ScalarType s_j         = 0;
        ScalarType c_j_minus_1 = 1;
        ScalarType c_j         = 1;

        const ScalarType initial_residual = gamma_j;

        if ( statistics_ )
        {
            statistics_->add_row(
                { { "tag", tag_ },
                  { "iteration", 0 },
                  { "relative_residual", 1.0 },
                  { "absolute_residual", initial_residual } } );
        }

        if ( initial_residual < params_.absolute_residual_tolerance() )
        {
            return;
        }

        for ( int iteration = 1; iteration <= params_.max_iterations(); ++iteration )
        {
            lincomb( z_, { 1.0 / gamma_j }, { z_ } );

            apply( A, z_, az_ );

            const ScalarType delta = dot( az_, z_ );

            lincomb( v_j_minus_1_, { 1.0, -delta / gamma_j, -gamma_j / gamma_j_minus_1 }, { az_, v_j_, v_j_minus_1_ } );
            swap( v_j_minus_1_, v_j_ );

            assign( z_j_plus_1_, 0.0 );
            solve( preconditioner_, A, z_j_plus_1_, v_j_ );

            const ScalarType gamma_j_plus_1 = std::sqrt( dot( z_j_plus_1_, v_j_ ) );

            const ScalarType alpha_0 = c_j * delta - c_j_minus_1 * s_j * gamma_j;
            const ScalarType alpha_1 = std::sqrt( alpha_0 * alpha_0 + gamma_j_plus_1 * gamma_j_plus_1 );
            const ScalarType alpha_2 = s_j * delta + c_j_minus_1 * c_j * gamma_j;
            const ScalarType alpha_3 = s_j_minus_1 * gamma_j;

            const ScalarType c_j_plus_1 = alpha_0 / alpha_1;
            const ScalarType s_j_plus_1 = gamma_j_plus_1 / alpha_1;

            lincomb(
                w_j_minus_1_, { 1.0 / alpha_1, -alpha_3 / alpha_1, -alpha_2 / alpha_1 }, { z_, w_j_minus_1_, w_j_ } );
            swap( w_j_minus_1_, w_j_ );

            lincomb( x, { 1.0, c_j_plus_1 * eta }, { x, w_j_ } );

            eta = -s_j_plus_1 * eta;

            const ScalarType absolute_residual = std::abs( eta );
            const ScalarType relative_residual = absolute_residual / initial_residual;

            if ( statistics_ )
            {
                statistics_->add_row(
                    { { "tag", tag_ },
                      { "iteration", iteration },
                      { "relative_residual", relative_residual },
                      { "absolute_residual", absolute_residual } } );
            }

            if ( relative_residual <= params_.relative_residual_tolerance() )
            {
                return;
            }

            if ( absolute_residual < params_.absolute_residual_tolerance() )
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
    std::string tag_; ///< Tag for statistics output.

    IterativeSolverParameters params_; ///< Solver parameters.

    std::shared_ptr< util::Table > statistics_; ///< Statistics table.

    std::vector< SolutionVectorType > tmp_; ///< Temporary workspace vectors.

    PreconditionerT preconditioner_; ///< Preconditioner solver.
};

/// @brief Static assertion: PMINRES satisfies SolverLike concept.
static_assert( SolverLike< PMINRES< linalg::detail::DummyOperator<
                   linalg::detail::DummyVector< double >,
                   linalg::detail::DummyVector< double > > > > );

} // namespace terra::linalg::solvers