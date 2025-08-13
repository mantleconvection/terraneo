#pragma once

#include "terra/linalg/operator.hpp"
#include "terra/linalg/solvers/solver.hpp"
#include "terra/linalg/vector.hpp"

namespace terra::linalg::solvers {

/// @brief Block-diagonal preconditioner for 2x2 block operators.
///
/// Satisfies the SolverLike concept (see solver.hpp).
/// Applies separate preconditioners to the (1,1) and (2,2) blocks.
/// The block-diagonal preconditioner solves:
/// \f[
/// \begin{pmatrix}
/// P_{11} & 0 \\
/// 0 & P_{22}
/// \end{pmatrix}
/// \begin{pmatrix}
/// x_1 \\ x_2
/// \end{pmatrix}
/// =
/// \begin{pmatrix}
/// b_1 \\ b_2
/// \end{pmatrix}
/// \f]
/// where \f$ P_{11} \f$ and \f$ P_{22} \f$ are preconditioners for the (1,1) and (2,2) blocks, respectively.
/// @tparam OperatorT Operator type (must satisfy Block2x2OperatorLike).
/// @tparam Block11Preconditioner Preconditioner for the (1,1) block (must satisfy SolverLike).
/// @tparam Block22Preconditioner Preconditioner for the (2,2) block (must satisfy SolverLike).
template < Block2x2OperatorLike OperatorT, SolverLike Block11Preconditioner, SolverLike Block22Preconditioner >
class BlockDiagonalPreconditioner2x2
{
  public:
    /// @brief Operator type to be solved.
    using OperatorType       = OperatorT;
    /// @brief Solution vector type (must be Block2VectorLike).
    using SolutionVectorType = SrcOf< OperatorType >;
    /// @brief Right-hand side vector type (must be Block2VectorLike).
    using RHSVectorType      = DstOf< OperatorType >;

    /// @brief Static assertions to ensure block vector types.
    static_assert(
        Block2VectorLike< SolutionVectorType >,
        "The solution vector of the BlockPreconditioner2x2 must be Block2VectorLike." );
    static_assert(
        Block2VectorLike< RHSVectorType >,
        "The RHS vector of the BlockPreconditioner2x2 must be Block2VectorLike." );

    /// @brief Construct a block-diagonal preconditioner with given block preconditioners.
    /// @param block11_preconditioner Preconditioner for the (1,1) block.
    /// @param block22_preconditioner Preconditioner for the (2,2) block.
    BlockDiagonalPreconditioner2x2(
        const Block11Preconditioner& block11_preconditioner,
        const Block22Preconditioner& block22_preconditioner )
    : block11_preconditioner_( block11_preconditioner )
    , block22_preconditioner_( block22_preconditioner )
    {}

    /// @brief Solve the block-diagonal preconditioner system.
    /// Applies the block11 and block22 preconditioners to the respective blocks.
    /// @param A Block 2x2 operator.
    /// @param x Solution block vector (output).
    /// @param b Right-hand side block vector (input).
    void solve_impl( OperatorType& A, SolutionVectorType& x, const RHSVectorType& b )
    {
        solve( block11_preconditioner_, A.block_11(), x.block_1(), b.block_1() );
        solve( block22_preconditioner_, A.block_22(), x.block_2(), b.block_2() );
    }

  private:
    Block11Preconditioner block11_preconditioner_; ///< Preconditioner for (1,1) block.
    Block22Preconditioner block22_preconditioner_; ///< Preconditioner for (2,2) block.
};

/// @brief Static assertion: BlockDiagonalPreconditioner2x2 satisfies SolverLike concept.
static_assert( SolverLike< BlockDiagonalPreconditioner2x2<
                   linalg::detail::DummyConcreteBlock2x2Operator,
                   detail::DummySolver< linalg::detail::DummyConcreteBlock2x2Operator::Block11Type >,
                   detail::DummySolver< linalg::detail::DummyConcreteBlock2x2Operator::Block22Type > > > );

} // namespace terra::linalg::solvers