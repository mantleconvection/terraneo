#pragma once
#include "linalg.hpp"
#include "vector.hpp"

namespace terra::linalg {

/// @brief Modes for applying an operator to a vector.
/// Replace: Overwrite the destination vector.
/// Add: Add to the destination vector.
enum class OperatorApplyMode
{
    Replace,
    Add,
};

/// @brief Modes for communication during operator application.
/// SkipCommunication: Do not communicate.
/// CommunicateAdditively: Communicate and add results.
enum class OperatorCommunicationMode
{
    SkipCommunication,
    CommunicateAdditively,
};

/// @brief Concept for types that behave like linear operators.
/// Requires vector types, matvec implementation, and compatibility with VectorLike.
template < typename T >
concept OperatorLike = requires(
    const T&                         self_const,
    T&                               self,
    const typename T::SrcVectorType& src,
    typename T::DstVectorType&       dst,
    dense::Vec< int, 2 >             block ) {
    // Requires exposing the vector types.
    typename T::SrcVectorType;
    typename T::DstVectorType;

    // Require that Src and Dst vector types satisfy VectorLike
    requires VectorLike< typename T::SrcVectorType >;
    requires VectorLike< typename T::DstVectorType >;

    // Required matvec implementation
    // TODO: T& self is not const because having apply_impl const is not convenient - mostly because
    //       it is handy to reuse send/recv buffers that are members of an operator implementation.
    //       To modify these members (i.e., to communicate) we cannot be const :(
    { self.apply_impl( src, dst ) } -> std::same_as< void >;
};

/// @brief Concept for types that can be used as Galerkin coarse-grid operators in a multigrid hierarchy.
/// See galerkin_coarsening_linear.hpp for more information on GCA.
template < typename Op >
concept GCACapable = requires(
    // dummy variables to define requirements
    Op&                                                self,
    const int                                          local_subdomain_id,
    const int                                          x_cell,
    const int                                          y_cell,
    const int                                          r_cell,
    const int                                          wedge,
    const int                                          dimi,
    const int                                          dimj,
    const dense::Mat< typename Op::ScalarType, 6, 6 >& mat ) {

    // Require that the argument to be a linear operator
    requires OperatorLike< Op >;

    // The operator must allow read/write access to his local matrices in order to GCA them
    {
        self.get_lmatrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge, dimi, dimj )
    } -> std::same_as< dense::Mat< typename Op::ScalarType, 6, 6 > >;
    { self.set_lmatrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge, dimi, dimj, mat ) } -> std::same_as< void >;

    // Since TwoGridGCA works with multiple operators, and their respective domains (coarse-fine nested loop),
    // it is required to have access to geometric information stored in the operators.
    { self.get_domain() } -> std::same_as< grid::shell::DistributedDomain& >;
    { self.get_radii() } -> std::same_as< grid::Grid2DDataScalar< typename Op::ScalarType >& >;

};

/// @brief Concept for types that behave like block 2x2 operators.
/// Extends OperatorLike and requires block types and accessors.
template < typename T >
concept Block2x2OperatorLike = OperatorLike< T > && requires( const T& self_const, T& self ) {
    // Require that Src and Dst vector types satisfy VectorLike
    requires Block2VectorLike< typename T::SrcVectorType >;
    requires Block2VectorLike< typename T::DstVectorType >;

    typename T::Block11Type;
    typename T::Block12Type;
    typename T::Block21Type;
    typename T::Block22Type;

    requires OperatorLike< typename T::Block11Type >;
    requires OperatorLike< typename T::Block12Type >;
    requires OperatorLike< typename T::Block21Type >;
    requires OperatorLike< typename T::Block22Type >;

    { self_const.block_11() } -> std::same_as< const typename T::Block11Type& >;
    { self_const.block_12() } -> std::same_as< const typename T::Block12Type& >;
    { self_const.block_21() } -> std::same_as< const typename T::Block21Type& >;
    { self_const.block_22() } -> std::same_as< const typename T::Block22Type& >;

    { self.block_11() } -> std::same_as< typename T::Block11Type& >;
    { self.block_12() } -> std::same_as< typename T::Block12Type& >;
    { self.block_21() } -> std::same_as< typename T::Block21Type& >;
    { self.block_22() } -> std::same_as< typename T::Block22Type& >;
};

/// @brief Alias for the source vector type of an operator.
template < OperatorLike Operator >
using SrcOf = typename Operator::SrcVectorType;

/// @brief Alias for the destination vector type of an operator.
template < OperatorLike Operator >
using DstOf = typename Operator::DstVectorType;

/// @brief Apply an operator to a source vector and write to a destination vector.
/// @param A Operator to apply.
/// @param src Source vector.
/// @param dst Destination vector.
template < OperatorLike Operator >
void apply( Operator& A, const SrcOf< Operator >& src, DstOf< Operator >& dst )
{
    A.apply_impl( src, dst );
}

namespace detail {

/// @brief Dummy operator for testing concepts.
/// Implements apply_impl as a no-op.
template < VectorLike SrcVectorT, VectorLike DstVectorT >
class DummyOperator
{
  public:
    using SrcVectorType = SrcVectorT;
    using DstVectorType = DstVectorT;

    /// @brief Dummy apply_impl, does nothing.
    /// @param src Source vector.
    /// @param dst Destination vector.
    void apply_impl( const SrcVectorType& src, DstVectorType& dst ) const
    {
        (void) src;
        (void) dst;
    }
};

/// @brief Dummy concrete operator for concept checks.
/// Uses DummyVector as vector types.
class DummyConcreteOperator
{
  public:
    using SrcVectorType = DummyVector< double >;
    using DstVectorType = DummyVector< double >;

    /// @brief Dummy apply_impl, does nothing.
    /// @param src Source vector.
    /// @param dst Destination vector.
    void apply_impl( const SrcVectorType& src, DstVectorType& dst ) const
    {
        (void) src;
        (void) dst;
    }
};

/// @brief Dummy block 2x2 operator for concept checks.
/// Contains four DummyConcreteOperator blocks.
class DummyConcreteBlock2x2Operator
{
  public:
    using SrcVectorType = DummyBlock2Vector< double >;
    using DstVectorType = DummyBlock2Vector< double >;

    using Block11Type = DummyConcreteOperator;
    using Block12Type = DummyConcreteOperator;
    using Block21Type = DummyConcreteOperator;
    using Block22Type = DummyConcreteOperator;

    /// @brief Dummy apply_impl, does nothing.
    /// @param src Source block vector.
    /// @param dst Destination block vector.
    void apply_impl( const SrcVectorType& src, DstVectorType& dst ) const
    {
        (void) src;
        (void) dst;
    }

    /// @brief Get const reference to block 11.
    const Block11Type& block_11() const { return block_11_; }
    /// @brief Get const reference to block 12.
    const Block12Type& block_12() const { return block_12_; }
    /// @brief Get const reference to block 21.
    const Block21Type& block_21() const { return block_21_; }
    /// @brief Get const reference to block 22.
    const Block22Type& block_22() const { return block_22_; }

    /// @brief Get mutable reference to block 11.
    Block11Type& block_11() { return block_11_; }
    /// @brief Get mutable reference to block 12.
    Block12Type& block_12() { return block_12_; }
    /// @brief Get mutable reference to block 21.
    Block21Type& block_21() { return block_21_; }
    /// @brief Get mutable reference to block 22.
    Block22Type& block_22() { return block_22_; }

  private:
    Block11Type block_11_;
    Block12Type block_12_;
    Block21Type block_21_;
    Block22Type block_22_;
};

/// @brief Static assertions to check concepts for dummy operators.
static_assert( OperatorLike< DummyOperator< DummyVector< double >, DummyVector< double > > > );
static_assert( OperatorLike< DummyConcreteOperator > );
static_assert( Block2x2OperatorLike< DummyConcreteBlock2x2Operator > );

} // namespace detail

} // namespace terra::linalg