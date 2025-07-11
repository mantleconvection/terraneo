
#pragma once
#include "vector.hpp"

namespace terra::linalg {

template < typename T >
concept OperatorLike =

    requires(
        const T&                         self_const,
        T&                               self,
        const typename T::SrcVectorType& src,
        typename T::DstVectorType&       dst,
        int                              level,
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
        { self.apply_impl( src, dst, level ) } -> std::same_as< void >;
    };

template < typename T >
concept Block2x2OperatorLike = OperatorLike< T > &&

                               requires( const T& self_const, T& self, dense::Vec< int, 2 > block ) {
                                   typename T::Block11Type;
                                   typename T::Block12Type;
                                   typename T::Block21Type;
                                   typename T::Block22Type;

                                   requires OperatorLike< typename T::Block11Type >;
                                   requires OperatorLike< typename T::Block11Type >;
                                   requires OperatorLike< typename T::Block21Type >;
                                   requires OperatorLike< typename T::Block22Type >;

                                   { self_const.block_11() } -> std::same_as< typename T::Block11Type >;
                                   { self_const.block_12() } -> std::same_as< typename T::Block12Type >;
                                   { self_const.block_21() } -> std::same_as< typename T::Block21Type >;
                                   { self_const.block_22() } -> std::same_as< typename T::Block22Type >;
                               };

template < OperatorLike Operator >
using SrcOf = typename Operator::SrcVectorType;

template < OperatorLike Operator >
using DstOf = typename Operator::DstVectorType;

template < OperatorLike Operator >
void apply( Operator& A, const SrcOf< Operator >& src, DstOf< Operator >& dst, const int level )
{
    A.apply_impl( src, dst, level );
}

namespace detail {

template < VectorLike SrcVectorT, VectorLike DstVectorT >
class DummyOperator
{
  public:
    using SrcVectorType = SrcVectorT;
    using DstVectorType = DstVectorT;

    void apply_impl( const SrcVectorType& src, DstVectorType& dst, const int level ) const
    {
        (void) src;
        (void) dst;
        (void) level;
    }
};

class DummyConcreteOperator
{
  public:
    using SrcVectorType = DummyVector< double >;
    using DstVectorType = DummyVector< double >;

    void apply_impl( const SrcVectorType& src, DstVectorType& dst, const int level ) const
    {
        (void) src;
        (void) dst;
        (void) level;
    }
};

class DummyConcreteBlock2x2Operator
{
  public:
    using SrcVectorType = DummyBlock2Vector< double >;
    using DstVectorType = DummyBlock2Vector< double >;

    using Block11Type = DummyConcreteOperator;
    using Block12Type = DummyConcreteOperator;
    using Block21Type = DummyConcreteOperator;
    using Block22Type = DummyConcreteOperator;

    void apply_impl( const SrcVectorType& src, DstVectorType& dst, const int level ) const
    {
        (void) src;
        (void) dst;
        (void) level;
    }

    DummyConcreteOperator block_11() const { return block_11_; }
    DummyConcreteOperator block_12() const { return block_12_; }
    DummyConcreteOperator block_21() const { return block_21_; }
    DummyConcreteOperator block_22() const { return block_22_; }

  private:
    DummyConcreteOperator block_11_;
    DummyConcreteOperator block_12_;
    DummyConcreteOperator block_21_;
    DummyConcreteOperator block_22_;
};

static_assert( OperatorLike< DummyOperator< DummyVector< double >, DummyVector< double > > > );
static_assert( OperatorLike< DummyConcreteOperator > );
static_assert( Block2x2OperatorLike< DummyConcreteBlock2x2Operator > );

} // namespace detail

} // namespace terra::linalg