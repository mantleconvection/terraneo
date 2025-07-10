
#pragma once
#include "vector.hpp"

namespace terra::linalg {

template < typename T >
concept OperatorLike =

    // TODO: T& self is not const because having apply_impl const is not convenient - mostly because
    //       it is handy to reuse send/recv buffers that are members of an operator implementation.
    //       To modify these members (i.e., to communicate) we cannot be const :(
    requires( T& self, const typename T::SrcVectorType& src, typename T::DstVectorType& dst, int level ) {
        // Requires exposing the vector types.
        typename T::SrcVectorType;
        typename T::DstVectorType;

        // Require that Src and Dst vector types satisfy VectorLike
        requires VectorLike< typename T::SrcVectorType >;
        requires VectorLike< typename T::DstVectorType >;

        // Required lincomb overload with 4 args
        { self.apply_impl( src, dst, level ) } -> std::same_as< void >;
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

static_assert( OperatorLike< DummyOperator< DummyVector< double >, DummyVector< double > > > );
static_assert( OperatorLike< DummyConcreteOperator > );

} // namespace detail

} // namespace terra::linalg