
#pragma once

namespace terra::linalg {

template < typename T >
concept VectorLike = requires(
    const T&                                     self_const,
    T&                                           self,
    const std::vector< typename T::ScalarType >& c,
    const T&                                     x,
    const std::vector< T >&                      xx,
    const typename T::ScalarType                 c0,
    const int                                    level ) {
    // Requires exposing the scalar type.
    typename T::ScalarType;

    // Required lincomb overload with 4 args
    { self.lincomb_impl( c, xx, c0, level ) } -> std::same_as< void >;

    // Required dot product
    { self_const.dot_impl( x, level ) } -> std::same_as< typename T::ScalarType >;

    // Required max magnitude
    { self_const.max_magnitude_impl( level ) } -> std::same_as< typename T::ScalarType >;

    // Required nan check
    { self_const.has_nan_impl( level ) } -> std::same_as< bool >;
};

template < VectorLike Vector >
using ScalarOf = typename Vector::ScalarType;

template < VectorLike Vector >
void lincomb(
    Vector&                                  y,
    const std::vector< ScalarOf< Vector > >& c,
    const std::vector< Vector >&             x,
    const ScalarOf< Vector >&                c0,
    const int                                level )
{
    y.lincomb_impl( c, x, c0, level );
}

template < VectorLike Vector >
void lincomb( Vector& y, const std::vector< ScalarOf< Vector > >& c, const std::vector< Vector >& x, const int level )
{
    lincomb( y, c, x, static_cast< ScalarOf< Vector > >( 0 ), level );
}

template < VectorLike Vector >
void assign( Vector& y, const ScalarOf< Vector >& c0, const int level )
{
    lincomb( y, {}, {}, c0, level );
}

template < VectorLike Vector >
void assign( Vector& y, const Vector& x, const int level )
{
    lincomb( y, { static_cast< ScalarOf< Vector > >( 1 ) }, { x }, level );
}

template < VectorLike Vector >
ScalarOf< Vector > dot( const Vector& y, const Vector& x, const int level )
{
    return y.dot_impl( x, level );
}

template < VectorLike Vector >
ScalarOf< Vector > inf_norm( const Vector& y, const int level )
{
    return y.max_magnitude_impl( level );
}

template < VectorLike Vector >
bool has_nan( const Vector& y, const int level )
{
    return y.has_nan_impl( level );
}

namespace detail {

template < typename ScalarT >
class DummyVector
{
  public:
    using ScalarType = ScalarT;

    void lincomb_impl(
        const std::vector< ScalarType >&  c,
        const std::vector< DummyVector >& x,
        const ScalarType                  c0,
        const int                         level )
    {
        (void) c;
        (void) x;
        (void) c0;
        (void) level;
    }

    ScalarType dot_impl( const DummyVector& x, const int level ) const
    {
        (void) x;
        (void) level;
        return 0;
    }

    ScalarType max_magnitude_impl( const int level ) const
    {
        (void) level;
        return 0;
    }

    bool has_nan_impl( const int level ) const
    {
        (void) level;
        return false;
    }
};

static_assert( VectorLike< DummyVector< double > > );

} // namespace detail

} // namespace terra::linalg