#pragma once

#include "../terra/kokkos/kokkos_wrapper.hpp"

namespace terra::dense {

template < typename T, int N >
struct Vec
{
    T data[N] = {};

    KOKKOS_INLINE_FUNCTION
    constexpr T& operator()( int i ) { return data[i]; }

    KOKKOS_INLINE_FUNCTION
    constexpr const T& operator()( int i ) const { return data[i]; }

    void fill( const T value )
    {
        for ( int i = 0; i < N; ++i )
        {
            data[i] = value;
        }
    }

    KOKKOS_INLINE_FUNCTION
    T dot( const Vec& other ) const
    {
        T sum = 0;
        for ( int i = 0; i < N; ++i )
            sum += data[i] * other( i );
        return sum;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Vec operator+( const Vec& rhs ) const
    {
        Vec out;
        for ( int i = 0; i < N; ++i )
            out( i ) = data[i] + rhs( i );
        return out;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Vec operator-( const Vec& rhs ) const
    {
        Vec out;
        for ( int i = 0; i < N; ++i )
            out( i ) = data[i] - rhs( i );
        return out;
    }

    KOKKOS_INLINE_FUNCTION
    T norm() const
    {
        T sum = 0;
        for ( int i = 0; i < N; ++i )
        {
            sum += data[i] * data[i];
        }
        return Kokkos::sqrt( sum );
    }

    /// @brief Normalize the vector to unit length in-place.
    KOKKOS_INLINE_FUNCTION
    void normalize()
    {
        T mag = norm();
        if ( mag > 1e-15 )
        { // Avoid division by zero
            for ( int i = 0; i < N; ++i )
            {
                data[i] /= mag;
            }
        }
    }

    /// @brief Return a normalized copy of the vector.
    KOKKOS_INLINE_FUNCTION
    Vec normalized() const
    {
        Vec res = *this; // Make a copy
        res.normalize();
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    Vec inverted_elementwise() const
    {
        Vec res;
        for ( int i = 0; i < N; ++i )
            res( i ) = 1.0 / data[i];
        return res;
    }
};


template < typename T, int N >
KOKKOS_INLINE_FUNCTION
constexpr Vec< T, N > operator*( const Vec< T, N >& v, const T scalar ) noexcept
{
    Vec< T, N > result{};
    for ( int i = 0; i < N; ++i )
        result( i ) = v( i ) * scalar;
    return result;
}

template < typename T, int N >
KOKKOS_INLINE_FUNCTION
constexpr Vec< T, N > operator*( const T scalar, const Vec< T, N >& v ) noexcept
{
    Vec< T, N > result{};
    for ( int i = 0; i < N; ++i )
        result( i ) = v( i ) * scalar;
    return result;
}

template < typename T, int N >
std::ostream& operator<<( std::ostream& os, const Vec< T, N >& v )
{
    for ( int i = 0; i < N; ++i )
    {
        os << v( i ) << '\n';
    }
    return os;
}

using Vec3 = Vec< double, 3 >;

} // namespace terra::dense
