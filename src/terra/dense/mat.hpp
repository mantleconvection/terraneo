#pragma once

#include "../kokkos/kokkos_wrapper.hpp"
#include "terra/dense/vec.hpp"

namespace terra::dense {

template < typename T, int Rows, int Cols >
struct Mat
{
    T                    data[Rows][Cols] = {};
    static constexpr int rows             = Rows;
    static constexpr int cols             = Cols;

    KOKKOS_INLINE_FUNCTION
    T& operator()( int i, int j ) { return data[i][j]; }

    KOKKOS_INLINE_FUNCTION
    const T& operator()( int i, int j ) const { return data[i][j]; }

    // Matrix-vector multiplication
    KOKKOS_INLINE_FUNCTION
    Vec< T, Rows > operator*( const Vec< T, Cols >& vec ) const
    {
        Vec< T, Rows > result;
        for ( int i = 0; i < Rows; ++i )
        {
            result( i ) = 0;
            for ( int j = 0; j < Cols; ++j )
            {
                result( i ) += data[i][j] * vec( j );
            }
        }
        return result;
    }

    KOKKOS_INLINE_FUNCTION
    Mat< double, Rows, Cols >& operator+=( const Mat& mat )
    {
        for ( int i = 0; i < Rows; ++i )
        {
            for ( int j = 0; j < Cols; ++j )
            {
                data[i][j] += mat.data[i][j];
            }
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    void fill( const T value )
    {
        for ( int i = 0; i < Rows; ++i )
        {
            for ( int j = 0; j < Cols; ++j )
            {
                data[i][j] = value;
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    Mat< double, Rows, Cols >& hadamard_product( const Mat& mat )
    {
        for ( int i = 0; i < Rows; ++i )
        {
            for ( int j = 0; j < Cols; ++j )
            {
                data[i][j] *= mat.data[i][j];
            }
        }
        return *this;
    }

    Mat() = default;
};

template < typename T, int Rows, int Cols >
std::ostream& operator<<( std::ostream& os, const Mat< T, Rows, Cols >& A )
{
    for ( int i = 0; i < A.rows; ++i )
    {
        for ( int j = 0; j < A.cols; ++j )
        {
            os << A( i, j ) << " ";
        }
        os << '\n';
    }
    return os;
}

} // namespace terra::dense
