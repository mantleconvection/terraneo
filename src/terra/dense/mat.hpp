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

    static_assert( Rows > 0 && Cols > 0, "Matrix dimensions must be positive" );

    KOKKOS_INLINE_FUNCTION
    constexpr static Mat from_col_vecs( const Vec< T, Rows >& col0, const Vec< T, Rows >& col1 )
    {
        static_assert( Rows == 2 && Cols == 2, "This constructor is only for 2x2 matrices" );
        Mat mat;
        mat.data[0][0] = col0( 0 );
        mat.data[0][1] = col1( 0 );
        mat.data[1][0] = col0( 1 );
        mat.data[1][1] = col1( 1 );
        return mat;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr static Mat
        from_col_vecs( const Vec< T, Rows >& col0, const Vec< T, Rows >& col1, const Vec< T, Rows >& col2 )
    {
        static_assert( Rows == 3 && Cols == 3, "This constructor is only for 3x3 matrices" );
        Mat mat;
        mat.data[0][0] = col0( 0 );
        mat.data[0][1] = col1( 0 );
        mat.data[0][2] = col2( 0 );
        mat.data[1][0] = col0( 1 );
        mat.data[1][1] = col1( 1 );
        mat.data[1][2] = col2( 1 );
        mat.data[2][0] = col0( 2 );
        mat.data[2][1] = col1( 2 );
        mat.data[2][2] = col2( 2 );
        return mat;
    }

    KOKKOS_INLINE_FUNCTION
    constexpr static Mat diagonal_from_vec( const Vec< T, Rows >& diagonal )
    {
        Mat mat;
        for ( int i = 0; i < Rows; ++i )
        {
            mat( i, i ) = diagonal( i );
        }
        return mat;
    }

    KOKKOS_INLINE_FUNCTION
    T& operator()( int i, int j ) { return data[i][j]; }

    KOKKOS_INLINE_FUNCTION
    const T& operator()( int i, int j ) const { return data[i][j]; }

    // Matrix-matrix multiplication
    KOKKOS_INLINE_FUNCTION
    Mat operator*( const Mat< T, Cols, Cols >& rhs ) const
    {
        Mat< T, Rows, Cols > result;
        for ( int i = 0; i < Rows; ++i )
        {
            for ( int j = 0; j < Cols; ++j )
            {
                result( i, j ) = T( 0 );
                for ( int k = 0; k < Cols; ++k )
                {
                    result( i, j ) += data[i][k] * rhs( k, j );
                }
            }
        }
        return result;
    }

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
    Mat& operator+=( const Mat& mat )
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
    constexpr Mat< T, Cols, Rows > transposed() const
    {
        Mat< T, Cols, Rows > result;
        for ( int i = 0; i < Rows; ++i )
        {
            for ( int j = 0; j < Cols; ++j )
            {
                result( j, i ) = data[i][j];
            }
        }
        return result;
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
    Mat& hadamard_product( const Mat& mat )
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

    KOKKOS_INLINE_FUNCTION
    constexpr T det() const
    {
        if constexpr ( Rows == 2 && Cols == 2 )
        {
            return data[0][0] * data[1][1] - data[0][1] * data[1][0];
        }
        else if constexpr ( Rows == 3 && Cols == 3 )
        {
            return data[0][0] * ( data[1][1] * data[2][2] - data[1][2] * data[2][1] ) -
                   data[0][1] * ( data[1][0] * data[2][2] - data[1][2] * data[2][0] ) +
                   data[0][2] * ( data[1][0] * data[2][1] - data[1][1] * data[2][0] );
        }
        else
        {
            static_assert( Rows == -1, "det() only implemented for 2x2 and 3x3 matrices" );
        }
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Mat inv() const
    {
        if constexpr ( Rows == 2 && Cols == 2 )
        {
            const T d = det();
            if ( d == T( 0 ) )
                Kokkos::abort( "Singular matrix" );
            const T invDet = T( 1 ) / d;
            return { { { data[1][1] * invDet, -data[0][1] * invDet }, { -data[1][0] * invDet, data[0][0] * invDet } } };
        }
        else if constexpr ( Rows == 3 && Cols == 3 )
        {
            const T d = det();
#ifndef NDEBUG
            if ( d == T( 0 ) )
                Kokkos::abort( "Singular matrix" );
#endif
            const T id = T( 1 ) / d;

            Mat< T, 3, 3 > r;
            r( 0, 0 ) = ( data[1][1] * data[2][2] - data[1][2] * data[2][1] ) * id;
            r( 0, 1 ) = -( data[0][1] * data[2][2] - data[0][2] * data[2][1] ) * id;
            r( 0, 2 ) = ( data[0][1] * data[1][2] - data[0][2] * data[1][1] ) * id;

            r( 1, 0 ) = -( data[1][0] * data[2][2] - data[1][2] * data[2][0] ) * id;
            r( 1, 1 ) = ( data[0][0] * data[2][2] - data[0][2] * data[2][0] ) * id;
            r( 1, 2 ) = -( data[0][0] * data[1][2] - data[0][2] * data[1][0] ) * id;

            r( 2, 0 ) = ( data[1][0] * data[2][1] - data[1][1] * data[2][0] ) * id;
            r( 2, 1 ) = -( data[0][0] * data[2][1] - data[0][1] * data[2][0] ) * id;
            r( 2, 2 ) = ( data[0][0] * data[1][1] - data[0][1] * data[1][0] ) * id;
            return r;
        }
        else
        {
            static_assert( Rows == -1, "inv() only implemented for 2x2 and 3x3 matrices" );
        }
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Mat inv_transposed() const
    {
        if constexpr ( Rows == 2 && Cols == 2 )
        {
            const T d = det();
            if ( d == T( 0 ) )
                Kokkos::abort( "Singular matrix" );
            const T invDet = T( 1 ) / d;
            return { { { data[1][1] * invDet, -data[0][1] * invDet }, { -data[1][0] * invDet, data[0][0] * invDet } } };
        }
        else if constexpr ( Rows == 3 && Cols == 3 )
        {
            const T d = det();
#ifndef NDEBUG
            if ( d == T( 0 ) )
                Kokkos::abort( "Singular matrix" );
#endif
            const T id = T( 1 ) / d;

            Mat< T, 3, 3 > r;
            r( 0, 0 ) = ( data[1][1] * data[2][2] - data[1][2] * data[2][1] ) * id;
            r( 0, 1 ) = -( data[1][0] * data[2][2] - data[1][2] * data[2][0] ) * id;
            r( 0, 2 ) = ( data[1][0] * data[2][1] - data[1][1] * data[2][0] ) * id;

            r( 1, 0 ) = -( data[0][1] * data[2][2] - data[0][2] * data[2][1] ) * id;
            r( 1, 1 ) = ( data[0][0] * data[2][2] - data[0][2] * data[2][0] ) * id;
            r( 1, 2 ) = -( data[0][0] * data[2][1] - data[0][1] * data[2][0] ) * id;

            r( 2, 0 ) = ( data[0][1] * data[1][2] - data[0][2] * data[1][1] ) * id;
            r( 2, 1 ) = -( data[0][0] * data[1][2] - data[0][2] * data[1][0] ) * id;
            r( 2, 2 ) = ( data[0][0] * data[1][1] - data[0][1] * data[1][0] ) * id;
            return r;
        }
        else
        {
            static_assert( Rows == -1, "inv() only implemented for 2x2 and 3x3 matrices" );
        }
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Mat inv_transposed( const T& det ) const
    {
        if constexpr ( Rows == 2 && Cols == 2 )
        {
            if ( det == T( 0 ) )
                Kokkos::abort( "Singular matrix" );
            const T invDet = T( 1 ) / det;
            return { { { data[1][1] * invDet, -data[0][1] * invDet }, { -data[1][0] * invDet, data[0][0] * invDet } } };
        }
        else if constexpr ( Rows == 3 && Cols == 3 )
        {
#ifndef NDEBUG
            if ( det == T( 0 ) )
                Kokkos::abort( "Singular matrix" );
#endif
            const T id = T( 1 ) / det;

            Mat< T, 3, 3 > r;
            r( 0, 0 ) = ( data[1][1] * data[2][2] - data[1][2] * data[2][1] ) * id;
            r( 0, 1 ) = -( data[1][0] * data[2][2] - data[1][2] * data[2][0] ) * id;
            r( 0, 2 ) = ( data[1][0] * data[2][1] - data[1][1] * data[2][0] ) * id;

            r( 1, 0 ) = -( data[0][1] * data[2][2] - data[0][2] * data[2][1] ) * id;
            r( 1, 1 ) = ( data[0][0] * data[2][2] - data[0][2] * data[2][0] ) * id;
            r( 1, 2 ) = -( data[0][0] * data[2][1] - data[0][1] * data[2][0] ) * id;

            r( 2, 0 ) = ( data[0][1] * data[1][2] - data[0][2] * data[1][1] ) * id;
            r( 2, 1 ) = -( data[0][0] * data[1][2] - data[0][2] * data[1][0] ) * id;
            r( 2, 2 ) = ( data[0][0] * data[1][1] - data[0][1] * data[1][0] ) * id;
            return r;
        }
        else
        {
            static_assert( Rows == -1, "inv() only implemented for 2x2 and 3x3 matrices" );
        }
    }

    KOKKOS_INLINE_FUNCTION
    constexpr Mat diagonal() const
    {
        Mat result;
        for ( int i = 0; i < Rows; ++i )
        {
            for ( int j = 0; j < Cols; ++j )
            {
                result( i, j ) = ( i == j ) ? data[i][j] : T( 0 );
            }
        }
        return result;
    }
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
