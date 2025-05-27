#pragma once

#include "../../kokkos/kokkos_wrapper.hpp"
#include "terra/grid/grid_types.hpp"

namespace terra::kernels::common {

template < typename ScalarType >
void set_constant( const grid::Grid4DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::parallel_for(
        "set_constant (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int subdomain, int i, int j, int k ) { x( subdomain, i, j, k ) = value; } );
}

template < typename ScalarType >
void scale( const grid::Grid3DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::parallel_for(
        "scale (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ) } ),
        KOKKOS_LAMBDA( int i, int j, int k ) { x( i, j, k ) *= value; } );
}

template < typename ScalarType >
void lincomb(
    const grid::Grid3DDataScalar< ScalarType >& y,
    ScalarType                                  c_1,
    const grid::Grid3DDataScalar< ScalarType >& x_1,
    ScalarType                                  c_2,
    const grid::Grid3DDataScalar< ScalarType >& x_2 )
{
    Kokkos::parallel_for(
        "lincomb 2 args (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ) } ),
        KOKKOS_LAMBDA( int i, int j, int k ) { y( i, j, k ) = c_1 * x_1( i, j, k ) + c_2 * x_2( i, j, k ); } );
}

template < typename ScalarType >
void lincomb(
    const grid::Grid3DDataScalar< ScalarType >& y,
    ScalarType                                  c_1,
    const grid::Grid3DDataScalar< ScalarType >& x_1,
    ScalarType                                  c_2,
    const grid::Grid3DDataScalar< ScalarType >& x_2,
    ScalarType                                  c_3,
    const grid::Grid3DDataScalar< ScalarType >& x_3 )
{
    Kokkos::parallel_for(
        "lincomb 3 args (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ) } ),
        KOKKOS_LAMBDA( int i, int j, int k ) {
            y( i, j, k ) = c_1 * x_1( i, j, k ) + c_2 * x_2( i, j, k ) + c_3 * x_3( i, j, k );
        } );
}

template < typename ScalarType >
void axpy(
    ScalarType                                  alpha,
    const grid::Grid3DDataScalar< ScalarType >& x,
    const grid::Grid3DDataScalar< ScalarType >& y )
{
    if ( x.extent( 0 ) != y.extent( 0 ) || x.extent( 1 ) != y.extent( 1 ) || x.extent( 2 ) != y.extent( 2 ) )
    {
        throw std::runtime_error( "axpy: x and y must have the same extent" );
    }

    Kokkos::parallel_for(
        "axpy (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ) } ),
        KOKKOS_LAMBDA( int i, int j, int k ) { y( i, j, k ) += alpha * x( i, j, k ); } );
}

} // namespace terra::kernels::common
