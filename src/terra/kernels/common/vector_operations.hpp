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
void scale( const grid::Grid4DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::parallel_for(
        "scale (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) { x( local_subdomain, i, j, k ) *= value; } );
}

template < typename ScalarType >
void lincomb(
    const grid::Grid4DDataScalar< ScalarType >& y,
    ScalarType                                  c_1,
    const grid::Grid4DDataScalar< ScalarType >& x_1,
    ScalarType                                  c_2,
    const grid::Grid4DDataScalar< ScalarType >& x_2 )
{
    Kokkos::parallel_for(
        "lincomb 2 args (Grid4DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) =
                c_1 * x_1( local_subdomain, i, j, k ) + c_2 * x_2( local_subdomain, i, j, k );
        } );
}

template < typename ScalarType >
void lincomb(
    const grid::Grid4DDataScalar< ScalarType >& y,
    ScalarType                                  c_1,
    const grid::Grid4DDataScalar< ScalarType >& x_1,
    ScalarType                                  c_2,
    const grid::Grid4DDataScalar< ScalarType >& x_2,
    ScalarType                                  c_3,
    const grid::Grid4DDataScalar< ScalarType >& x_3 )
{
    Kokkos::parallel_for(
        "lincomb 3 args (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) = c_1 * x_1( local_subdomain, i, j, k ) +
                                            c_2 * x_2( local_subdomain, i, j, k ) +
                                            c_3 * x_3( local_subdomain, i, j, k );
        } );
}

template < typename ScalarType >
ScalarType max_magnitude( const grid::Grid4DDataScalar< ScalarType >& x )
{
    ScalarType max_mag = 0.0;
    Kokkos::parallel_reduce(
        "lincomb 3 args (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_max ) {
            ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k ) );
            local_max      = Kokkos::max( local_max, val );
        },
        Kokkos::Max< ScalarType >( max_mag ) );
    return max_mag;
}

} // namespace terra::kernels::common
