#pragma once

#include "../../kokkos/kokkos_wrapper.hpp"
#include "grid/grid_types.hpp"

namespace terra::kernels::common {

template < typename ScalarType >
void set_constant( const grid::Grid3DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::parallel_for(
        "set_constant (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ) } ),
        KOKKOS_LAMBDA( int i, int j, int k ) { x( i, j, k ) = value; } );

    Kokkos::fence();
}

template < typename ScalarType >
void set_constant( const grid::Grid4DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::parallel_for(
        "set_constant (Grid4DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int subdomain, int i, int j, int k ) { x( subdomain, i, j, k ) = value; } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim >
void set_constant( const grid::Grid4DDataVec< ScalarType, VecDim >& x, ScalarType value )
{
    Kokkos::parallel_for(
        "set_constant (Grid4DDataVec)",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ), x.extent( 4 ) } ),
        KOKKOS_LAMBDA( int subdomain, int i, int j, int k, int d ) { x( subdomain, i, j, k, d ) = value; } );

    Kokkos::fence();
}

template < typename ScalarType >
void scale( const grid::Grid4DDataScalar< ScalarType >& x, ScalarType value )
{
    Kokkos::parallel_for(
        "scale (Grid3DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) { x( local_subdomain, i, j, k ) *= value; } );

    Kokkos::fence();
}

template < typename ScalarType >
void lincomb(
    const grid::Grid4DDataScalar< ScalarType >& y,
    ScalarType                                  c_0,
    ScalarType                                  c_1,
    const grid::Grid4DDataScalar< ScalarType >& x_1 )
{
    Kokkos::parallel_for(
        "lincomb 1 arg (Grid4DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) = c_0 + c_1 * x_1( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType >
void lincomb(
    const grid::Grid4DDataScalar< ScalarType >& y,
    ScalarType                                  c_0,
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
                c_0 + c_1 * x_1( local_subdomain, i, j, k ) + c_2 * x_2( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType >
void lincomb(
    const grid::Grid4DDataScalar< ScalarType >& y,
    ScalarType                                  c_0,
    ScalarType                                  c_1,
    const grid::Grid4DDataScalar< ScalarType >& x_1,
    ScalarType                                  c_2,
    const grid::Grid4DDataScalar< ScalarType >& x_2,
    ScalarType                                  c_3,
    const grid::Grid4DDataScalar< ScalarType >& x_3 )
{
    Kokkos::parallel_for(
        "lincomb 3 args (Grid4DDataScalar)",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) = c_0 + c_1 * x_1( local_subdomain, i, j, k ) +
                                            c_2 * x_2( local_subdomain, i, j, k ) +
                                            c_3 * x_3( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim >
void lincomb(
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    ScalarType                                       c_0,
    ScalarType                                       c_1,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_1 )
{
    Kokkos::parallel_for(
        "lincomb 1 arg (Grid4DDataVec)",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ), y.extent( 4 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, int d ) {
            y( local_subdomain, i, j, k, d ) = c_0 + c_1 * x_1( local_subdomain, i, j, k, d );
        } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim >
void lincomb(
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    ScalarType                                       c_0,
    ScalarType                                       c_1,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_1,
    ScalarType                                       c_2,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_2 )
{
    Kokkos::parallel_for(
        "lincomb 2 args (Grid4DDataVec)",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ), y.extent( 4 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, int d ) {
            y( local_subdomain, i, j, k, d ) =
                c_0 + c_1 * x_1( local_subdomain, i, j, k, d ) + c_2 * x_2( local_subdomain, i, j, k, d );
        } );

    Kokkos::fence();
}

template < typename ScalarType, int VecDim >
void lincomb(
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    ScalarType                                       c_0,
    ScalarType                                       c_1,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_1,
    ScalarType                                       c_2,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_2,
    ScalarType                                       c_3,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x_3 )
{
    Kokkos::parallel_for(
        "lincomb 3 args (Grid4DDataVec)",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ), y.extent( 4 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, int d ) {
            y( local_subdomain, i, j, k, d ) = c_0 + c_1 * x_1( local_subdomain, i, j, k, d ) +
                                               c_2 * x_2( local_subdomain, i, j, k, d ) +
                                               c_3 * x_3( local_subdomain, i, j, k, d );
        } );

    Kokkos::fence();
}

template < typename ScalarType >
void invert_inplace( const grid::Grid4DDataScalar< ScalarType >& y )
{
    Kokkos::parallel_for(
        "invert",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) = 1.0 / y( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType >
void mult_elementwise_inplace(
    const grid::Grid4DDataScalar< ScalarType >& y,
    const grid::Grid4DDataScalar< ScalarType >& x )
{
    Kokkos::parallel_for(
        "mult_elementwise_inplace",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            y( local_subdomain, i, j, k ) *= x( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType >
ScalarType min_entry( const grid::Grid4DDataScalar< ScalarType >& x )
{
    ScalarType min_val = 0.0;
    Kokkos::parallel_reduce(
        "min_entry",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_min ) {
            ScalarType val = x( local_subdomain, i, j, k );
            local_min      = Kokkos::min( local_min, val );
        },
        Kokkos::Min< ScalarType >( min_val ) );

    Kokkos::fence();

    return min_val;
}

template < typename ScalarType >
ScalarType min_abs_entry( const grid::Grid4DDataScalar< ScalarType >& x )
{
    ScalarType min_mag = 0.0;
    Kokkos::parallel_reduce(
        "min_abs_entry",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_min ) {
            ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k ) );
            local_min      = Kokkos::min( local_min, val );
        },
        Kokkos::Min< ScalarType >( min_mag ) );

    Kokkos::fence();

    return min_mag;
}

template < typename ScalarType >
ScalarType max_abs_entry( const grid::Grid4DDataScalar< ScalarType >& x )
{
    ScalarType max_mag = 0.0;
    Kokkos::parallel_reduce(
        "max_abs_entry",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_max ) {
            ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k ) );
            local_max      = Kokkos::max( local_max, val );
        },
        Kokkos::Max< ScalarType >( max_mag ) );

    Kokkos::fence();

    return max_mag;
}

template < typename ScalarType, int VecDim >
ScalarType max_abs_entry( const grid::Grid4DDataVec< ScalarType, VecDim >& x )
{
    ScalarType max_mag = 0.0;
    Kokkos::parallel_reduce(
        "max_abs_entry",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ), x.extent( 4 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, int d, ScalarType& local_max ) {
            ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k, d ) );
            local_max      = Kokkos::max( local_max, val );
        },
        Kokkos::Max< ScalarType >( max_mag ) );

    Kokkos::fence();

    return max_mag;
}

template < typename ScalarType >
ScalarType sum_of_absolutes( const grid::Grid4DDataScalar< ScalarType >& x )
{
    ScalarType sum_abs = 0.0;
    Kokkos::parallel_reduce(
        "sum_of_absolutes",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_sum_abs ) {
            ScalarType val = Kokkos::abs( x( local_subdomain, i, j, k ) );
            local_sum_abs  = local_sum_abs + val;
        },
        Kokkos::Sum< ScalarType >( sum_abs ) );

    Kokkos::fence();

    return sum_abs;
}

template < typename ScalarType, typename MaskType >
ScalarType masked_sum( const grid::Grid4DDataScalar< ScalarType >& x, const grid::Grid4DDataScalar< MaskType >& mask )
{
    ScalarType sum = 0.0;

    Kokkos::parallel_reduce(
        "masked_sum",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_sum ) {
            ScalarType val =
                x( local_subdomain, i, j, k ) * static_cast< ScalarType >( mask( local_subdomain, i, j, k ) );
            local_sum = local_sum + val;
        },
        Kokkos::Sum< ScalarType >( sum ) );

    Kokkos::fence();

    return sum;
}

template < typename ScalarType >
ScalarType dot_product( const grid::Grid4DDataScalar< ScalarType >& x, const grid::Grid4DDataScalar< ScalarType >& y )
{
    ScalarType dot_prod = 0.0;

    Kokkos::parallel_reduce(
        "dot_product",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_dot_prod ) {
            ScalarType val = x( local_subdomain, i, j, k ) * y( local_subdomain, i, j, k );
            local_dot_prod = local_dot_prod + val;
        },
        Kokkos::Sum< ScalarType >( dot_prod ) );

    Kokkos::fence( "dot_product" );

    return dot_prod;
}

template < typename ScalarType, typename MaskType >
ScalarType masked_dot_product(
    const grid::Grid4DDataScalar< ScalarType >& x,
    const grid::Grid4DDataScalar< ScalarType >& y,
    const grid::Grid4DDataScalar< MaskType >&   mask )
{
    ScalarType dot_prod = 0.0;

    Kokkos::parallel_reduce(
        "masked_dot_product",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_dot_prod ) {
            ScalarType val = x( local_subdomain, i, j, k ) * y( local_subdomain, i, j, k ) *
                             static_cast< ScalarType >( mask( local_subdomain, i, j, k ) );
            local_dot_prod = local_dot_prod + val;
        },
        Kokkos::Sum< ScalarType >( dot_prod ) );

    Kokkos::fence( "masked_dot_product" );

    return dot_prod;
}

template < typename ScalarType, typename MaskType, int VecDim >
ScalarType masked_dot_product(
    const grid::Grid4DDataVec< ScalarType, VecDim >& x,
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    const grid::Grid4DDataScalar< MaskType >&        mask )
{
    ScalarType dot_prod = 0.0;

    Kokkos::parallel_reduce(
        "masked_dot_product",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ), x.extent( 4 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, int d, ScalarType& local_dot_prod ) {
            ScalarType val = x( local_subdomain, i, j, k, d ) * y( local_subdomain, i, j, k, d ) *
                             static_cast< ScalarType >( mask( local_subdomain, i, j, k ) );
            local_dot_prod = local_dot_prod + val;
        },
        Kokkos::Sum< ScalarType >( dot_prod ) );

    Kokkos::fence( "masked_dot_product" );

    return dot_prod;
}

template < typename ScalarType >
bool has_nan( const grid::Grid4DDataScalar< ScalarType >& x )
{
    bool has_nan = false;

    Kokkos::parallel_reduce(
        "masked_dot_product",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, bool& local_has_nan ) {
            local_has_nan = local_has_nan || Kokkos::isnan( x( local_subdomain, i, j, k ) );
        },
        Kokkos::LOr< bool >( has_nan ) );

    Kokkos::fence();

    return has_nan;
}

template < typename ScalarTypeDst, typename ScalarTypeSrc >
void cast( const grid::Grid4DDataScalar< ScalarTypeDst >& dst, const grid::Grid4DDataScalar< ScalarTypeSrc >& src )
{
    Kokkos::parallel_for(
        "cast",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { dst.extent( 0 ), dst.extent( 1 ), dst.extent( 2 ), dst.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            dst( local_subdomain, i, j, k ) = static_cast< ScalarTypeDst >( src( local_subdomain, i, j, k ) );
        } );

    Kokkos::fence();
}

} // namespace terra::kernels::common
