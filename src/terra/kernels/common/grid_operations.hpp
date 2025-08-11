#pragma once

#include "../../kokkos/kokkos_wrapper.hpp"
#include "grid/grid_types.hpp"
#include "terra/util/bit_masking.hpp"

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
void assign_masked_else_keep_old(
    const grid::Grid4DDataScalar< ScalarType >&     dst,
    const ScalarType&                               value,
    const grid::Grid4DDataScalar< util::MaskType >& mask_grid,
    const util::MaskAndValue                        mask_and_value )
{
    Kokkos::parallel_for(
        "assign_masked",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { dst.extent( 0 ), dst.extent( 1 ), dst.extent( 2 ), dst.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            const ScalarType mask_val =
                util::check_bits( mask_grid( local_subdomain, i, j, k ), mask_and_value ) ? 1.0 : 0.0;
            dst( local_subdomain, i, j, k ) = mask_val * value + ( 1.0 - mask_val ) * dst( local_subdomain, i, j, k );
        } );

    Kokkos::fence();
}

template < typename ScalarType >
void assign_masked_else_keep_old(
    const grid::Grid4DDataScalar< ScalarType >&     dst,
    const grid::Grid4DDataScalar< ScalarType >&     src,
    const grid::Grid4DDataScalar< util::MaskType >& mask_grid,
    const util::MaskAndValue                        mask_and_value )
{
    Kokkos::parallel_for(
        "assign_masked",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { dst.extent( 0 ), dst.extent( 1 ), dst.extent( 2 ), dst.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            const ScalarType mask_val =
                util::check_bits( mask_grid( local_subdomain, i, j, k ), mask_and_value ) ? 1.0 : 0.0;
            dst( local_subdomain, i, j, k ) =
                mask_val * src( local_subdomain, i, j, k ) + ( 1.0 - mask_val ) * dst( local_subdomain, i, j, k );
        } );

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

template < typename ScalarType, int VecDim >
void invert_inplace( const grid::Grid4DDataVec< ScalarType, VecDim >& y )
{
    Kokkos::parallel_for(
        "invert",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ), y.extent( 4 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, int d ) {
            y( local_subdomain, i, j, k, d ) = 1.0 / y( local_subdomain, i, j, k, d );
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

template < typename ScalarType, int VecDim >
void mult_elementwise_inplace(
    const grid::Grid4DDataVec< ScalarType, VecDim >& y,
    const grid::Grid4DDataVec< ScalarType, VecDim >& x )
{
    Kokkos::parallel_for(
        "mult_elementwise_inplace",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0, 0 }, { y.extent( 0 ), y.extent( 1 ), y.extent( 2 ), y.extent( 3 ), y.extent( 4 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, int d ) {
            y( local_subdomain, i, j, k, d ) *= x( local_subdomain, i, j, k, d );
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
ScalarType count_masked( const grid::Grid4DDataScalar< MaskType >& mask, const util::MaskAndValue& mask_and_value )
{
    auto count = static_cast< ScalarType >( 0 );

    Kokkos::parallel_reduce(
        "masked_sum",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0 }, { mask.extent( 0 ), mask.extent( 1 ), mask.extent( 2 ), mask.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_sum ) {
            const ScalarType mask_val = util::check_bits( mask( local_subdomain, i, j, k ), mask_and_value ) ?
                                            static_cast< ScalarType >( 1 ) :
                                            static_cast< ScalarType >( 0 );
            local_sum                 = local_sum + mask_val;
        },
        Kokkos::Sum< ScalarType >( count ) );

    Kokkos::fence();

    return count;
}

template < typename ScalarType, typename MaskType >
ScalarType masked_sum(
    const grid::Grid4DDataScalar< ScalarType >& x,
    const grid::Grid4DDataScalar< MaskType >&   mask,
    const util::MaskAndValue&                   mask_and_value )
{
    ScalarType sum = 0.0;

    Kokkos::parallel_reduce(
        "masked_sum",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_sum ) {
            const ScalarType mask_val =
                util::check_bits( mask( local_subdomain, i, j, k ), mask_and_value ) ? 1.0 : 0.0;
            ScalarType val = x( local_subdomain, i, j, k ) * mask_val;
            local_sum      = local_sum + val;
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
    const grid::Grid4DDataScalar< MaskType >&   mask,
    const util::MaskAndValue&                   mask_and_value )
{
    ScalarType dot_prod = 0.0;

    Kokkos::parallel_reduce(
        "masked_dot_product",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, ScalarType& local_dot_prod ) {
            const ScalarType mask_val =
                util::check_bits( mask( local_subdomain, i, j, k ), mask_and_value ) ? 1.0 : 0.0;
            ScalarType val = x( local_subdomain, i, j, k ) * y( local_subdomain, i, j, k ) * mask_val;
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
    const grid::Grid4DDataScalar< MaskType >&        mask,
    const util::MaskAndValue&                        mask_and_value )
{
    ScalarType dot_prod = 0.0;

    Kokkos::parallel_reduce(
        "masked_dot_product",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ), x.extent( 4 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k, int d, ScalarType& local_dot_prod ) {
            const ScalarType mask_val =
                util::check_bits( mask( local_subdomain, i, j, k ), mask_and_value ) ? 1.0 : 0.0;
            ScalarType val = x( local_subdomain, i, j, k, d ) * y( local_subdomain, i, j, k, d ) * mask_val;
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

template < typename ScalarTypeDst >
void rand( const grid::Grid4DDataScalar< ScalarTypeDst >& dst )
{
    static_assert(
        std::is_same_v< ScalarTypeDst, double > || std::is_same_v< ScalarTypeDst, float >,
        "Random integers not implemented. But can be done easily below." );

    Kokkos::Random_XorShift64_Pool<> random_pool( /*seed=*/12345 );
    Kokkos::parallel_for(
        "rand",
        Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { dst.extent( 0 ), dst.extent( 1 ), dst.extent( 2 ), dst.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain, int i, int j, int k ) {
            auto generator                  = random_pool.get_state();
            dst( local_subdomain, i, j, k ) = static_cast< ScalarTypeDst >( generator.drand() );
            random_pool.free_state( generator );
        } );

    Kokkos::fence();
}

} // namespace terra::kernels::common
