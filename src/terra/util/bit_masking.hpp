
#pragma once

#include <concepts>

#include "terra/kokkos/kokkos_wrapper.hpp"

namespace terra::util {

using MaskType = unsigned char;

struct MaskAndValue
{
    const MaskType mask;
    const MaskType value;

    [[nodiscard]] MaskAndValue combine( const MaskAndValue other ) const
    {
        if ( ( mask & other.mask ) != 0 )
        {
            Kokkos::abort( "Masks are overlapping and cannot be combined safely." );
        }
        const MaskType combined_mask  = mask | other.mask;
        const MaskType combined_value = value | other.value;
        return { combined_mask, combined_value };
    }
};

template < std::unsigned_integral ValueType, std::unsigned_integral MaskT >
KOKKOS_INLINE_FUNCTION constexpr void set_bits( ValueType& value, MaskT mask, MaskT masked_field_value )
{
    // Clear the bits in the mask, then set the masked value
    static_assert( sizeof( ValueType ) >= sizeof( MaskT ) );
    value = ( value & ~mask ) | ( masked_field_value & mask );
}

template < std::unsigned_integral ValueType >
KOKKOS_INLINE_FUNCTION constexpr void set_bits( ValueType& value, const MaskAndValue& mask_and_value )
{
    set_bits( value, mask_and_value.mask, mask_and_value.value );
}

template < std::unsigned_integral ValueType, std::unsigned_integral MaskT >
KOKKOS_INLINE_FUNCTION constexpr bool check_bits( ValueType value, MaskT mask, MaskT expected_masked_value )
{
    static_assert( sizeof( ValueType ) >= sizeof( MaskT ) );
    return ( value & mask ) == ( expected_masked_value & mask );
}

template < std::unsigned_integral ValueType >
KOKKOS_INLINE_FUNCTION constexpr bool check_bits( ValueType& value, const MaskAndValue& mask_and_value )
{
    return check_bits( value, mask_and_value.mask, mask_and_value.value );
}

} // namespace terra::util