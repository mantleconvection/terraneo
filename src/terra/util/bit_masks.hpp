
#pragma once

namespace terra::util {

using MaskType = unsigned char;

struct MaskAndValue
{
    const MaskType mask;
    const MaskType value;
};

template < std::unsigned_integral ValueType, std::unsigned_integral MaskT >
constexpr void set_bits( ValueType& value, MaskT mask, MaskT masked_field_value )
{
    // Clear the bits in the mask, then set the masked value
    static_assert( sizeof( ValueType ) >= sizeof( MaskT ) );
    value = ( value & ~mask ) | ( masked_field_value & mask );
}

template < std::unsigned_integral ValueType >
constexpr void set_bits( ValueType& value, const MaskAndValue& mask_and_value )
{
    set_bits( value, mask_and_value.mask, mask_and_value.value );
}

template < std::unsigned_integral ValueType, std::unsigned_integral MaskT >
constexpr bool check_bits( ValueType value, MaskT mask, MaskT expected_masked_value )
{
    static_assert( sizeof( ValueType ) >= sizeof( MaskT ) );
    return ( value & mask ) == ( expected_masked_value & mask );
}

template < std::unsigned_integral ValueType >
constexpr bool check_bits( ValueType& value, const MaskAndValue& mask_and_value )
{
    return check_bits( value, mask_and_value.mask, mask_and_value.value );
}

} // namespace terra::util