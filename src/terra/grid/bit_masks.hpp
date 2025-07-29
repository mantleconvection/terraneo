

#pragma once
#include "terra/util/bit_masking.hpp"

namespace terra::grid {

KOKKOS_INLINE_FUNCTION
constexpr util::MaskAndValue mask_non_owned()
{
    return util::MaskAndValue{ 0b1, 0b0 };
}

KOKKOS_INLINE_FUNCTION
constexpr util::MaskAndValue mask_owned()
{
    return util::MaskAndValue{ 0b1, 0b1 };
}

static_assert( mask_non_owned().mask == mask_owned().mask );
static_assert( mask_non_owned().value != mask_owned().value );


} // namespace terra::grid::shell