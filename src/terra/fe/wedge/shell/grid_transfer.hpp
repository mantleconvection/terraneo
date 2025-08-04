
#pragma once
#include "dense/vec.hpp"

namespace terra::fe::wedge::shell {

KOKKOS_INLINE_FUNCTION
constexpr void prolongation_fine_grid_stencil_offsets_at_coarse_vertex( dense::Vec< int, 3 > ( &offsets )[21] )
{
    for ( int r = -1; r <= 1; ++r )
    {
        const auto index_offset   = ( r + 1 ) * 7;
        offsets[index_offset + 0] = { -1, 0, r };
        offsets[index_offset + 1] = { -1, 1, r };
        offsets[index_offset + 2] = { 0, 1, r };
        offsets[index_offset + 3] = { 1, 0, r };
        offsets[index_offset + 4] = { 1, -1, r };
        offsets[index_offset + 5] = { 0, -1, r };
        offsets[index_offset + 6] = { 0, 0, r };
    }
}

template < typename ScalarType >
KOKKOS_INLINE_FUNCTION constexpr ScalarType prolongation_weight(
    const int x_fine,
    const int y_fine,
    const int r_fine,
    const int x_coarse,
    const int y_coarse,
    const int r_coarse )
{
    if ( r_fine == 2 * r_coarse )
    {
        if ( x_fine == 2 * x_coarse && y_fine == 2 * y_coarse )
        {
            return 1.0;
        }

        if ( y_fine == 2 * y_coarse && ( x_fine - 2 * x_coarse == 1 || x_fine - 2 * x_coarse == -1 ) )
        {
            return 0.5;
        }

        if ( x_fine == 2 * x_coarse && ( y_fine - 2 * y_coarse == 1 || y_fine - 2 * y_coarse == -1 ) )
        {
            return 0.5;
        }

        if ( x_fine - 2 * x_coarse == -1 && y_fine - 2 * y_coarse == 1 )
        {
            return 0.5;
        }

        if ( x_fine - 2 * x_coarse == 1 && y_fine - 2 * y_coarse == -1 )
        {
            return 0.5;
        }
    }

    if ( r_fine - 2 * r_coarse == -1 || r_fine - 2 * r_coarse == 1 )
    {
        if ( x_fine == 2 * x_coarse && y_fine == 2 * y_coarse )
        {
            return 0.5;
        }

        if ( y_fine == 2 * y_coarse && ( x_fine - 2 * x_coarse == 1 || x_fine - 2 * x_coarse == -1 ) )
        {
            return 0.25;
        }

        if ( x_fine == 2 * x_coarse && ( y_fine - 2 * y_coarse == 1 || y_fine - 2 * y_coarse == -1 ) )
        {
            return 0.25;
        }

        if ( x_fine - 2 * x_coarse == -1 && y_fine - 2 * y_coarse == 1 )
        {
            return 0.25;
        }

        if ( x_fine - 2 * x_coarse == 1 && y_fine - 2 * y_coarse == -1 )
        {
            return 0.25;
        }
    }

    return 0.0;
}

} // namespace terra::fe::wedge::shell