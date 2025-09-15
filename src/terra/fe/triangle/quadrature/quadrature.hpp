
#pragma once

#include "terra/dense/vec.hpp"

namespace terra::fe::triangle::quadrature {

constexpr int quad_triangle_3_num_quad_points = 3;

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void
    quad_triangle_3_quad_points( dense::Vec< T, 2 > ( &quad_points )[quad_triangle_3_num_quad_points] )
{
    quad_points[0] = { 0.6666666666666666, 0.1666666666666667 };
    quad_points[1] = { 0.1666666666666667, 0.6666666666666666 };
    quad_points[2] = { 0.1666666666666667, 0.1666666666666667 };
}

template < std::floating_point T >
KOKKOS_INLINE_FUNCTION constexpr void
    quad_triangle_3_quad_weights( T ( &quad_weights )[quad_triangle_3_num_quad_points] )
{
    quad_weights[0] = 0.1666666666666667;
    quad_weights[1] = 0.1666666666666667;
    quad_weights[2] = 0.1666666666666667;
}

} // namespace terra::fe::triangle::quadrature