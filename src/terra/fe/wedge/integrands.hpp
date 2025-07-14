
#pragma once

#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"

/// Relevant integrands for wedge elements.
///
/// Geometry:
///
///   xi, eta ∈ [0, 1] (lateral reference coords)
///   zeta ∈ [-1, 1]   (radial reference coords)
///
///   0 <= xi + eta <= 1
///
/// Node enumeration:
///
///   r_node_idx = r_cell_idx + 1 (outer):
///
///   5
///   |\
///   | \
///   3--4
///
///   r_node_idx = r_cell_idx (inner):
///
///   2
///   |\
///   | \
///   0--1
///
/// Enumeration of shape functions:
///
///   N_lat:
///     N_lat_0 = N_lat_3 = 1 - xi - eta
///     N_lat_1 = N_lat_4 = xi
///     N_lat_2 = N_lat_5 = eta
///
///   N_rad:
///     N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
///     N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
///
///   N_i = N_lat_i * N_rad_i
///
/// Physical coordinates:
///
///   r_1, r_2                     radii of bottom and top (r_1 < r_2)
///   p1_phy, p2_phy, p3_phy       coords of triangle on unit sphere
namespace terra::fe::wedge {

/// @brief Radial shape function.
///
/// N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
/// N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
KOKKOS_INLINE_FUNCTION
constexpr double shape_rad( const int node_idx, const double zeta )
{
    const double N_rad[2] = { 0.5 * ( 1 - zeta ), 0.5 * ( 1 + zeta ) };
    return N_rad[node_idx / 3];
}

/// @brief Radial shape function.
///
/// N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
/// N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
KOKKOS_INLINE_FUNCTION
constexpr double shape_rad( const int node_idx, const dense::Vec< double, 3 >& xi_eta_zeta )
{
    return shape_rad( node_idx, xi_eta_zeta( 2 ) );
}

/// @brief Lateral shape function.
///
/// N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
/// N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
KOKKOS_INLINE_FUNCTION
constexpr double shape_lat( const int node_idx, const double xi, const double eta )
{
    const double N_lat[3] = { 1.0 - xi - eta, xi, eta };
    return N_lat[node_idx % 3];
}

/// @brief Lateral shape function.
///
/// N_rad_0 = N_rad_1 = N_rad_2 = 0.5 ( 1 - zeta )
/// N_rad_3 = N_rad_4 = N_rad_5 = 0.5 ( 1 + zeta )
KOKKOS_INLINE_FUNCTION
constexpr double shape_lat( const int node_idx, const dense::Vec< double, 3 >& xi_eta_zeta )
{
    return shape_lat( node_idx, xi_eta_zeta( 0 ), xi_eta_zeta( 1 ) );
}

/// @brief (Tensor-product) Shape function N_j = N_lat_j * N_rad_j.
KOKKOS_INLINE_FUNCTION
constexpr double shape( const int node_idx, const double xi, const double eta, const double zeta )
{
    return shape_lat( node_idx, xi, eta ) * shape_rad( node_idx, zeta );
}

/// @brief (Tensor-product) Shape function N_j = N_lat_j * N_rad_j.
KOKKOS_INLINE_FUNCTION
constexpr double shape( const int node_idx, const dense::Vec< double, 3 >& xi_eta_zeta )
{
    return shape_lat( node_idx, xi_eta_zeta ) * shape_rad( node_idx, xi_eta_zeta );
}

KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_rad( const int node_idx )
{
    constexpr double grad_N_rad[2] = { -0.5, 0.5 };
    return grad_N_rad[node_idx / 3];
}

KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_lat_xi( const int node_idx )
{
    constexpr double grad_N_lat_xi[3] = { -1.0, 1.0, 0.0 };
    return grad_N_lat_xi[node_idx % 3];
}

KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_lat_eta( const int node_idx )
{
    constexpr double grad_N_lat_eta[3] = { -1.0, 0.0, 1.0 };
    return grad_N_lat_eta[node_idx % 3];
}

KOKKOS_INLINE_FUNCTION
constexpr double forward_map_rad( const double r_1, const double r_2, const double zeta )
{
    return r_1 + 0.5 * ( r_2 - r_1 ) * ( 1 + zeta );
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > forward_map_lat(
    const dense::Vec< double, 3 >& p1_phy,
    const dense::Vec< double, 3 >& p2_phy,
    const dense::Vec< double, 3 >& p3_phy,
    const double                   xi,
    const double                   eta )
{
    return ( 1 - xi - eta ) * p1_phy + xi * p2_phy + eta * p3_phy;
}

KOKKOS_INLINE_FUNCTION
constexpr double grad_forward_map_rad( const double r_1, const double r_2 )
{
    return 0.5 * ( r_2 - r_1 );
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > grad_forward_map_lat_xi(
    const dense::Vec< double, 3 >& p1_phy,
    const dense::Vec< double, 3 >& p2_phy,
    const dense::Vec< double, 3 >& /* p3_phy */ )
{
    return p2_phy - p1_phy;
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > grad_forward_map_lat_eta(
    const dense::Vec< double, 3 >& p1_phy,
    const dense::Vec< double, 3 >& /* p2_phy */,
    const dense::Vec< double, 3 >& p3_phy )
{
    return p3_phy - p1_phy;
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Mat< double, 3, 3 > jac_lat(
    const dense::Vec< double, 3 >& p1_phy,
    const dense::Vec< double, 3 >& p2_phy,
    const dense::Vec< double, 3 >& p3_phy,
    const double                   xi,
    const double                   eta )
{
    const auto col_0 = grad_forward_map_lat_xi( p1_phy, p2_phy, p3_phy );
    const auto col_1 = grad_forward_map_lat_eta( p1_phy, p2_phy, p3_phy );
    const auto col_2 = forward_map_lat( p1_phy, p2_phy, p3_phy, xi, eta );
    return dense::Mat< double, 3, 3 >::from_col_vecs( col_0, col_1, col_2 );
}

KOKKOS_INLINE_FUNCTION
constexpr dense::Vec< double, 3 > jac_rad( const double r_1, const double r_2, const double zeta )
{
    const auto r      = forward_map_rad( r_1, r_2, zeta );
    const auto grad_r = grad_forward_map_rad( r_1, r_2 );
    return dense::Vec< double, 3 >{ r, r, grad_r };
}

} // namespace terra::fe::wedge