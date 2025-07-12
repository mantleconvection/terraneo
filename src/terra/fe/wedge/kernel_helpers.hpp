
#pragma once

namespace terra::fe::wedge {

constexpr int num_wedges_per_hex_cell     = 2;
constexpr int num_nodes_per_wedge_surface = 3;
constexpr int num_nodes_per_wedge         = 6;

/// @brief Extracts the (unit sphere) surface vertex coords of the two wedges of a hex cell.
///
/// Useful for wedge-based kernels that update two wedges that make up a hex cell at once.
///
/// 2--3
/// |\ |
/// | \|   =>  [(p0, p1, p2), (p3, p2, p1)]
/// 0--1
///
/// @param wedge_surf_phy_coords [out] first dim: wedge/triangle index, second dim: vertex index
/// @param lateral_grid          [in]  the unit sphere vertex coordinates
/// @param local_subdomain_id    [in]  shell subdomain id on this process
/// @param x_cell                [in]  hex cell x-coordinate
/// @param y_cell                [in]  hex cell y-coordinate
KOKKOS_INLINE_FUNCTION
void wedge_physical_coords(
    dense::Vec< double, 3 > ( &wedge_surf_phy_coords )[num_wedges_per_hex_cell][num_nodes_per_wedge_surface],
    const grid::Grid3DDataVec< double, 3 >& lateral_grid,
    const int                               local_subdomain_id,
    const int                               x_cell,
    const int                               y_cell )
{
    // Extract vertex positions of quad
    // (0, 0), (1, 0), (0, 1), (1, 1).
    dense::Vec< double, 3 > quad_surface_coords[2][2];

    for ( int x = x_cell; x <= x_cell + 1; x++ )
    {
        for ( int y = y_cell; y <= y_cell + 1; y++ )
        {
            for ( int d = 0; d < 3; d++ )
            {
                quad_surface_coords[x - x_cell][y - y_cell]( d ) = lateral_grid( local_subdomain_id, x, y, d );
            }
        }
    }

    // Sort coords for the two wedge surfaces.
    wedge_surf_phy_coords[0][0] = quad_surface_coords[0][0];
    wedge_surf_phy_coords[0][1] = quad_surface_coords[1][0];
    wedge_surf_phy_coords[0][2] = quad_surface_coords[0][1];

    wedge_surf_phy_coords[1][0] = quad_surface_coords[1][1];
    wedge_surf_phy_coords[1][1] = quad_surface_coords[0][1];
    wedge_surf_phy_coords[1][2] = quad_surface_coords[1][0];
}

/// @brief Evaluate the lateral shape function of a specific wedge node index at a quadrature point.
KOKKOS_INLINE_FUNCTION
constexpr double shape_lat_wedge_node( const int node_idx, const dense::Vec< double, 3 >& quad_point )
{
    return shape_lat( quad_point( 0 ), quad_point( 1 ) )( node_idx % 3 );
}

/// @brief Evaluate the radial shape function of a specific wedge node index at a quadrature point.
KOKKOS_INLINE_FUNCTION
constexpr double shape_rad_wedge_node( const int node_idx, const dense::Vec< double, 3 >& quad_point )
{
    return shape_rad( quad_point( 2 ) )( node_idx / 3 );
}

/// @brief Evaluate the (constant) gradient in xi direction of the lateral shape function of a specific wedge node index.
KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_lat_xi_wedge_node( const int node_idx )
{
    return grad_shape_lat_xi()( node_idx % 3 );
}

/// @brief Evaluate the (constant) gradient in eta direction of the lateral shape function of a specific wedge node index.
KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_lat_eta_wedge_node( const int node_idx )
{
    return grad_shape_lat_eta()( node_idx % 3 );
}

/// @brief Evaluate the (constant) gradient of the radial shape function of a specific wedge node index.
KOKKOS_INLINE_FUNCTION
constexpr double grad_shape_rad_wedge_node( const int node_idx )
{
    return grad_shape_rad()( node_idx / 3 );
}

/// @brief Extracts the local vector coefficients for the two wedges of a hex cell from the global coefficient vector.
///
/// r = r_cell + 1 (outer)
/// 6--7
/// |\ |
/// | \|
/// 4--5
///
/// r = r_cell (inner)
/// 2--3
/// |\ |
/// | \|
/// 0--1
///
/// v0 = (0, 1, 2, 4, 5, 6)
/// v1 = (3, 2, 1, 7, 6, 5)
///
/// @param local_coefficients  [out] the local coefficient vector
/// @param local_subdomain_id  [in]  shell subdomain id on this process
/// @param x_cell              [in]  hex cell x-coordinate
/// @param y_cell              [in]  hex cell y-coordinate
/// @param r_cell              [in]  hex cell r-coordinate
/// @param global_coefficients [in]  the global coefficient vector
KOKKOS_INLINE_FUNCTION
void extract_local_wedge_scalar_coefficients(
    dense::Vec< double, 6 > ( &local_coefficients )[2],
    const int                               local_subdomain_id,
    const int                               x_cell,
    const int                               y_cell,
    const int                               r_cell,
    const grid::Grid4DDataScalar< double >& global_coefficients )
{
    local_coefficients[0]( 0 ) = global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell );
    local_coefficients[0]( 1 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell );
    local_coefficients[0]( 2 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell );
    local_coefficients[0]( 3 ) = global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell + 1 );
    local_coefficients[0]( 4 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
    local_coefficients[0]( 5 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );

    local_coefficients[1]( 0 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell );
    local_coefficients[1]( 1 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell );
    local_coefficients[1]( 2 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell );
    local_coefficients[1]( 3 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 );
    local_coefficients[1]( 4 ) = global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
    local_coefficients[1]( 5 ) = global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
}

/// @brief Performs an atomic add of the two local wedge coefficient vectors of a hex cell into the global coefficient
/// vector.
///
/// r = r_cell + 1 (outer)
/// 6--7
/// |\ |
/// | \|
/// 4--5
///
/// r = r_cell (inner)
/// 2--3
/// |\ |
/// | \|
/// 0--1
///
/// v0 = (0, 1, 2, 4, 5, 6)
/// v1 = (3, 2, 1, 7, 6, 5)
///
/// @param global_coefficients [inout] the global coefficient vector
/// @param local_subdomain_id  [in]    shell subdomain id on this process
/// @param x_cell              [in]    hex cell x-coordinate
/// @param y_cell              [in]    hex cell y-coordinate
/// @param r_cell              [in]    hex cell r-coordinate
/// @param local_coefficients  [in]    the local coefficient vector
KOKKOS_INLINE_FUNCTION
void atomically_add_local_wedge_scalar_coefficients(
    const grid::Grid4DDataScalar< double >& global_coefficients,
    const int                               local_subdomain_id,
    const int                               x_cell,
    const int                               y_cell,
    const int                               r_cell,
    const dense::Vec< double, 6 > ( &local_coefficients )[2] )
{
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell ), local_coefficients[0]( 0 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell ),
        local_coefficients[0]( 1 ) + local_coefficients[1]( 2 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell ),
        local_coefficients[0]( 2 ) + local_coefficients[1]( 1 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), local_coefficients[0]( 3 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ),
        local_coefficients[0]( 4 ) + local_coefficients[1]( 5 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ),
        local_coefficients[0]( 5 ) + local_coefficients[1]( 4 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell ), local_coefficients[1]( 0 ) );
    Kokkos::atomic_add(
        &global_coefficients( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 ), local_coefficients[1]( 3 ) );
}

} // namespace terra::fe::wedge