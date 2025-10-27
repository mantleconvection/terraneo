
#pragma once
#include "dense/vec.hpp"
#include "grid/shell/spherical_shell.hpp"

namespace terra::fe::wedge::shell {

/// @brief Computes prolongation weights for the spherical shell.
///
/// @note See overload of this function for details.
///
/// This covers the (simpler) case that the fine node index and the corresponding coarse grid nodes are radially aligned.
template < typename ScalarType >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< ScalarType, 2 > prolongation_linear_weights(
    const dense::Vec< int, 4 >&                 idx_fine,
    const dense::Vec< int, 4 >&                 idx_coarse_bot,
    const grid::Grid3DDataVec< ScalarType, 3 >& subdomain_shell_coords_fine,
    const grid::Grid2DDataScalar< ScalarType >  subdomain_radii_fine )
{
    dense::Vec< ScalarType, 2 > weights{};

    const auto idx_coarse_top = idx_coarse_bot + dense::Vec< int, 4 >{ 0, 0, 0, 1 };

    const auto local_subdomain = idx_fine( 0 );

    const auto idx_coarse_bot_fine = dense::Vec< int, 4 >{
        local_subdomain, 2 * idx_coarse_bot( 1 ), 2 * idx_coarse_bot( 2 ), 2 * idx_coarse_bot( 3 ) };
    const auto idx_coarse_top_fine = dense::Vec< int, 4 >{
        local_subdomain, 2 * idx_coarse_top( 1 ), 2 * idx_coarse_top( 2 ), 2 * idx_coarse_top( 3 ) };

    // First we find the two points we want to interpolate between.

    // Compute bottom point.

    const auto coarse_bot =
        grid::shell::coords( idx_coarse_bot_fine, subdomain_shell_coords_fine, subdomain_radii_fine );

    // Compute top point.

    const auto coarse_top =
        grid::shell::coords( idx_coarse_top_fine, subdomain_shell_coords_fine, subdomain_radii_fine );

    // Now find the linear interpolation coefficient of the fine point with respect to the two points.

    const auto fine = grid::shell::coords( idx_fine, subdomain_shell_coords_fine, subdomain_radii_fine );

    const auto fine_norm       = fine.norm();
    const auto coarse_bot_norm = coarse_bot.norm();
    const auto coarse_top_norm = coarse_top.norm();

    const auto nu = ( fine_norm - coarse_bot_norm ) / ( coarse_top_norm - coarse_bot_norm );

    weights( 0 ) = 1.0 - nu;
    weights( 1 ) = nu;

    return weights;
}

/// @brief Computes prolongation weights for the spherical shell.
///
/// Ensures that affine scalar functions (f(x) = a_1 * x_1 + a_2 * x_2 + a_3 * x_3 + c) are interpolated exactly.
///
/// Returns the non-zero columns of the prolongation matrix, i.e., the weights have to be multiplied with corresponding
/// coarse grid values, summed, and written to the fine grid node. Therefore, this function is best used in a loop
/// over the fine grid points.
///
/// A fine grid node is either located exactly at a coarse grid node, or between a pair of coarse grid nodes.
/// There are three cases to consider overall:
///
/// a) fine grid node is at same position as coarse grid point => weight is 1.0
/// b) fine grid node is between two radially aligned coarse grid points (the projection of the two coarse grid points
///    onto the unit sphere is equal) => call the other overload of this function
/// c) otherwise => CALL THIS FUNCTION
///
/// We can in that last case find four (distinct) coarse grid nodes bot_0, bot_1, top_0, top_1 with top_j and bot_j
/// being aligned on a radial layer for j = 0, 1, such that all these nodes are aligned on a plane that also contains
/// the fine grid node.
///
/// (Technically, that fine grid node is located on one facet of a spherical wedge element. We can therefore ignore the
/// contributions from the other two wedge nodes that are not part of that facet).
///
/// It is required to specify the two bottom indices of those coarse nodes. The top ones are computed within this
/// function.
///
/// @note Due to the refinement algorithm, negative prolongation weights are possible (and required) since in some cases
///       at the boundary (largely the entire outer boundary) during refinement new nodes are created that are not
///       contained in the "affine" coarse grid mesh.
///
/// @param idx_fine         index (local_subdomain_id, x, y, r) of the fine grid node
/// @param idx_coarse_bot_0 coarse grid index (local_subdomain_id, x_coarse, y_coarse, r_coarse) of one of the nodes at
///                         the bottom of the wedge's plane that contains the fine-grid node.
/// @param idx_coarse_bot_1 coarse grid index (local_subdomain_id, x_coarse, y_coarse, r_coarse) of the other node at
///                         the bottom of the wedge's plane that contains the fine-grid node.
/// @param subdomain_shell_coords_fine the coords of the nodes on the unit sphere
/// @param subdomain_radii_fine        the node radii
///
/// @return (weight_bot, weight_top) to scale the four coarse grid nodes:
///         fine = weight_bot * ( coarse(bot_0) + coarse(bot_1) ) + weight_top * ( coarse(top_0) + coarse(top_1) )
template < typename ScalarType >
KOKKOS_INLINE_FUNCTION constexpr dense::Vec< ScalarType, 2 > prolongation_linear_weights(
    const dense::Vec< int, 4 >&                 idx_fine,
    const dense::Vec< int, 4 >&                 idx_coarse_bot_0,
    const dense::Vec< int, 4 >&                 idx_coarse_bot_1,
    const grid::Grid3DDataVec< ScalarType, 3 >& subdomain_shell_coords_fine,
    const grid::Grid2DDataScalar< ScalarType >  subdomain_radii_fine )
{
    dense::Vec< ScalarType, 2 > weights{};

    const auto idx_coarse_top_0 = idx_coarse_bot_0 + dense::Vec< int, 4 >{ 0, 0, 0, 1 };
    const auto idx_coarse_top_1 = idx_coarse_bot_1 + dense::Vec< int, 4 >{ 0, 0, 0, 1 };

    const auto local_subdomain = idx_fine( 0 );

    const auto idx_coarse_bot_0_fine = dense::Vec< int, 4 >{
        local_subdomain, 2 * idx_coarse_bot_0( 1 ), 2 * idx_coarse_bot_0( 2 ), 2 * idx_coarse_bot_0( 3 ) };
    const auto idx_coarse_bot_1_fine = dense::Vec< int, 4 >{
        local_subdomain, 2 * idx_coarse_bot_1( 1 ), 2 * idx_coarse_bot_1( 2 ), 2 * idx_coarse_bot_1( 3 ) };
    const auto idx_coarse_top_0_fine = dense::Vec< int, 4 >{
        local_subdomain, 2 * idx_coarse_top_0( 1 ), 2 * idx_coarse_top_0( 2 ), 2 * idx_coarse_top_0( 3 ) };
    const auto idx_coarse_top_1_fine = dense::Vec< int, 4 >{
        local_subdomain, 2 * idx_coarse_top_1( 1 ), 2 * idx_coarse_top_1( 2 ), 2 * idx_coarse_top_1( 3 ) };

    // First we find the two points we want to interpolate between.
    // For that we compute the corresponding two points on the chords.
    // This also captures the case that we are exactly located on a coarse point.

    // Compute bottom point on chord.

    const auto coarse_bot_0 =
        grid::shell::coords( idx_coarse_bot_0_fine, subdomain_shell_coords_fine, subdomain_radii_fine );
    const auto coarse_bot_1 =
        grid::shell::coords( idx_coarse_bot_1_fine, subdomain_shell_coords_fine, subdomain_radii_fine );
    const auto coarse_bot_chord = 0.5 * ( coarse_bot_0 + coarse_bot_1 );

    // Compute top point on chord.

    const auto coarse_top_0 =
        grid::shell::coords( idx_coarse_top_0_fine, subdomain_shell_coords_fine, subdomain_radii_fine );
    const auto coarse_top_1 =
        grid::shell::coords( idx_coarse_top_1_fine, subdomain_shell_coords_fine, subdomain_radii_fine );
    const auto coarse_top_chord = 0.5 * ( coarse_top_0 + coarse_top_1 );

    // Now find the linear interpolation coefficient of the fine point with respect to the two chord points.

    const auto fine = grid::shell::coords( idx_fine, subdomain_shell_coords_fine, subdomain_radii_fine );

    const auto fine_norm       = fine.norm();
    const auto coarse_bot_norm = coarse_bot_chord.norm();
    const auto coarse_top_norm = coarse_top_chord.norm();

    const auto nu = ( fine_norm - coarse_bot_norm ) / ( coarse_top_norm - coarse_bot_norm );

    weights( 0 ) = 0.5 * ( 1.0 - nu );
    weights( 1 ) = 0.5 * nu;

    return weights;
}

} // namespace terra::fe::wedge::shell