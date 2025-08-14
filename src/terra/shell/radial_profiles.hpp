#pragma once

#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::shell {

/// @brief Compute radial profiles (min, max, avg, count) for a Q1 scalar field.
///
/// @details Computes the radial profiles for a Q1 scalar field by iterating over all radial shells.
/// The profiles include minimum, maximum, average value, and count of nodes in each shell.
/// The radial shells are defined by the radial dimension of the Q1 scalar field.
/// The output is a Grid2DDataScalar with shape [num_shells, 4] where:
/// - Column 0: Minimum value in the shell
/// - Column 1: Maximum value in the shell
/// - Column 2: Average value in the shell
/// - Column 3: Count of nodes in the shell
/// 
/// Mask data is used to filter out non-owned nodes.
///
/// @note The returned Grid2DDataScalar is still on the device.
///       To convert it to a util::Table for output, use the `radial_profiles_to_table()` function.
/// 
/// To nicely format the output, use the `radial_profiles_to_table()` function, e.g., via
/// @code
/// auto radii = domain_info.radii(); // the DomainInfo is also available in the DistributedDomain
/// auto table = radial_profiles_to_table( radial_profiles( some_field_like_temperature ), radii );
/// std::ofstream out( "radial_profiles.csv" );
/// table.print_csv( out );
/// @endcode
///
/// @tparam ScalarType Scalar type of the field.
/// @param data Q1 scalar field data.
/// @return Grid2DDataScalar containing [min, max, avg, count] for each radial shell.
template < typename ScalarType >
grid::Grid2DDataScalar< ScalarType > radial_profiles( const linalg::VectorQ1Scalar< ScalarType >& data )
{
    const int radial_shells = data.grid_data().extent( 3 );

    // For now, we'll do min/max/avg/count.
    // Need to adapt size and init kernel if stuff is added.
    grid::Grid2DDataScalar< ScalarType > reduction_data( "radial_profiles", radial_shells, 4 );

    const auto data_grid = data.grid_data();
    const auto data_mask = data.mask_data();

    Kokkos::parallel_for(
        "radial profiles init", radial_shells, KOKKOS_LAMBDA( int r ) {
            reduction_data( r, 0 ) = Kokkos::Experimental::finite_max_v< ScalarType >;
            reduction_data( r, 1 ) = Kokkos::Experimental::finite_min_v< ScalarType >;
            reduction_data( r, 2 ) = 0;
            reduction_data( r, 3 ) = 0;
        } );

    Kokkos::parallel_for(
        "radial profiles reduction",
        Kokkos::MDRangePolicy(
            { 0, 0, 0, 0 },
            { data_grid.extent( 0 ), data_grid.extent( 1 ), data_grid.extent( 2 ), data_grid.extent( 3 ) } ),
        KOKKOS_LAMBDA( int local_subdomain_id, int x, int y, int r ) {
            if ( util::check_bits( data_mask( local_subdomain_id, x, y, r ), grid::mask_non_owned() ) )
            {
                return;
            }
            Kokkos::atomic_min( &reduction_data( r, 0 ), data_grid( local_subdomain_id, x, y, r ) );
            Kokkos::atomic_max( &reduction_data( r, 1 ), data_grid( local_subdomain_id, x, y, r ) );
            Kokkos::atomic_add( &reduction_data( r, 2 ), data_grid( local_subdomain_id, x, y, r ) );
            Kokkos::atomic_add( &reduction_data( r, 3 ), static_cast< ScalarType >( 1 ) );
        } );

    Kokkos::parallel_for(
        "radial profiles average", radial_shells, KOKKOS_LAMBDA( int r ) {
            reduction_data( r, 2 ) /= reduction_data( r, 3 );
        } );

    return reduction_data;
}

/// @brief Convert radial profile data to a util::Table for analysis or output.
///
/// @details Converts the radial profile data (min, max, avg, count) into a util::Table.
///
/// This table can then be used for further analysis or output to CSV/JSON.
/// The table will have the following columns:
/// - tag: "radial_profiles"
/// - shell_idx: Index of the radial shell
/// - radius: Radius of the shell
/// - min: Minimum value in the shell
/// - max: Maximum value in the shell
/// - avg: Average value in the shell
/// - cnt: Count of nodes in the shell
///
/// To use this function, you can compute the radial profiles using `radial_profiles()` and then convert
/// the result to a table using this function:
///
/// @code
/// auto radii = domain_info.radii(); // the DomainInfo is also available in the DistributedDomain
/// auto table = radial_profiles_to_table( radial_profiles( some_field_like_temperature ), radii );
/// std::ofstream out( "radial_profiles.csv" );
/// table.print_csv( out );
/// @endcode
///
/// @tparam ScalarType Scalar type of the profile data.
/// @param radial_profiles Grid2DDataScalar containing radial profile statistics. Compute this using `radial_profiles()` function. Data is expected to be on the device still. It is copied to host for table creation in this function.
/// @param radii Vector of shell radii. Can for instance be obtained from the DomainInfo.
/// @return Table with columns: tag, shell_idx, radius, min, max, avg, cnt.
template < typename ScalarType >
util::Table radial_profiles_to_table(
    const grid::Grid2DDataScalar< ScalarType >& radial_profiles,
    const std::vector< ScalarType >             radii )
{
    if ( radii.size() != radial_profiles.extent( 0 ) )
    {
        throw std::runtime_error( "Radial profiles and radii do not have the same number of shells." );
    }

    const auto radial_profiles_host = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles );

    util::Table table;
    for ( int r = 0; r < radial_profiles.extent( 0 ); r++ )
    {
        table.add_row(
            { { "tag", "radial_profiles" },
              { "shell_idx", r },
              { "radius", radii[r] },
              { "min", radial_profiles( r, 0 ) },
              { "max", radial_profiles( r, 1 ) },
              { "avg", radial_profiles( r, 2 ) },
              { "cnt", radial_profiles( r, 3 ) } } );
    }
    return table;
}

} // namespace terra::shell