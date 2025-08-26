#pragma once

#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::shell {

/// @brief Simple struct, holding device-views for radial profiles.
///
/// See `radial_profiles()` and `radial_profiles_to_table()` for radial profiles computation.
template < typename ScalarType >
struct RadialProfiles
{
    explicit RadialProfiles( int radial_shells )
    : radial_min_( "radial_profiles_min", radial_shells )
    , radial_max_( "radial_profiles_max", radial_shells )
    , radial_sum_( "radial_profiles_sum", radial_shells )
    , radial_cnt_( "radial_profiles_cnt", radial_shells )
    {}

    grid::Grid1DDataScalar< ScalarType > radial_min_;
    grid::Grid1DDataScalar< ScalarType > radial_max_;
    grid::Grid1DDataScalar< ScalarType > radial_sum_;
    grid::Grid1DDataScalar< ScalarType > radial_cnt_;
};

/// @brief Compute radial profiles (min, max, sum, count) for a Q1 scalar field.
///
/// @details Computes the radial profiles for a Q1 scalar field by iterating over all radial shells.
/// The profiles include minimum, maximum, sum, and count of nodes in each shell.
/// The radial shells are defined by the radial dimension of the Q1 scalar field.
/// The output is a RadialProfiles struct with device-side arrays of size num_shells which contain:
/// - Minimum value in the shell
/// - Maximum value in the shell
/// - Sum of values in the shell (compute avg with this and count)
/// - Count of nodes in the shell (compute avg with this and sum)
/// Performs reduction per process on the device and also inter-device reduction with MPI_Allreduce.
/// All processes will carry the same result.
///
/// Mask data is used internally to filter out non-owned nodes. So this really only reduces owned nodes.
///
/// @note The returned Views are still on the device.
///       To convert it to a util::Table for output, use the `radial_profiles_to_table()` function.
///       So unless you really want to compute any further data on the device using the profile, you like always want
///       to call the `radial_profiles_to_table()` function as outlined below.
///
/// To nicely format the output, use the `radial_profiles_to_table()` function, e.g., via
/// @code
/// auto radii = domain_info.radii(); // the DomainInfo is also available in the DistributedDomain
/// auto table = radial_profiles_to_table( radial_profiles( some_field_like_temperature ), radii );
/// std::ofstream out( "radial_profiles.csv" );
/// table.print_csv( out );
/// @endcode
/// That will compute output the average values as well.
///
/// @tparam ScalarType Scalar type of the field.
/// @param data Q1 scalar field data.
/// @return RadialProfiles struct containing [min, max, sum, count] for each radial shell (still on the device).
template < typename ScalarType >
RadialProfiles< ScalarType > radial_profiles( const linalg::VectorQ1Scalar< ScalarType >& data )
{
    const int radial_shells = data.grid_data().extent( 3 );

    // For now, we'll do min/max/avg/count.

    RadialProfiles< ScalarType > radial_profiles( radial_shells );

    const auto data_grid = data.grid_data();
    const auto data_mask = data.mask_data();

    Kokkos::parallel_for(
        "radial profiles init", radial_shells, KOKKOS_LAMBDA( int r ) {
            radial_profiles.radial_min_( r ) = Kokkos::Experimental::finite_max_v< ScalarType >;
            radial_profiles.radial_max_( r ) = Kokkos::Experimental::finite_min_v< ScalarType >;
            radial_profiles.radial_sum_( r ) = 0;
            radial_profiles.radial_cnt_( r ) = 0;
        } );

    Kokkos::fence();

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
            Kokkos::atomic_min( &radial_profiles.radial_min_( r ), data_grid( local_subdomain_id, x, y, r ) );
            Kokkos::atomic_max( &radial_profiles.radial_max_( r ), data_grid( local_subdomain_id, x, y, r ) );
            Kokkos::atomic_add( &radial_profiles.radial_sum_( r ), data_grid( local_subdomain_id, x, y, r ) );
            Kokkos::atomic_add( &radial_profiles.radial_cnt_( r ), static_cast< ScalarType >( 1 ) );
        } );

    Kokkos::fence();

    MPI_Allreduce(
        MPI_IN_PLACE,
        radial_profiles.radial_min_.data(),
        radial_profiles.radial_min_.size(),
        mpi::mpi_datatype< ScalarType >(),
        MPI_MIN,
        MPI_COMM_WORLD );

    MPI_Allreduce(
        MPI_IN_PLACE,
        radial_profiles.radial_max_.data(),
        radial_profiles.radial_max_.size(),
        mpi::mpi_datatype< ScalarType >(),
        MPI_MAX,
        MPI_COMM_WORLD );

    MPI_Allreduce(
        MPI_IN_PLACE,
        radial_profiles.radial_sum_.data(),
        radial_profiles.radial_sum_.size(),
        mpi::mpi_datatype< ScalarType >(),
        MPI_SUM,
        MPI_COMM_WORLD );

    MPI_Allreduce(
        MPI_IN_PLACE,
        radial_profiles.radial_cnt_.data(),
        radial_profiles.radial_cnt_.size(),
        mpi::mpi_datatype< ScalarType >(),
        MPI_SUM,
        MPI_COMM_WORLD );

    return radial_profiles;
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
/// - sum: Average value in the shell
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
/// @param radial_profiles RadialProfiles struct containing radial profile statistics.
///                        Compute this using `radial_profiles()` function. Data is expected to be on the device still.
///                        It is copied to host for table creation in this function.
/// @param radii Vector of shell radii. Can for instance be obtained from the DomainInfo.
/// @return Table with columns: tag, shell_idx, radius, min, max, avg, cnt.
template < typename ScalarType >
util::Table radial_profiles_to_table(
    const RadialProfiles< ScalarType >& radial_profiles,
    const std::vector< ScalarType >     radii )
{
    if ( radii.size() != radial_profiles.radial_min_.extent( 0 ) ||
         radii.size() != radial_profiles.radial_max_.extent( 0 ) ||
         radii.size() != radial_profiles.radial_sum_.extent( 0 ) ||
         radii.size() != radial_profiles.radial_cnt_.extent( 0 ) )
    {
        throw std::runtime_error( "Radial profiles and radii do not have the same number of shells." );
    }

    const auto radial_profiles_host_min =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_min_ );
    const auto radial_profiles_host_max =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_max_ );
    const auto radial_profiles_host_sum =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_sum_ );
    const auto radial_profiles_host_cnt =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), radial_profiles.radial_cnt_ );

    util::Table table;
    for ( int r = 0; r < radii.size(); r++ )
    {
        table.add_row(
            { { "tag", "radial_profiles" },
              { "shell_idx", r },
              { "radius", radii[r] },
              { "min", radial_profiles_host_min( r ) },
              { "max", radial_profiles_host_max( r ) },
              { "sum", radial_profiles_host_sum( r ) },
              { "avg", radial_profiles_host_sum( r ) / radial_profiles_host_cnt( r ) },
              { "cnt", radial_profiles_host_cnt( r ) } } );
    }
    return table;
}

} // namespace terra::shell