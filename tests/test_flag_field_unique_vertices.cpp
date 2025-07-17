#include <fstream>
#include <iomanip>

#include "../src/terra/communication/shell/communication.hpp"
#include "kernels/common/grid_operations.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/vtk/vtk.hpp"

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    const auto domain = terra::grid::shell::DistributedDomain::create_uniform_single_subdomain( 4, 4, 0.5, 1.0 );

    const auto u     = terra::grid::shell::allocate_scalar_grid< double >( "u", domain );
    const auto ones  = terra::grid::shell::allocate_scalar_grid< double >( "ones", domain );
    const auto error = terra::grid::shell::allocate_scalar_grid< double >( "error", domain );

    terra::kernels::common::set_constant( ones, 1.0 );

    const auto subdomain_shell_coords = terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain );
    const auto subdomain_radii        = terra::grid::shell::subdomain_shell_radii( domain );

    terra::communication::shell::SubdomainNeighborhoodSendBuffer< double > send_buffers( domain );
    terra::communication::shell::SubdomainNeighborhoodRecvBuffer< double > recv_buffers( domain );

    std::vector< std::array< int, 11 > > expected_recvs_metadata;
    std::vector< MPI_Request >           expected_recvs_requests;

    // Interpolate the unique subdomain ID.
    for ( const auto& [subdomain_info, value] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = value;

        const auto global_subdomain_id = subdomain_info.global_id();

        terra::kernels::common::set_constant(
            terra::grid::Grid3DDataScalar< double >( u, local_subdomain_id, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL ),
            (double) global_subdomain_id );
    }

    // Communicate and reduce with minimum.
    terra::communication::shell::pack_and_send_local_subdomain_boundaries(
        domain, u, send_buffers, expected_recvs_requests, expected_recvs_metadata );

    terra::communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
        domain,
        u,
        recv_buffers,
        expected_recvs_requests,
        expected_recvs_metadata,
        terra::communication::shell::CommuncationReduction::MIN );

    // Set all nodes to 1 if the global_subdomain_id matches - 0 otherwise.
    for ( const auto& [subdomain_info, value] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = value;

        const auto global_subdomain_id = subdomain_info.global_id();

        Kokkos::parallel_for(
            "set_flags",
            Kokkos::MDRangePolicy( { 0, 0, 0 }, { u.extent( 1 ), u.extent( 2 ), u.extent( 3 ) } ),
            KOKKOS_LAMBDA( const int x, const int y, const int r ) {
                if ( u( local_subdomain_id, x, y, r ) == (double) global_subdomain_id )
                {
                    u( local_subdomain_id, x, y, r ) = 1.0;
                }
                else
                {
                    u( local_subdomain_id, x, y, r ) = 0.0;
                }
            } );
    }

    // Check global min/max/sum.
    min_mag = terra::kernels::common::min_abs_entry( u );
    max_mag = terra::kernels::common::max_abs_entry( u );
    sum_mag = terra::kernels::common::sum_of_absolutes( u );

    std::cout << "Before comm" << std::endl;
    std::cout << "min_mag = " << min_mag << std::endl;
    std::cout << "max_mag = " << max_mag << std::endl;
    std::cout << "sum_mag = " << sum_mag << std::endl;

    // Communicate and reduce with sum (nothing should change).
    terra::communication::shell::pack_and_send_local_subdomain_boundaries(
        domain, u, send_buffers, expected_recvs_requests, expected_recvs_metadata );

    terra::communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
        domain,
        u,
        recv_buffers,
        expected_recvs_requests,
        expected_recvs_metadata,
        terra::communication::shell::CommuncationReduction::SUM );

    // Check global min/max/sum again. Sum should now be the same as if we count all vertices (including boundaries).
    min_mag = terra::kernels::common::min_abs_entry( u );
    max_mag = terra::kernels::common::max_abs_entry( u );
    sum_mag = terra::kernels::common::sum_of_absolutes( u );

    std::cout << "After comm" << std::endl;
    std::cout << "min_mag = " << min_mag << std::endl;
    std::cout << "max_mag = " << max_mag << std::endl;
    std::cout << "sum_mag = " << sum_mag << std::endl;

    // Check global min/max/sum again. Sum should now be the same as if we count all vertices (including boundaries).
    min_mag = terra::kernels::common::min_abs_entry( ones );
    max_mag = terra::kernels::common::max_abs_entry( ones );
    sum_mag = terra::kernels::common::sum_of_absolutes( ones );

    std::cout << "Ones" << std::endl;
    std::cout << "min_mag = " << min_mag << std::endl;
    std::cout << "max_mag = " << max_mag << std::endl;
    std::cout << "sum_mag = " << sum_mag << std::endl;

    terra::kernels::common::lincomb( error, 0.0, 1.0, u, -1.0, ones );

    min_mag = terra::kernels::common::min_abs_entry( error );
    max_mag = terra::kernels::common::max_abs_entry( error );
    sum_mag = terra::kernels::common::sum_of_absolutes( error );

    std::cout << "Error" << std::endl;
    std::cout << "min_mag = " << min_mag << std::endl;
    std::cout << "max_mag = " << max_mag << std::endl;
    std::cout << "sum_mag = " << sum_mag << std::endl;

    terra::vtk::VTKOutput vtk( subdomain_shell_coords, subdomain_radii, "test_flag_field_unique_vertices.vtu", true );
    vtk.add_scalar_field( u );
    vtk.write();

    MPI_Finalize();

    return 0;
}