#pragma once
#include <iostream>
#include <ranges>
#include <vector>

#include "dense/vec.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kokkos/kokkos_tn_structs.hpp"

using terra::grid::shell::SubdomainNeighborhood;

namespace terra::communication {
constexpr int MPI_TAG_BOUNDARY_METADATA = 100;
constexpr int MPI_TAG_BOUNDARY_DATA     = 101;

class SubdomainNeighborhoodSendBuffer
{
  public:
    explicit SubdomainNeighborhoodSendBuffer( const grid::shell::DistributedDomain& domain )
    {
        setup_buffers( domain );
    }

    const grid::Grid1DDataScalar< double >& buffer_edge(
        const grid::shell::SubdomainInfo& subdomain_info,
        const grid::BoundaryEdge          local_boundary_edge ) const
    {
        return send_buffers_edge_.at( { subdomain_info, local_boundary_edge } );
    }

    const grid::Grid2DDataScalar< double >& buffer_face(
        const grid::shell::SubdomainInfo& subdomain_info,
        const grid::BoundaryFace          local_boundary_face ) const
    {
        return send_buffers_face_.at( { subdomain_info, local_boundary_face } );
    }

  private:
    void setup_buffers( const grid::shell::DistributedDomain& domain )
    {
        for ( const auto& [subdomain_info, data] : domain.subdomains() )
        {
            const auto& [local_subdomain_idx, neighborhood] = data;
            for ( const auto& local_boundary_vertex : neighborhood.neighborhood_vertex() | std::views::keys )
            {
                send_buffers_vertex_[{ subdomain_info, local_boundary_vertex }] =
                    grid::Grid0DDataScalar< double >( "send_buffer" );
            }

            for ( const auto& local_boundary_edge : neighborhood.neighborhood_edge() | std::views::keys )
            {
                const int buffer_size = grid::is_edge_boundary_radial( local_boundary_edge ) ?
                                            domain.domain_info().subdomain_num_nodes_radially() :
                                            domain.domain_info().subdomain_num_nodes_per_side_laterally();

                send_buffers_edge_[{ subdomain_info, local_boundary_edge }] =
                    grid::Grid1DDataScalar< double >( "send_buffer", buffer_size );
            }

            for ( const auto& local_boundary_face : neighborhood.neighborhood_face() | std::views::keys )
            {
                const int buffer_size_i = domain.domain_info().subdomain_num_nodes_per_side_laterally();
                const int buffer_size_j = grid::is_face_boundary_normal_to_radial_direction( local_boundary_face ) ?
                                              domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                              domain.domain_info().subdomain_num_nodes_radially();

                send_buffers_face_[{ subdomain_info, local_boundary_face }] =
                    grid::Grid2DDataScalar< double >( "send_buffer", buffer_size_i, buffer_size_j );
            }
        }
    }

    std::map< std::pair< grid::shell::SubdomainInfo, grid::BoundaryVertex >, grid::Grid0DDataScalar< double > >
        send_buffers_vertex_;
    std::map< std::pair< grid::shell::SubdomainInfo, grid::BoundaryEdge >, grid::Grid1DDataScalar< double > >
        send_buffers_edge_;
    std::map< std::pair< grid::shell::SubdomainInfo, grid::BoundaryFace >, grid::Grid2DDataScalar< double > >
        send_buffers_face_;
};

class SubdomainNeighborhoodRecvBuffer
{
  public:
    explicit SubdomainNeighborhoodRecvBuffer( const grid::shell::DistributedDomain& domain )
    {
        setup_buffers( domain );
    }

    const grid::Grid1DDataScalar< double >& buffer_edge(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryEdge          local_boundary_edge,
        const grid::shell::SubdomainInfo& sender_subdomain,
        const grid::BoundaryEdge          sender_boundary_edge ) const
    {
        return recv_buffers_edge_.at(
            { local_subdomain, local_boundary_edge, sender_subdomain, sender_boundary_edge } );
    }

    const grid::Grid2DDataScalar< double >& buffer_face(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryFace          local_boundary_face,
        const grid::shell::SubdomainInfo& sender_subdomain,
        const grid::BoundaryFace          sender_boundary_face ) const
    {
        return recv_buffers_face_.at(
            { local_subdomain, local_boundary_face, sender_subdomain, sender_boundary_face } );
    }

  private:
    void setup_buffers( const grid::shell::DistributedDomain& domain )
    {
        for ( const auto& [subdomain_info, data] : domain.subdomains() )
        {
            const auto& [local_subdomain_idx, neighborhood] = data;

            for ( const auto& [local_boundary_vertex, neighbor] : neighborhood.neighborhood_vertex() )
            {
                for ( const auto& [sender_subdomain, sender_boundary_vertex, mpi_rank] : neighbor )
                {
                    recv_buffers_vertex_[{
                        subdomain_info, local_boundary_vertex, sender_subdomain, sender_boundary_vertex }] =
                        grid::Grid0DDataScalar< double >( "recv_buffer" );
                }
            }

            for ( const auto& [local_boundary_edge, neighbor] : neighborhood.neighborhood_edge() )
            {
                for ( const auto& [sender_subdomain, sender_boundary_edge, mpi_rank] : neighbor )
                {
                    const int buffer_size = grid::is_edge_boundary_radial( local_boundary_edge ) ?
                                                domain.domain_info().subdomain_num_nodes_radially() :
                                                domain.domain_info().subdomain_num_nodes_per_side_laterally();

                    recv_buffers_edge_[{
                        subdomain_info, local_boundary_edge, sender_subdomain, sender_boundary_edge }] =
                        grid::Grid1DDataScalar< double >( "recv_buffer", buffer_size );
                }
            }

            for ( const auto& [local_boundary_face, neighbor] : neighborhood.neighborhood_face() )
            {
                const auto& [sender_subdomain, sender_boundary_face, mpi_rank] = neighbor;

                const int buffer_size_i = domain.domain_info().subdomain_num_nodes_per_side_laterally();
                const int buffer_size_j = grid::is_face_boundary_normal_to_radial_direction( local_boundary_face ) ?
                                              domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                              domain.domain_info().subdomain_num_nodes_radially();

                recv_buffers_face_[{ subdomain_info, local_boundary_face, sender_subdomain, sender_boundary_face }] =
                    grid::Grid2DDataScalar< double >( "recv_buffer", buffer_size_i, buffer_size_j );
            }
        }
    }

    // TODO: document: local subdomain, local boundary, sender subdomain, sender-local boundary
    std::map<
        std::
            tuple< grid::shell::SubdomainInfo, grid::BoundaryVertex, grid::shell::SubdomainInfo, grid::BoundaryVertex >,
        grid::Grid0DDataScalar< double > >
        recv_buffers_vertex_;
    std::map<
        std::tuple< grid::shell::SubdomainInfo, grid::BoundaryEdge, grid::shell::SubdomainInfo, grid::BoundaryEdge >,
        grid::Grid1DDataScalar< double > >
        recv_buffers_edge_;
    std::map<
        std::tuple< grid::shell::SubdomainInfo, grid::BoundaryFace, grid::shell::SubdomainInfo, grid::BoundaryFace >,
        grid::Grid2DDataScalar< double > >
        recv_buffers_face_;
};

template < typename KokkosBufferType >
void schedule_send( const KokkosBufferType& kokkos_send_buffer, int receiver_rank )
{
    if ( !kokkos_send_buffer.span_is_contiguous() )
    {
        // Just to make completely sure that the Kokkos data layout is contiguous.
        // Otherwise, we have a problem ...
        throw std::runtime_error( "send_buffer: send_buffer is not contiguous." );
    }

    // Schedule the send.
    MPI_Request request_buffer;
    MPI_Isend(
        kokkos_send_buffer.data(),
        kokkos_send_buffer.span(),
        MPI_DOUBLE,
        receiver_rank,
        MPI_TAG_BOUNDARY_DATA,
        MPI_COMM_WORLD,
        &request_buffer );
}

template < typename KokkosBufferType >
MPI_Request schedule_recv( const KokkosBufferType& kokkos_recv_buffer, int sender_rank )
{
    if ( !kokkos_recv_buffer.span_is_contiguous() )
    {
        // Just to make completely sure that the Kokkos data layout is contiguous.
        // Otherwise, we have a problem ...
        throw std::runtime_error( "recv_buffer: recv_buffer is not contiguous." );
    }

    // Schedule the recv.
    MPI_Request request_buffer;
    MPI_Irecv(
        kokkos_recv_buffer.data(),
        kokkos_recv_buffer.span(),
        MPI_DOUBLE,
        sender_rank,
        MPI_TAG_BOUNDARY_DATA,
        MPI_COMM_WORLD,
        &request_buffer );
    return request_buffer;
}

namespace detail {
struct RecvRequestBoundaryInfo
{
    grid::shell::SubdomainInfo                                                   sender_subdomain_info;
    std::variant< grid::BoundaryVertex, grid::BoundaryEdge, grid::BoundaryFace > sender_local_boundary;
    grid::shell::SubdomainInfo                                                   receiver_subdomain_info;
    std::variant< grid::BoundaryVertex, grid::BoundaryEdge, grid::BoundaryFace > receiver_local_boundary;
};

} // namespace detail

/// Scheduling sends and recvs in order for every neighbor subdomain with overlapping nodes.
///
/// @param neighborhoods
/// @param data
/// @param boundary_send_buffers
/// @param boundary_recv_buffers
/// @param expected_recvs
/// @param expected_recvs_requests
void pack_and_send_local_subdomain_boundaries(
    const grid::shell::DistributedDomain&   domain,
    const grid::Grid4DDataScalar< double >& data,
    SubdomainNeighborhoodSendBuffer&        boundary_send_buffers,
    std::vector< MPI_Request >&             metadata_recv_requests,
    std::vector< std::array< int, 11 > >&   metadata_recv_buffers )
{
    metadata_recv_requests.clear();
    metadata_recv_buffers.clear();

    // For all local subdomains ...
    for ( const auto& [local_subdomain_info, idx_and_neighborhood] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = idx_and_neighborhood;

        // For all ...
        // ... vertex-boundaries of the local subdomain ...
        for ( const auto& _ : neighborhood.neighborhood_vertex() | std::views::keys )
        {
            throw std::logic_error( "Vertex boundary packing not implemented." );
        }

        // ... edge-boundaries of the local subdomain ...
        for ( const auto& [local_edge_boundary, neighbors] : neighborhood.neighborhood_edge() )
        {
            // Create a slice of the simulation data ...
            const auto slice =
                grid::kokkos_slice_boundary< grid::Grid3DDataScalar< double >, grid::Grid1DDataScalar< double > >(
                    Kokkos::subview( data, local_subdomain_id, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL ),
                    local_edge_boundary );

            // Copy data into buffers (to ensure contiguous memory layout) ...
            const auto& send_buffer = boundary_send_buffers.buffer_edge( local_subdomain_info, local_edge_boundary );
            Kokkos::deep_copy( send_buffer, slice );

            // Schedule sends and recvs (in the same order per process)...
            for ( const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] : neighbors )
            {
                // Post metadata recv.
                metadata_recv_requests.emplace_back( MPI_Request() );
                metadata_recv_buffers.emplace_back( std::array< int, 11 >{} );

                MPI_Irecv(
                    metadata_recv_buffers.back().data(),
                    11,
                    MPI_INT,
                    neighbor_rank,
                    MPI_TAG_BOUNDARY_METADATA,
                    MPI_COMM_WORLD,
                    &metadata_recv_requests.back() );

                // Post metadata send.
                std::array metadata = {
                    1, // boundary type = edge
                    local_subdomain_info.diamond_id(),
                    local_subdomain_info.subdomain_x(),
                    local_subdomain_info.subdomain_y(),
                    local_subdomain_info.subdomain_r(),
                    static_cast< int >( local_edge_boundary ),
                    neighbor_subdomain_info.diamond_id(),
                    neighbor_subdomain_info.subdomain_x(),
                    neighbor_subdomain_info.subdomain_y(),
                    neighbor_subdomain_info.subdomain_r(),
                    static_cast< int >( neighbor_local_boundary ),
                };

                MPI_Request request_buffer;
                MPI_Isend(
                    metadata.data(),
                    metadata.size(),
                    MPI_INT,
                    neighbor_rank,
                    MPI_TAG_BOUNDARY_METADATA,
                    MPI_COMM_WORLD,
                    &request_buffer );

                // Post data send
                schedule_send(
                    boundary_send_buffers.buffer_edge( local_subdomain_info, local_edge_boundary ), neighbor_rank );
            }
        }

        // ... face-boundaries of the local subdomain ...
        for ( const auto& [local_face_boundary, neighbor] : neighborhood.neighborhood_face() )
        {
            // Create a slice of the simulation data ...
            const auto slice =
                grid::kokkos_slice_boundary< grid::Grid3DDataScalar< double >, grid::Grid2DDataScalar< double > >(
                    Kokkos::subview( data, local_subdomain_id, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL ),
                    local_face_boundary );

            // Copy data into buffers (to ensure contiguous memory layout) ...
            Kokkos::deep_copy( boundary_send_buffers.buffer_face( local_subdomain_info, local_face_boundary ), slice );
            Kokkos::fence( "deep_copy_into_send_buffer" );

            // Schedule sends and recvs (in the same order per process)...
            const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

            // Post metadata recv.
            metadata_recv_requests.emplace_back( MPI_Request() );
            metadata_recv_buffers.emplace_back( std::array< int, 11 >{} );

            MPI_Irecv(
                metadata_recv_buffers.back().data(),
                11,
                MPI_INT,
                neighbor_rank,
                MPI_TAG_BOUNDARY_METADATA,
                MPI_COMM_WORLD,
                &metadata_recv_requests.back() );

            // Post metadata send.
            std::array metadata = {
                2, // boundary type = face
                local_subdomain_info.diamond_id(),
                local_subdomain_info.subdomain_x(),
                local_subdomain_info.subdomain_y(),
                local_subdomain_info.subdomain_r(),
                static_cast< int >( local_face_boundary ),
                neighbor_subdomain_info.diamond_id(),
                neighbor_subdomain_info.subdomain_x(),
                neighbor_subdomain_info.subdomain_y(),
                neighbor_subdomain_info.subdomain_r(),
                static_cast< int >( neighbor_local_boundary ),
            };

            MPI_Request request_buffer;
            MPI_Isend(
                metadata.data(),
                metadata.size(),
                MPI_INT,
                neighbor_rank,
                MPI_TAG_BOUNDARY_METADATA,
                MPI_COMM_WORLD,
                &request_buffer );

            // Post data send.
            schedule_send(
                boundary_send_buffers.buffer_face( local_subdomain_info, local_face_boundary ), neighbor_rank );
        }
    }
}

void recv_unpack_and_add_local_subdomain_boundaries(
    const grid::shell::DistributedDomain&   domain,
    const grid::Grid4DDataScalar< double >& data,
    SubdomainNeighborhoodRecvBuffer&        boundary_recv_buffers,
    std::vector< MPI_Request >&             metadata_recv_requests,
    std::vector< std::array< int, 11 > >&   metadata_recv_buffers )
{
    if ( metadata_recv_requests.size() != metadata_recv_buffers.size() )
    {
        throw std::logic_error( "Number of expected messages and requests do not match." );
    }

    const int num_expected_recvs = metadata_recv_requests.size();

    std::vector packet_processed( num_expected_recvs, false );

    while ( true )
    {
        bool all_done = true;

        for ( int packed_idx = 0; packed_idx < num_expected_recvs; packed_idx++ )
        {
            if ( !packet_processed[packed_idx] )
            {
                int        metadata_packet_has_arrived;
                MPI_Status status;
                MPI_Test( &metadata_recv_requests[packed_idx], &metadata_packet_has_arrived, &status );
                if ( metadata_packet_has_arrived )
                {
                    grid::shell::SubdomainInfo sender_subdomain_info(
                        metadata_recv_buffers[packed_idx][1],
                        metadata_recv_buffers[packed_idx][2],
                        metadata_recv_buffers[packed_idx][3],
                        metadata_recv_buffers[packed_idx][4] );

                    grid::shell::SubdomainInfo receiver_subdomain_info(
                        metadata_recv_buffers[packed_idx][6],
                        metadata_recv_buffers[packed_idx][7],
                        metadata_recv_buffers[packed_idx][8],
                        metadata_recv_buffers[packed_idx][9] );

                    const int boundary_type = metadata_recv_buffers[packed_idx][0];
                    if ( boundary_type == 1 )
                    {
                        const auto sender_boundary_edge =
                            static_cast< grid::BoundaryEdge >( metadata_recv_buffers[packed_idx][5] );
                        const auto receiver_boundary_edge =
                            static_cast< grid::BoundaryEdge >( metadata_recv_buffers[packed_idx][10] );

                        auto recv_buffer = boundary_recv_buffers.buffer_edge(
                            receiver_subdomain_info,
                            receiver_boundary_edge,
                            sender_subdomain_info,
                            sender_boundary_edge );

                        // TODO: we can post Irecvs here instead and handle unpacking later.

                        MPI_Status recv_data_status;
                        MPI_Recv(
                            recv_buffer.data(),
                            recv_buffer.span(),
                            MPI_DOUBLE,
                            status.MPI_SOURCE,
                            MPI_TAG_BOUNDARY_DATA,
                            MPI_COMM_WORLD,
                            &recv_data_status );

                        const auto& [local_subdomain_id, neighborhood] =
                            domain.subdomains().at( receiver_subdomain_info );

                        const auto volume_slice = grid::
                            kokkos_slice_boundary< grid::Grid3DDataScalar< double >, grid::Grid1DDataScalar< double > >(
                                Kokkos::subview( data, local_subdomain_id, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL ),
                                receiver_boundary_edge );

                        switch ( receiver_boundary_edge )
                        {
                        case grid::BoundaryEdge::E_00R:
                            Kokkos::parallel_for(
                                "add_boundary_edge_" + to_string( receiver_boundary_edge ),
                                Kokkos::RangePolicy( 0, recv_buffer.extent( 0 ) ),
                                [=]( const int idx ) {
                                    Kokkos::atomic_add( &volume_slice( idx ), recv_buffer( idx ) );
                                } );
                            break;
                        default:
                            throw std::runtime_error( "Recv (unpack) not implemented for the passed boundary edge." );
                        }
                    }
                    else if ( boundary_type == 2 )
                    {
                        const auto sender_boundary_face =
                            static_cast< grid::BoundaryFace >( metadata_recv_buffers[packed_idx][5] );
                        const auto receiver_boundary_face =
                            static_cast< grid::BoundaryFace >( metadata_recv_buffers[packed_idx][10] );

                        auto recv_buffer = boundary_recv_buffers.buffer_face(
                            receiver_subdomain_info,
                            receiver_boundary_face,
                            sender_subdomain_info,
                            sender_boundary_face );

                        // TODO: we can post Irecvs here instead and handle unpacking later.

                        MPI_Status recv_data_status;
                        MPI_Recv(
                            recv_buffer.data(),
                            recv_buffer.span(),
                            MPI_DOUBLE,
                            status.MPI_SOURCE,
                            MPI_TAG_BOUNDARY_DATA,
                            MPI_COMM_WORLD,
                            &recv_data_status );

                        const auto& [local_subdomain_id, neighborhood] =
                            domain.subdomains().at( receiver_subdomain_info );

                        const auto volume_slice = grid::
                            kokkos_slice_boundary< grid::Grid3DDataScalar< double >, grid::Grid2DDataScalar< double > >(
                                Kokkos::subview( data, local_subdomain_id, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL ),
                                receiver_boundary_face );

                        switch ( receiver_boundary_face )
                        {
                        case grid::BoundaryFace::F_0YR:
                        case grid::BoundaryFace::F_X0R:
                            // receiving from same pole (e.g., diamond 0 from diamond 1)
                            Kokkos::parallel_for(
                                "add_boundary_face_" + to_string( receiver_boundary_face ),
                                Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                                [=]( const int idx_i, const int idx_j ) {
                                    Kokkos::atomic_add( &volume_slice( idx_i, idx_j ), recv_buffer( idx_i, idx_j ) );
                                } );
                            break;

                        case grid::BoundaryFace::F_1YR:
                        case grid::BoundaryFace::F_X1R:
                            // receiving from the other pole (e.g., diamond 0 from diamond 5)
                            Kokkos::parallel_for(
                                "add_boundary_face_" + to_string( receiver_boundary_face ),
                                Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                                [=]( const int idx_i, const int idx_j ) {
                                    Kokkos::atomic_add(
                                        &volume_slice( idx_i, idx_j ),
                                        recv_buffer( recv_buffer.extent( 0 ) - 1 - idx_i, idx_j ) );
                                } );
                            break;

                        default:
                            throw std::runtime_error( "Recv (unpack) not implemented for the passed boundary face." );
                        }
                    }
                    else
                    {
                        throw std::runtime_error(
                            "Recv (unpack) not implemented for the received boundary type in metadata." );
                    }

                    packet_processed[packed_idx] = true;
                }
                else
                {
                    all_done = false;
                }
            }
        }

        if ( all_done )
        {
            break;
        }
    }
}

namespace detail {

class NoMPIAllDiamondsFacetsSend
{
  public:
    explicit NoMPIAllDiamondsFacetsSend(
        const std::vector< grid::Grid3DDataScalar< double > >& diamond_grids,
        const int                                              num_nodes_xy )
    : num_nodes_xy_( num_nodes_xy )
    {
        for ( int i = 0; i < 10; i++ )
        {
            diamond_grids_( i ) = diamond_grids[i];
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()( const int xy, const int r ) const
    {
        // Node equivalences: part one - communication between diamonds at the same poles

        // d_0( 0, :, r ) = d_1( :, 0, r )
        // d_1( 0, :, r ) = d_2( :, 0, r )
        // d_2( 0, :, r ) = d_3( :, 0, r )
        // d_3( 0, :, r ) = d_4( :, 0, r )
        // d_4( 0, :, r ) = d_0( :, 0, r )

        // d_5( 0, :, r ) = d_6( :, 0, r )
        // d_6( 0, :, r ) = d_7( :, 0, r )
        // d_7( 0, :, r ) = d_8( :, 0, r )
        // d_8( 0, :, r ) = d_9( :, 0, r )
        // d_9( 0, :, r ) = d_5( :, 0, r )

        {
            dense::Vec< double, 10 > d_0_i;
            dense::Vec< double, 10 > d_i_0;

            for ( int i = 0; i < 10; i++ )
            {
                d_0_i( i ) = diamond_grids_( i )( 0, xy, r );
                d_i_0( i ) = diamond_grids_( i )( xy, 0, r );
            }

            for ( int i = 0; i < 5; i++ )
            {
                const int lhs_north = i;
                const int rhs_north = ( i + 1 ) % 5;
                diamond_grids_( lhs_north )( 0, xy, r ) += d_i_0( rhs_north );
                diamond_grids_( rhs_north )( xy, 0, r ) += d_0_i( lhs_north );

                const int lhs_south = i + 5;
                const int rhs_south = ( ( ( i + 5 ) + 1 ) % 5 ) + 5;
                diamond_grids_( lhs_south )( 0, xy, r ) += d_i_0( rhs_south );
                diamond_grids_( rhs_south )( xy, 0, r ) += d_0_i( lhs_south );
            }
        }

        // Node equivalences: part two - communication between diamonds at different poles

        // d_0( :, end, r ) = d_5( end, :, r )
        // d_1( :, end, r ) = d_6( end, :, r )
        // d_2( :, end, r ) = d_7( end, :, r )
        // d_3( :, end, r ) = d_8( end, :, r )
        // d_4( :, end, r ) = d_9( end, :, r )

        // d_5( :, end, r ) = d_1( end, :, r )
        // d_6( :, end, r ) = d_2( end, :, r )
        // d_7( :, end, r ) = d_3( end, :, r )
        // d_8( :, end, r ) = d_4( end, :, r )
        // d_9( :, end, r ) = d_0( end, :, r )

        // We need to add both ways.
        // Thus, we must first read into temporaries.

        {
            dense::Vec< double, 10 > d_e_i;
            dense::Vec< double, 10 > d_i_e;

            for ( int i = 0; i < 10; i++ )
            {
                d_e_i( i ) = diamond_grids_( i )( num_nodes_xy_ - 1, xy, r );
                d_i_e( i ) = diamond_grids_( i )( xy, num_nodes_xy_ - 1, r );
            }

            for ( int i = 0; i < 5; i++ )
            {
                {
                    const int lhs = i;
                    const int rhs = i + 5;
                    diamond_grids_( lhs )( num_nodes_xy_ - 1, xy, r ) += d_i_e( lhs );
                    diamond_grids_( rhs )( xy, num_nodes_xy_ - 1, r ) += d_e_i( rhs );
                }

                {
                    const int lhs = i + 5;
                    const int rhs = ( i + 1 ) % 5;
                    diamond_grids_( lhs )( num_nodes_xy_ - 1, xy, r ) += d_i_e( lhs );
                    diamond_grids_( rhs )( xy, num_nodes_xy_ - 1, r ) += d_e_i( rhs );
                }
            }
        }
    }

  private:
    int                                                    num_nodes_xy_;
    terra::kokkos::T10< grid::Grid3DDataScalar< double > > diamond_grids_;
};

class NoMPIAllDiamondsFacetsRecv
{
  public:
    explicit NoMPIAllDiamondsFacetsRecv(
        const std::vector< grid::Grid3DDataScalar< double > >& diamond_grids,
        const int                                              num_nodes_xy )
    : num_nodes_xy_( num_nodes_xy )
    {
        for ( int i = 0; i < 10; i++ )
        {
            diamond_grids_( i ) = diamond_grids[i];
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()( const int xy, const int r ) const
    {
        // Node equivalences: part one - communication between diamonds at the same poles

        // d_0( 0, :, r ) = d_1( :, 0, r )
        // d_1( 0, :, r ) = d_2( :, 0, r )
        // d_2( 0, :, r ) = d_3( :, 0, r )
        // d_3( 0, :, r ) = d_4( :, 0, r )
        // d_4( 0, :, r ) = d_0( :, 0, r )

        // d_5( 0, :, r ) = d_6( :, 0, r )
        // d_6( 0, :, r ) = d_7( :, 0, r )
        // d_7( 0, :, r ) = d_8( :, 0, r )
        // d_8( 0, :, r ) = d_9( :, 0, r )
        // d_9( 0, :, r ) = d_5( :, 0, r )

        {
            dense::Vec< double, 10 > d_0_i;
            dense::Vec< double, 10 > d_i_0;

            for ( int i = 0; i < 10; i++ )
            {
                d_0_i( i ) = diamond_grids_( i )( 0, xy, r );
                d_i_0( i ) = diamond_grids_( i )( xy, 0, r );
            }

            for ( int i = 0; i < 5; i++ )
            {
                const int lhs_north = i;
                const int rhs_north = ( i + 1 ) % 5;
                diamond_grids_( lhs_north )( 0, xy, r ) += d_i_0( rhs_north );
                diamond_grids_( rhs_north )( xy, 0, r ) += d_0_i( lhs_north );

                const int lhs_south = i + 5;
                const int rhs_south = ( ( ( i + 5 ) + 1 ) % 5 ) + 5;
                diamond_grids_( lhs_south )( 0, xy, r ) += d_i_0( rhs_south );
                diamond_grids_( rhs_south )( xy, 0, r ) += d_0_i( lhs_south );
            }
        }

        // Node equivalences: part two - communication between diamonds at different poles

        // d_0( :, end, r ) = d_5( end, :, r )
        // d_1( :, end, r ) = d_6( end, :, r )
        // d_2( :, end, r ) = d_7( end, :, r )
        // d_3( :, end, r ) = d_8( end, :, r )
        // d_4( :, end, r ) = d_9( end, :, r )

        // d_5( :, end, r ) = d_1( end, :, r )
        // d_6( :, end, r ) = d_2( end, :, r )
        // d_7( :, end, r ) = d_3( end, :, r )
        // d_8( :, end, r ) = d_4( end, :, r )
        // d_9( :, end, r ) = d_0( end, :, r )

        // We need to add both ways.
        // Thus, we must first read into temporaries.

        {
            dense::Vec< double, 10 > d_e_i;
            dense::Vec< double, 10 > d_i_e;

            for ( int i = 0; i < 10; i++ )
            {
                d_e_i( i ) = diamond_grids_( i )( num_nodes_xy_ - 1, xy, r );
                d_i_e( i ) = diamond_grids_( i )( xy, num_nodes_xy_ - 1, r );
            }

            for ( int i = 0; i < 5; i++ )
            {
                {
                    const int lhs = i;
                    const int rhs = i + 5;
                    diamond_grids_( lhs )( num_nodes_xy_ - 1, xy, r ) += d_i_e( lhs );
                    diamond_grids_( rhs )( xy, num_nodes_xy_ - 1, r ) += d_e_i( rhs );
                }

                {
                    const int lhs = i + 5;
                    const int rhs = ( i + 1 ) % 5;
                    diamond_grids_( lhs )( num_nodes_xy_ - 1, xy, r ) += d_i_e( lhs );
                    diamond_grids_( rhs )( xy, num_nodes_xy_ - 1, r ) += d_e_i( rhs );
                }
            }
        }
    }

  private:
    int                                                    num_nodes_xy_;
    terra::kokkos::T10< grid::Grid3DDataScalar< double > > diamond_grids_;
};

class NoMPIAllDiamondsPoles
{
  public:
    explicit NoMPIAllDiamondsPoles( const std::vector< grid::Grid3DDataScalar< double > >& diamond_grids )

    {
        for ( int i = 0; i < 10; i++ )
        {
            diamond_grids_( i ) = diamond_grids[i];
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()( const int r ) const
    {
        dense::Vec< double, 10 > d_pole;

        for ( int i = 0; i < 10; i++ )
        {
            d_pole( i ) = diamond_grids_( i )( 0, 0, r );
        }

        // North

        for ( int i = 0; i < 5; i++ )
        {
            for ( int j = 0; j < 5; j++ )
            {
                if ( i != j )
                {
                    diamond_grids_( i )( 0, 0, r ) += d_pole( j );
                }
            }
        }

        // South

        for ( int i = 5; i < 10; i++ )
        {
            for ( int j = 5; j < 10; j++ )
            {
                if ( i != j )
                {
                    diamond_grids_( i )( 0, 0, r ) += d_pole( j );
                }
            }
        }
    }

  private:
    terra::kokkos::T10< grid::Grid3DDataScalar< double > > diamond_grids_;
};

} // namespace detail

/// @brief Communicate the boundaries of all diamonds additively.
///
/// Subdomains inside the diamonds are disallowed here. Also, this only works on a single GPU.
inline void no_mpi_all_diamonds_additive( const std::vector< grid::Grid3DDataScalar< double > >& diamond_grids )
{
#if 0
    const int xy = diamond_grids[0].extent( 0 );
    const int r  = diamond_grids[0].extent( 2 );

    Kokkos::parallel_for(
        "diamond facet comm",
        Kokkos::MDRangePolicy( { 0, 0 }, { xy, r } ),
        detail::NoMPIAllDiamondsFacets( diamond_grids, xy ) );

    Kokkos::parallel_for(
        "diamond pole comm", Kokkos::RangePolicy( 0, r ), detail::NoMPIAllDiamondsPoles( diamond_grids ) );
#endif
}

} // namespace terra::communication