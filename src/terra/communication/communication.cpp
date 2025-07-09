
#include "communication.hpp"

namespace terra::communication {

SubdomainNeighborhoodSendBuffer::SubdomainNeighborhoodSendBuffer( const grid::shell::DistributedDomain& domain )
{
    setup_buffers( domain );
}

const grid::Grid1DDataScalar< double >& SubdomainNeighborhoodSendBuffer::buffer_edge(
    const grid::shell::SubdomainInfo& subdomain_info,
    const grid::BoundaryEdge          local_boundary_edge ) const
{
    return send_buffers_edge_.at( { subdomain_info, local_boundary_edge } );
}

const grid::Grid2DDataScalar< double >& SubdomainNeighborhoodSendBuffer::buffer_face(
    const grid::shell::SubdomainInfo& subdomain_info,
    const grid::BoundaryFace          local_boundary_face ) const
{
    return send_buffers_face_.at( { subdomain_info, local_boundary_face } );
}

void SubdomainNeighborhoodSendBuffer::setup_buffers( const grid::shell::DistributedDomain& domain )
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

SubdomainNeighborhoodRecvBuffer::SubdomainNeighborhoodRecvBuffer( const grid::shell::DistributedDomain& domain )
{
    setup_buffers( domain );
}

const grid::Grid1DDataScalar< double >& SubdomainNeighborhoodRecvBuffer::buffer_edge(
    const grid::shell::SubdomainInfo& local_subdomain,
    const grid::BoundaryEdge          local_boundary_edge,
    const grid::shell::SubdomainInfo& sender_subdomain,
    const grid::BoundaryEdge          sender_boundary_edge ) const
{
    return recv_buffers_edge_.at( { local_subdomain, local_boundary_edge, sender_subdomain, sender_boundary_edge } );
}

const grid::Grid2DDataScalar< double >& SubdomainNeighborhoodRecvBuffer::buffer_face(
    const grid::shell::SubdomainInfo& local_subdomain,
    const grid::BoundaryFace          local_boundary_face,
    const grid::shell::SubdomainInfo& sender_subdomain,
    const grid::BoundaryFace          sender_boundary_face ) const
{
    return recv_buffers_face_.at( { local_subdomain, local_boundary_face, sender_subdomain, sender_boundary_face } );
}

void SubdomainNeighborhoodRecvBuffer::setup_buffers( const grid::shell::DistributedDomain& domain )
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

                recv_buffers_edge_[{ subdomain_info, local_boundary_edge, sender_subdomain, sender_boundary_edge }] =
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
            const auto& send_buffer = boundary_send_buffers.buffer_edge( local_subdomain_info, local_edge_boundary );

            if ( local_edge_boundary == grid::BoundaryEdge::E_00R )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                Kokkos::parallel_for(
                    "fill_boundary_send_buffer_edge_" + to_string( local_edge_boundary ),
                    Kokkos::RangePolicy( 0, send_buffer.extent( 0 ) ),
                    KOKKOS_LAMBDA( const int idx ) { send_buffer( idx ) = data( local_subdomain_id, 0, 0, idx ); } );
            }
            else
            {
                Kokkos::abort( "Send not implemented for that edge boundary." );
            }

            Kokkos::fence( "deep_copy_into_send_buffer" );

            // Schedule sends and recvs (in the same order per process)...
            for ( const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] : neighbors )
            {
                // Post metadata recv.
                metadata_recv_requests.emplace_back( MPI_Request() );
                metadata_recv_buffers.emplace_back( std::array< int, 11 >{} );

                metadata_recv_requests.back() =
                    detail::schedule_metadata_recv( metadata_recv_buffers.back(), neighbor_rank );

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

                detail::schedule_metadata_send( metadata, neighbor_rank );

                // Post data send
                detail::schedule_send(
                    boundary_send_buffers.buffer_edge( local_subdomain_info, local_edge_boundary ), neighbor_rank );
            }
        }

        // ... face-boundaries of the local subdomain ...
        for ( const auto& [local_face_boundary, neighbor] : neighborhood.neighborhood_face() )
        {
            const auto& send_buffer = boundary_send_buffers.buffer_face( local_subdomain_info, local_face_boundary );

            if ( local_face_boundary == grid::BoundaryFace::F_X0R )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                Kokkos::parallel_for(
                    "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                    Kokkos::MDRangePolicy( { 0, 0 }, { send_buffer.extent( 0 ), send_buffer.extent( 1 ) } ),
                    KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                        send_buffer( idx_i, idx_j ) = data( local_subdomain_id, idx_i, 0, idx_j );
                    } );
            }
            else if ( local_face_boundary == grid::BoundaryFace::F_X1R )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                Kokkos::parallel_for(
                    "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                    Kokkos::MDRangePolicy( { 0, 0 }, { send_buffer.extent( 0 ), send_buffer.extent( 1 ) } ),
                    KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                        send_buffer( idx_i, idx_j ) = data( local_subdomain_id, idx_i, data.extent( 2 ) - 1, idx_j );
                    } );
            }
            else if ( local_face_boundary == grid::BoundaryFace::F_0YR )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                Kokkos::parallel_for(
                    "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                    Kokkos::MDRangePolicy( { 0, 0 }, { send_buffer.extent( 0 ), send_buffer.extent( 1 ) } ),
                    KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                        send_buffer( idx_i, idx_j ) = data( local_subdomain_id, 0, idx_i, idx_j );
                    } );
            }
            else if ( local_face_boundary == grid::BoundaryFace::F_1YR )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                Kokkos::parallel_for(
                    "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                    Kokkos::MDRangePolicy( { 0, 0 }, { send_buffer.extent( 0 ), send_buffer.extent( 1 ) } ),
                    KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                        send_buffer( idx_i, idx_j ) = data( local_subdomain_id, data.extent( 1 ) - 1, idx_i, idx_j );
                    } );
            }
            else
            {
                Kokkos::abort( "Send not implemented for that face boundary." );
            }

            Kokkos::fence( "deep_copy_into_send_buffer" );

            // Schedule sends and recvs (in the same order per process)...
            const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

            // Post metadata recv.
            metadata_recv_requests.emplace_back( MPI_Request() );
            metadata_recv_buffers.emplace_back( std::array< int, 11 >{} );

            metadata_recv_requests.back() =
                detail::schedule_metadata_recv( metadata_recv_buffers.back(), neighbor_rank );

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

            detail::schedule_metadata_send( metadata, neighbor_rank );

            // Post data send.
            detail::schedule_send( send_buffer, neighbor_rank );
        }
    }
}

void recv_unpack_and_add_local_subdomain_boundaries(
    const grid::shell::DistributedDomain&   domain,
    const grid::Grid4DDataScalar< double >& data,
    SubdomainNeighborhoodRecvBuffer&        boundary_recv_buffers,
    std::vector< MPI_Request >&             metadata_recv_requests,
    std::vector< std::array< int, 11 > >&   metadata_recv_buffers,
    CommuncationReduction                   reduction )
{
    if ( metadata_recv_requests.size() != metadata_recv_buffers.size() )
    {
        Kokkos::abort( "Number of expected messages and requests do not match." );
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

                        switch ( receiver_boundary_edge )
                        {
                        case grid::BoundaryEdge::E_00R:
                            Kokkos::parallel_for(
                                "add_boundary_edge_" + to_string( receiver_boundary_edge ),
                                Kokkos::RangePolicy( 0, recv_buffer.extent( 0 ) ),
                                KOKKOS_LAMBDA( const int idx ) {
                                    if ( reduction == CommuncationReduction::SUM )
                                    {
                                        Kokkos::atomic_add(
                                            &data( local_subdomain_id, 0, 0, idx ), recv_buffer( idx ) );
                                    }
                                    else if ( reduction == CommuncationReduction::MIN )
                                    {
                                        Kokkos::atomic_min(
                                            &data( local_subdomain_id, 0, 0, idx ), recv_buffer( idx ) );
                                    }
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

                        switch ( receiver_boundary_face )
                        {
                        case grid::BoundaryFace::F_0YR:
                            Kokkos::parallel_for(
                                "add_boundary_face_" + to_string( receiver_boundary_face ),
                                Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                                KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                                    if ( reduction == CommuncationReduction::SUM )
                                    {
                                        Kokkos::atomic_add(
                                            &data( local_subdomain_id, 0, idx_i, idx_j ), recv_buffer( idx_i, idx_j ) );
                                    }
                                    else if ( reduction == CommuncationReduction::MIN )
                                    {
                                        Kokkos::atomic_min(
                                            &data( local_subdomain_id, 0, idx_i, idx_j ), recv_buffer( idx_i, idx_j ) );
                                    }
                                } );
                            break;
                        case grid::BoundaryFace::F_X0R:
                            Kokkos::parallel_for(
                                "add_boundary_face_" + to_string( receiver_boundary_face ),
                                Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                                KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                                    if ( reduction == CommuncationReduction::SUM )
                                    {
                                        Kokkos::atomic_add(
                                            &data( local_subdomain_id, idx_i, 0, idx_j ), recv_buffer( idx_i, idx_j ) );
                                    }
                                    else if ( reduction == CommuncationReduction::MIN )
                                    {
                                        Kokkos::atomic_min(
                                            &data( local_subdomain_id, idx_i, 0, idx_j ), recv_buffer( idx_i, idx_j ) );
                                    }
                                } );
                            break;

                        case grid::BoundaryFace::F_1YR:
                            Kokkos::parallel_for(
                                "add_boundary_face_" + to_string( receiver_boundary_face ),
                                Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                                KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                                    if ( reduction == CommuncationReduction::SUM )
                                    {
                                        Kokkos::atomic_add(
                                            &data( local_subdomain_id, data.extent( 1 ) - 1, idx_i, idx_j ),
                                            recv_buffer( recv_buffer.extent( 0 ) - 1 - idx_i, idx_j ) );
                                    }
                                    else if ( reduction == CommuncationReduction::MIN )
                                    {
                                        Kokkos::atomic_min(
                                            &data( local_subdomain_id, data.extent( 1 ) - 1, idx_i, idx_j ),
                                            recv_buffer( recv_buffer.extent( 0 ) - 1 - idx_i, idx_j ) );
                                    }
                                } );
                            break;
                        case grid::BoundaryFace::F_X1R:
                            Kokkos::parallel_for(
                                "add_boundary_face_" + to_string( receiver_boundary_face ),
                                Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                                KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                                    if ( reduction == CommuncationReduction::SUM )
                                    {
                                        Kokkos::atomic_add(
                                            &data( local_subdomain_id, idx_i, data.extent( 2 ) - 1, idx_j ),
                                            recv_buffer( recv_buffer.extent( 0 ) - 1 - idx_i, idx_j ) );
                                    }
                                    else if ( reduction == CommuncationReduction::MIN )
                                    {
                                        Kokkos::atomic_min(
                                            &data( local_subdomain_id, idx_i, data.extent( 2 ) - 1, idx_j ),
                                            recv_buffer( recv_buffer.extent( 0 ) - 1 - idx_i, idx_j ) );
                                    }
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

} // namespace terra::communication