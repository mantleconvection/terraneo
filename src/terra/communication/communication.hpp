#pragma once

#include <iostream>
#include <ranges>
#include <variant>
#include <vector>

#include "dense/vec.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"

using terra::grid::shell::SubdomainNeighborhood;

namespace terra::communication {

constexpr int MPI_TAG_BOUNDARY_METADATA = 100;
constexpr int MPI_TAG_BOUNDARY_DATA     = 101;

/// @brief Send buffers for all process-local subdomain boundaries.
///
/// Allocates views only for actually required subdomain boundaries (for those that overlap with neighboring subdomain
/// boundaries). Can technically be reused for any grid data instance (we do not need extra buffers for each FE function
/// so to speak - just reuse the same instance of this class to save memory).
class SubdomainNeighborhoodSendBuffer
{
  public:
    explicit SubdomainNeighborhoodSendBuffer( const grid::shell::DistributedDomain& domain );

    const grid::Grid1DDataScalar< double >& buffer_edge(
        const grid::shell::SubdomainInfo& subdomain_info,
        const grid::BoundaryEdge          local_boundary_edge ) const;

    const grid::Grid2DDataScalar< double >& buffer_face(
        const grid::shell::SubdomainInfo& subdomain_info,
        const grid::BoundaryFace          local_boundary_face ) const;

  private:
    void setup_buffers( const grid::shell::DistributedDomain& domain );

    /// Key ordering: local/sender subdomain, local/sender boundary
    std::map< std::pair< grid::shell::SubdomainInfo, grid::BoundaryVertex >, grid::Grid0DDataScalar< double > >
        send_buffers_vertex_;
    std::map< std::pair< grid::shell::SubdomainInfo, grid::BoundaryEdge >, grid::Grid1DDataScalar< double > >
        send_buffers_edge_;
    std::map< std::pair< grid::shell::SubdomainInfo, grid::BoundaryFace >, grid::Grid2DDataScalar< double > >
        send_buffers_face_;
};

/// @brief Receive buffers for all process-local subdomain boundaries.
///
/// Allocates views only for actually required subdomain boundaries (for those that overlap with neighboring subdomain
/// boundaries). Can technically be reused for any grid data instance (we do not need extra buffers for each FE function
/// so to speak - just reuse the same instance of this class to save memory).
///
/// This is different than the SubdomainNeighborhoodSendBuffer as possibly more than one buffer is needed per process-
/// local subdomain boundary. Since we mostly perform additive communication, we receive the data into the buffers from
/// potentially multiple overlapping neighbor boundaries, and then later add.
class SubdomainNeighborhoodRecvBuffer
{
  public:
    explicit SubdomainNeighborhoodRecvBuffer( const grid::shell::DistributedDomain& domain );

    const grid::Grid1DDataScalar< double >& buffer_edge(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryEdge          local_boundary_edge,
        const grid::shell::SubdomainInfo& sender_subdomain,
        const grid::BoundaryEdge          sender_boundary_edge ) const;

    const grid::Grid2DDataScalar< double >& buffer_face(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryFace          local_boundary_face,
        const grid::shell::SubdomainInfo& sender_subdomain,
        const grid::BoundaryFace          sender_boundary_face ) const;

  private:
    void setup_buffers( const grid::shell::DistributedDomain& domain );

    /// Key ordering: local/receiver subdomain, local/receiver boundary, sender subdomain, sender-local boundary
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

namespace detail {

template < typename ArrayType, size_t ArraySize >
void schedule_metadata_send( const std::array< ArrayType, ArraySize >& send_buffer, int receiver_rank )
{
    static_assert( std::is_same_v< ArrayType, int > );

    // Schedule the send.
    MPI_Request request_buffer;
    MPI_Isend(
        send_buffer.data(),
        send_buffer.size(),
        MPI_INT,
        receiver_rank,
        MPI_TAG_BOUNDARY_METADATA,
        MPI_COMM_WORLD,
        &request_buffer );
}

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

template < typename ArrayType, size_t ArraySize >
MPI_Request schedule_metadata_recv( std::array< ArrayType, ArraySize >& recv_buffer, int sender_rank )
{
    static_assert( std::is_same_v< ArrayType, int > );

    // Schedule the recv.
    MPI_Request request_buffer;
    MPI_Irecv(
        recv_buffer.data(),
        recv_buffer.size(),
        MPI_INT,
        sender_rank,
        MPI_TAG_BOUNDARY_METADATA,
        MPI_COMM_WORLD,
        &request_buffer );
    return request_buffer;
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

struct RecvRequestBoundaryInfo
{
    grid::shell::SubdomainInfo                                                   sender_subdomain_info;
    std::variant< grid::BoundaryVertex, grid::BoundaryEdge, grid::BoundaryFace > sender_local_boundary;
    grid::shell::SubdomainInfo                                                   receiver_subdomain_info;
    std::variant< grid::BoundaryVertex, grid::BoundaryEdge, grid::BoundaryFace > receiver_local_boundary;
};

} // namespace detail

enum class CommuncationReduction
{
    SUM,
    MIN
};

/// @brief Posts metadata receives, packs data into send buffers, sends data to neighboring subdomains.
void pack_and_send_local_subdomain_boundaries(
    const grid::shell::DistributedDomain&   domain,
    const grid::Grid4DDataScalar< double >& data,
    SubdomainNeighborhoodSendBuffer&        boundary_send_buffers,
    std::vector< MPI_Request >&             metadata_recv_requests,
    std::vector< std::array< int, 11 > >&   metadata_recv_buffers );

/// @brief Waits for metadata packets, then waits for corresponding boundary data, unpacks and adds the data.
void recv_unpack_and_add_local_subdomain_boundaries(
    const grid::shell::DistributedDomain&   domain,
    const grid::Grid4DDataScalar< double >& data,
    SubdomainNeighborhoodRecvBuffer&        boundary_recv_buffers,
    std::vector< MPI_Request >&             metadata_recv_requests,
    std::vector< std::array< int, 11 > >&   metadata_recv_buffers,
    CommuncationReduction                   reduction = CommuncationReduction::SUM );

} // namespace terra::communication