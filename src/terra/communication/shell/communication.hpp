#pragma once

#include <ranges>
#include <variant>
#include <vector>

#include "dense/vec.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"

using terra::grid::shell::SubdomainNeighborhood;

namespace terra::communication::shell {

constexpr int MPI_TAG_BOUNDARY_DATA = 100;

/// @brief Send and receive buffers for all process-local subdomain boundaries.
///
/// Allocates views for all boundaries of local subdomains. Those are the nodes that overlap with values from
/// neighboring subdomains.
///
/// One buffer per local boundary + neighbor is allocated. So, for instance, for an edge shared with several
/// neighbors, just as many buffers as neighbors are allocated. This facilitates the receiving step since all
/// neighbors that a subdomain receives data from can send their data simultaneously.
///
/// Can be reused after communication (send + recv) has been completed to avoid unnecessary reallocation.
template < typename ScalarType, int VecDim = 1 >
class SubdomainNeighborhoodSendRecvBuffer
{
  public:
    /// @brief Constructs a SubdomainNeighborhoodSendRecvBuffer for the passed distributed domain object.
    explicit SubdomainNeighborhoodSendRecvBuffer( const grid::shell::DistributedDomain& domain )
    {
        setup_buffers( domain );
    }

    /// @brief Const reference to the view that is a buffer for an edge of a subdomain.
    ///
    /// @param local_subdomain the SubdomainInfo identifying the local subdomain
    /// @param local_boundary_edge the boundary edge of the local subdomain
    /// @param neighbor_subdomain the SubdomainInfo identifying the neighboring subdomain
    /// @param neighbor_boundary_edge the boundary edge of the neighboring subdomain
    ///
    /// @return A const ref to a Kokkos::View with shape (N, VecDim), where N is the number of grid nodes on the edge
    ///         and VecDim is the number of scalars per node (class template parameter).
    const grid::Grid1DDataVec< ScalarType, VecDim >& buffer_edge(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryEdge          local_boundary_edge,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryEdge          neighbor_boundary_edge ) const
    {
        return buffers_edge_.at( { local_subdomain, local_boundary_edge, neighbor_subdomain, neighbor_boundary_edge } );
    }

    /// @brief Const reference to the view that is a buffer for a face of a subdomain.
    ///
    /// @param local_subdomain the SubdomainInfo identifying the local subdomain
    /// @param local_boundary_face the boundary face of the local subdomain
    /// @param neighbor_subdomain the SubdomainInfo identifying the neighboring subdomain
    /// @param neighbor_boundary_face the boundary face of the neighboring subdomain
    ///
    /// @return A const ref to a Kokkos::View with shape (N, M, VecDim), where N, M are the number of grid nodes on
    ///         each side of the face and VecDim is the number of scalars per node (class template parameter).
    const grid::Grid2DDataVec< ScalarType, VecDim >& buffer_face(
        const grid::shell::SubdomainInfo& local_subdomain,
        const grid::BoundaryFace          local_boundary_face,
        const grid::shell::SubdomainInfo& neighbor_subdomain,
        const grid::BoundaryFace          neighbor_boundary_face ) const
    {
        return buffers_face_.at( { local_subdomain, local_boundary_face, neighbor_subdomain, neighbor_boundary_face } );
    }

  private:
    /// @brief Helper called in the ctor that allocates the buffers.
    void setup_buffers( const grid::shell::DistributedDomain& domain )
    {
        for ( const auto& [subdomain_info, data] : domain.subdomains() )
        {
            const auto& [local_subdomain_idx, neighborhood] = data;

            for ( const auto& [local_boundary_vertex, neighbor] : neighborhood.neighborhood_vertex() )
            {
                for ( const auto& [neighbor_subdomain, neighbor_boundary_vertex, mpi_rank] : neighbor )
                {
                    buffers_vertex_[{
                        subdomain_info, local_boundary_vertex, neighbor_subdomain, neighbor_boundary_vertex }] =
                        grid::Grid0DDataVec< ScalarType, VecDim >( "recv_buffer" );
                }
            }

            for ( const auto& [local_boundary_edge, neighbor] : neighborhood.neighborhood_edge() )
            {
                for ( const auto& [neighbor_subdomain, neighbor_boundary_edge, mpi_rank] : neighbor )
                {
                    const int buffer_size = grid::is_edge_boundary_radial( local_boundary_edge ) ?
                                                domain.domain_info().subdomain_num_nodes_radially() :
                                                domain.domain_info().subdomain_num_nodes_per_side_laterally();

                    buffers_edge_[{ subdomain_info, local_boundary_edge, neighbor_subdomain, neighbor_boundary_edge }] =
                        grid::Grid1DDataVec< ScalarType, VecDim >( "recv_buffer", buffer_size );
                }
            }

            for ( const auto& [local_boundary_face, neighbor] : neighborhood.neighborhood_face() )
            {
                const auto& [neighbor_subdomain, neighbor_boundary_face, mpi_rank] = neighbor;

                const int buffer_size_i = domain.domain_info().subdomain_num_nodes_per_side_laterally();
                const int buffer_size_j = grid::is_face_boundary_normal_to_radial_direction( local_boundary_face ) ?
                                              domain.domain_info().subdomain_num_nodes_per_side_laterally() :
                                              domain.domain_info().subdomain_num_nodes_radially();

                buffers_face_[{ subdomain_info, local_boundary_face, neighbor_subdomain, neighbor_boundary_face }] =
                    grid::Grid2DDataVec< ScalarType, VecDim >( "recv_buffer", buffer_size_i, buffer_size_j );
            }
        }
    }

    /// Key ordering: local subdomain, local boundary, neighbor subdomain, neighbor-local boundary
    std::map<
        std::
            tuple< grid::shell::SubdomainInfo, grid::BoundaryVertex, grid::shell::SubdomainInfo, grid::BoundaryVertex >,
        grid::Grid0DDataVec< ScalarType, VecDim > >
        buffers_vertex_;
    std::map<
        std::tuple< grid::shell::SubdomainInfo, grid::BoundaryEdge, grid::shell::SubdomainInfo, grid::BoundaryEdge >,
        grid::Grid1DDataVec< ScalarType, VecDim > >
        buffers_edge_;
    std::map<
        std::tuple< grid::shell::SubdomainInfo, grid::BoundaryFace, grid::shell::SubdomainInfo, grid::BoundaryFace >,
        grid::Grid2DDataVec< ScalarType, VecDim > >
        buffers_face_;
};

/// @brief Communication reduction modes.
enum class CommunicationReduction
{
    /// Sums up the node values during receive.
    SUM,

    /// Stores the min of all received values during receive.
    MIN,

    /// Stores the max of all received values during receive.
    MAX,
};

/// @brief Packs, sends and recvs local subdomain boundaries using two sets of buffers.
///
/// @note Must be complemented with `unpack_and_reduce_local_subdomain_boundaries()` to complete communication.
///
/// Waits until all recv buffers are filled - but does not unpack.
///
/// Performs "additive" communication. Nodes at the subdomain interfaces overlap and will be reduced using some
/// reduction mode during the receiving phase. This is typically required for matrix-free matrix-vector multiplications
/// in a finite element context: nodes that are shared by elements of two neighboring subdomains receive contributions
/// from both subdomains that need to be added. In this case, the required reduction mode is `CommunicationReduction::SUM`.
///
/// The send buffers are only required until this function returns.
/// The recv buffers must be passed to the corresponding unpacking function `recv_unpack_and_add_local_subdomain_boundaries()`.
///
/// @param domain the DistributedDomain that this works on
/// @param data the data (Kokkos::View) to be communicated
/// @param boundary_send_buffers SubdomainNeighborhoodSendRecvBuffer instance that serves for sending data - can be
///                              reused after this function returns
/// @param boundary_recv_buffers SubdomainNeighborhoodSendRecvBuffer instance that serves for receiving data - must be
///                              passed to `unpack_and_reduce_local_subdomain_boundaries()`
template < typename GridDataType >
void pack_send_and_recv_local_subdomain_boundaries(
    const grid::shell::DistributedDomain& domain,
    const GridDataType&                   data,
    SubdomainNeighborhoodSendRecvBuffer< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() >&
        boundary_send_buffers,
    SubdomainNeighborhoodSendRecvBuffer< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() >&
        boundary_recv_buffers )
{
    // Since it is not clear whether a static last dimension of 1 impacts performance, we want to support both
    // scalar and vector-valued grid data views. To simplify matters, we always use the vector-valued versions for the
    // buffers.

    static_assert(
        std::is_same_v< GridDataType, grid::Grid4DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v<
            GridDataType,
            grid::Grid4DDataVec< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() > > );

    constexpr bool is_scalar =
        std::is_same_v< GridDataType, grid::Grid4DDataScalar< typename GridDataType::value_type > >;

    using ScalarType = typename GridDataType::value_type;

    // std::vector< MPI_Request >                              metadata_send_requests;
    // std::vector< std::unique_ptr< std::array< int, 11 > > > metadata_send_buffers;

    std::vector< MPI_Request > data_send_requests;

    ////////////////////////////////////////////
    // Collecting and sorting send-recv pairs //
    ////////////////////////////////////////////

    // First, we collect all the send-recv pairs and sort them.
    // This ensures the same message order per process.
    // We need to post the Isends and Irecvs in that correct order (per process pair).

    struct SendRecvPair
    {
        int                        boundary_type;
        mpi::MPIRank               local_rank;
        grid::shell::SubdomainInfo local_subdomain;
        int                        local_subdomain_boundary;
        int                        local_subdomain_id;
        mpi::MPIRank               neighbor_rank;
        grid::shell::SubdomainInfo neighbor_subdomain;
        int                        neighbor_subdomain_boundary;
    };

    std::vector< SendRecvPair > send_recv_pairs;

    for ( const auto& [local_subdomain_info, idx_and_neighborhood] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = idx_and_neighborhood;
        for ( const auto& _ : neighborhood.neighborhood_vertex() | std::views::keys )
        {
            throw std::logic_error( "Vertex boundary packing not implemented." );
        }

        for ( const auto& [local_edge_boundary, neighbors] : neighborhood.neighborhood_edge() )
        {
            // Multiple neighbor subdomains per edge.
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

                SendRecvPair send_recv_pair{
                    .boundary_type               = 1,
                    .local_rank                  = mpi::rank(),
                    .local_subdomain             = local_subdomain_info,
                    .local_subdomain_boundary    = static_cast< int >( local_edge_boundary ),
                    .local_subdomain_id          = local_subdomain_id,
                    .neighbor_rank               = neighbor_rank,
                    .neighbor_subdomain          = neighbor_subdomain_info,
                    .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ) };
                send_recv_pairs.push_back( send_recv_pair );
            }
        }

        for ( const auto& [local_face_boundary, neighbor] : neighborhood.neighborhood_face() )
        {
            // Single neighbor subdomain per facet.
            const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

            SendRecvPair send_recv_pair{
                .boundary_type               = 2,
                .local_rank                  = mpi::rank(),
                .local_subdomain             = local_subdomain_info,
                .local_subdomain_boundary    = static_cast< int >( local_face_boundary ),
                .local_subdomain_id          = local_subdomain_id,
                .neighbor_rank               = neighbor_rank,
                .neighbor_subdomain          = neighbor_subdomain_info,
                .neighbor_subdomain_boundary = static_cast< int >( neighbor_local_boundary ) };
            send_recv_pairs.push_back( send_recv_pair );
        }
    }

    ////////////////////
    // Posting Irecvs //
    ////////////////////

    // Sort the pairs by sender subdomains.
    std::sort( send_recv_pairs.begin(), send_recv_pairs.end(), []( const SendRecvPair& a, const SendRecvPair& b ) {
        if ( a.neighbor_subdomain != b.neighbor_subdomain )
            return a.neighbor_subdomain < b.neighbor_subdomain;
        if ( a.boundary_type != b.boundary_type )
            return a.boundary_type < b.boundary_type;
        return a.neighbor_subdomain_boundary < b.neighbor_subdomain_boundary;
    } );

    std::vector< MPI_Request > data_recv_requests;

    for ( const auto& send_recv_pair : send_recv_pairs )
    {
        ScalarType* recv_buffer_ptr  = nullptr;
        int         recv_buffer_size = 0;

        if ( send_recv_pair.boundary_type == 1 )
        {
            const auto& recv_buffer = boundary_recv_buffers.buffer_edge(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryEdge >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryEdge >( send_recv_pair.neighbor_subdomain_boundary ) );

            recv_buffer_ptr  = recv_buffer.data();
            recv_buffer_size = recv_buffer.span();
        }
        else if ( send_recv_pair.boundary_type == 2 )
        {
            const auto& recv_buffer = boundary_recv_buffers.buffer_face(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryFace >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryFace >( send_recv_pair.neighbor_subdomain_boundary ) );

            recv_buffer_ptr  = recv_buffer.data();
            recv_buffer_size = recv_buffer.span();
        }

        MPI_Request data_recv_request;
        MPI_Irecv(
            recv_buffer_ptr,
            recv_buffer_size,
            mpi::mpi_datatype< ScalarType >(),
            send_recv_pair.neighbor_rank,
            MPI_TAG_BOUNDARY_DATA,
            MPI_COMM_WORLD,
            &data_recv_request );
        data_recv_requests.push_back( data_recv_request );
    }

    /////////////////////////////////////////////////
    // Packing send data buffers and posting sends //
    /////////////////////////////////////////////////

    // Sort the pairs by sender subdomains.
    std::sort( send_recv_pairs.begin(), send_recv_pairs.end(), []( const SendRecvPair& a, const SendRecvPair& b ) {
        if ( a.local_subdomain != b.local_subdomain )
            return a.local_subdomain < b.local_subdomain;
        if ( a.boundary_type != b.boundary_type )
            return a.boundary_type < b.boundary_type;
        return a.local_subdomain_boundary < b.local_subdomain_boundary;
    } );

    for ( const auto& send_recv_pair : send_recv_pairs )
    {
        // Packing buffer.

        const auto local_subdomain_id = send_recv_pair.local_subdomain_id;

        // Deep-copy into device-side send buffer.

        ScalarType* send_buffer_ptr  = nullptr;
        int         send_buffer_size = 0;

        if ( send_recv_pair.boundary_type == 1 )
        {
            auto& send_buffer = boundary_send_buffers.buffer_edge(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryEdge >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryEdge >( send_recv_pair.neighbor_subdomain_boundary ) );

            send_buffer_ptr  = send_buffer.data();
            send_buffer_size = send_buffer.span();

            const auto local_edge_boundary =
                static_cast< grid::BoundaryEdge >( send_recv_pair.local_subdomain_boundary );

            if ( local_edge_boundary == grid::BoundaryEdge::E_00R )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                if constexpr ( is_scalar )
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_edge_" + to_string( local_edge_boundary ),
                        Kokkos::RangePolicy( 0, send_buffer.extent( 0 ) ),
                        KOKKOS_LAMBDA( const int idx ) {
                            send_buffer( idx, 0 ) = data( local_subdomain_id, 0, 0, idx );
                        } );
                }
                else
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_edge_" + to_string( local_edge_boundary ),
                        Kokkos::MDRangePolicy( { 0, 0 }, { send_buffer.extent( 0 ), send_buffer.extent( 1 ) } ),
                        KOKKOS_LAMBDA( const int idx, const int d ) {
                            send_buffer( idx, d ) = data( local_subdomain_id, 0, 0, idx, d );
                        } );
                }
            }
        }
        else if ( send_recv_pair.boundary_type == 2 )
        {
            const auto& send_buffer = boundary_send_buffers.buffer_face(
                send_recv_pair.local_subdomain,
                static_cast< grid::BoundaryFace >( send_recv_pair.local_subdomain_boundary ),
                send_recv_pair.neighbor_subdomain,
                static_cast< grid::BoundaryFace >( send_recv_pair.neighbor_subdomain_boundary ) );

            send_buffer_ptr  = send_buffer.data();
            send_buffer_size = send_buffer.span();

            const auto local_face_boundary =
                static_cast< grid::BoundaryFace >( send_recv_pair.local_subdomain_boundary );

            if ( local_face_boundary == grid::BoundaryFace::F_X0R )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                if constexpr ( is_scalar )
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy( { 0, 0 }, { send_buffer.extent( 0 ), send_buffer.extent( 1 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                            send_buffer( idx_i, idx_j, 0 ) = data( local_subdomain_id, idx_i, 0, idx_j );
                        } );
                }
                else
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy(
                            { 0, 0, 0 },
                            { send_buffer.extent( 0 ), send_buffer.extent( 1 ), send_buffer.extent( 2 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j, const int d ) {
                            send_buffer( idx_i, idx_j, d ) = data( local_subdomain_id, idx_i, 0, idx_j, d );
                        } );
                }
            }
            else if ( local_face_boundary == grid::BoundaryFace::F_X1R )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                if constexpr ( is_scalar )
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy( { 0, 0 }, { send_buffer.extent( 0 ), send_buffer.extent( 1 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                            send_buffer( idx_i, idx_j, 0 ) =
                                data( local_subdomain_id, idx_i, data.extent( 2 ) - 1, idx_j );
                        } );
                }
                else
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy(
                            { 0, 0, 0 },
                            { send_buffer.extent( 0 ), send_buffer.extent( 1 ), send_buffer.extent( 2 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j, const int d ) {
                            send_buffer( idx_i, idx_j, d ) =
                                data( local_subdomain_id, idx_i, data.extent( 2 ) - 1, idx_j, d );
                        } );
                }
            }
            else if ( local_face_boundary == grid::BoundaryFace::F_0YR )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                if constexpr ( is_scalar )
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy( { 0, 0 }, { send_buffer.extent( 0 ), send_buffer.extent( 1 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                            send_buffer( idx_i, idx_j, 0 ) = data( local_subdomain_id, 0, idx_i, idx_j );
                        } );
                }
                else
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy(
                            { 0, 0, 0 },
                            { send_buffer.extent( 0 ), send_buffer.extent( 1 ), send_buffer.extent( 2 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j, const int d ) {
                            send_buffer( idx_i, idx_j, d ) = data( local_subdomain_id, 0, idx_i, idx_j, d );
                        } );
                }
            }
            else if ( local_face_boundary == grid::BoundaryFace::F_1YR )
            {
                // Copy data into buffers (to ensure contiguous memory layout) ...
                if constexpr ( is_scalar )
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy( { 0, 0 }, { send_buffer.extent( 0 ), send_buffer.extent( 1 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                            send_buffer( idx_i, idx_j, 0 ) =
                                data( local_subdomain_id, data.extent( 1 ) - 1, idx_i, idx_j );
                        } );
                }
                else
                {
                    Kokkos::parallel_for(
                        "fill_boundary_send_buffer_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy(
                            { 0, 0, 0 },
                            { send_buffer.extent( 0 ), send_buffer.extent( 1 ), send_buffer.extent( 2 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j, const int d ) {
                            send_buffer( idx_i, idx_j, d ) =
                                data( local_subdomain_id, data.extent( 1 ) - 1, idx_i, idx_j, d );
                        } );
                }
            }
            else
            {
                Kokkos::abort( "Send not implemented for that face boundary." );
            }
        }

        Kokkos::fence( "deep_copy_into_send_buffer" );

        // Schedule Isend.

        MPI_Request data_send_request;
        MPI_Isend(
            send_buffer_ptr,
            send_buffer_size,
            mpi::mpi_datatype< ScalarType >(),
            send_recv_pair.neighbor_rank,
            MPI_TAG_BOUNDARY_DATA,
            MPI_COMM_WORLD,
            &data_send_request );
        data_send_requests.push_back( data_send_request );
    }

    /////////////////////////////////////
    // Wait for all sends to complete. //
    /////////////////////////////////////

    MPI_Waitall( data_send_requests.size(), data_send_requests.data(), MPI_STATUSES_IGNORE );
    MPI_Waitall( data_recv_requests.size(), data_recv_requests.data(), MPI_STATUSES_IGNORE );
}

namespace detail {

/// @brief Helper function to defer to the respective Kokkos::atomic_xxx() reduction function.
template < typename T >
KOKKOS_INLINE_FUNCTION void reduction_function( T* ptr, const T& val, const CommunicationReduction reduction_type )
{
    if ( reduction_type == CommunicationReduction::SUM )
    {
        Kokkos::atomic_add( ptr, val );
    }
    else if ( reduction_type == CommunicationReduction::MIN )
    {
        Kokkos::atomic_min( ptr, val );
    }
    else if ( reduction_type == CommunicationReduction::MAX )
    {
        Kokkos::atomic_max( ptr, val );
    }
}

} // namespace detail

/// @brief Unpacks and reduces local subdomain boundaries.
///
/// The recv buffers must be the same instances as used during sending in `pack_send_and_recv_local_subdomain_boundaries()`.
///
/// @param domain the DistributedDomain that this works on
/// @param data the data (Kokkos::View) to be communicated
/// @param boundary_recv_buffers SubdomainNeighborhoodSendRecvBuffer instance that serves for receiving data - must be
///                              the same that was previously populated by `pack_send_and_recv_local_subdomain_boundaries()`
/// @param reduction reduction mode
template < typename GridDataType >
void unpack_and_reduce_local_subdomain_boundaries(
    const grid::shell::DistributedDomain& domain,
    const GridDataType&                   data,
    SubdomainNeighborhoodSendRecvBuffer< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() >&
                           boundary_recv_buffers,
    CommunicationReduction reduction = CommunicationReduction::SUM )
{
    // Since it is not clear whether a static last dimension of 1 impacts performance, we want to support both
    // scalar and vector-valued grid data views. To simplify matters, we always use the vector-valued versions for the
    // buffers.

    static_assert(
        std::is_same_v< GridDataType, grid::Grid4DDataScalar< typename GridDataType::value_type > > ||
        std::is_same_v<
            GridDataType,
            grid::Grid4DDataVec< typename GridDataType::value_type, grid::grid_data_vec_dim< GridDataType >() > > );

    constexpr bool is_scalar =
        std::is_same_v< GridDataType, grid::Grid4DDataScalar< typename GridDataType::value_type > >;

    for ( const auto& [local_subdomain_info, idx_and_neighborhood] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = idx_and_neighborhood;
        for ( const auto& _ : neighborhood.neighborhood_vertex() | std::views::keys )
        {
            throw std::logic_error( "Vertex boundary packing not implemented." );
        }

        for ( const auto& [local_edge_boundary, neighbors] : neighborhood.neighborhood_edge() )
        {
            // Multiple neighbor subdomains per edge.
            for ( const auto& neighbor : neighbors )
            {
                const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

                auto recv_buffer = boundary_recv_buffers.buffer_edge(
                    local_subdomain_info, local_edge_boundary, neighbor_subdomain_info, neighbor_local_boundary );

                switch ( local_edge_boundary )
                {
                case grid::BoundaryEdge::E_00R:

                    if constexpr ( is_scalar )
                    {
                        Kokkos::parallel_for(
                            "add_boundary_edge_" + to_string( local_edge_boundary ),
                            Kokkos::RangePolicy( 0, recv_buffer.extent( 0 ) ),
                            KOKKOS_LAMBDA( const int idx ) {
                                detail::reduction_function(
                                    &data( local_subdomain_id, 0, 0, idx ), recv_buffer( idx, 0 ), reduction );
                            } );
                    }
                    else
                    {
                        Kokkos::parallel_for(
                            "add_boundary_edge_" + to_string( local_edge_boundary ),
                            Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                            KOKKOS_LAMBDA( const int idx, const int d ) {
                                detail::reduction_function(
                                    &data( local_subdomain_id, 0, 0, idx, d ), recv_buffer( idx, d ), reduction );
                            } );
                    }

                    break;
                default:
                    throw std::runtime_error( "Recv (unpack) not implemented for the passed boundary edge." );
                }
            }
        }

        for ( const auto& [local_face_boundary, neighbor] : neighborhood.neighborhood_face() )
        {
            // Single neighbor subdomain per facet.
            const auto& [neighbor_subdomain_info, neighbor_local_boundary, neighbor_rank] = neighbor;

            auto recv_buffer = boundary_recv_buffers.buffer_face(
                local_subdomain_info, local_face_boundary, neighbor_subdomain_info, neighbor_local_boundary );

            switch ( local_face_boundary )
            {
            case grid::BoundaryFace::F_0YR:
                if constexpr ( is_scalar )
                {
                    Kokkos::parallel_for(
                        "add_boundary_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                            detail::reduction_function(
                                &data( local_subdomain_id, 0, idx_i, idx_j ),
                                recv_buffer( idx_i, idx_j, 0 ),
                                reduction );
                        } );
                }
                else
                {
                    Kokkos::parallel_for(
                        "add_boundary_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy(
                            { 0, 0, 0 },
                            { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ), recv_buffer.extent( 2 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j, const int d ) {
                            detail::reduction_function(
                                &data( local_subdomain_id, 0, idx_i, idx_j, d ),
                                recv_buffer( idx_i, idx_j, d ),
                                reduction );
                        } );
                }

                break;
            case grid::BoundaryFace::F_X0R:

                if constexpr ( is_scalar )
                {
                    Kokkos::parallel_for(
                        "add_boundary_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                            detail::reduction_function(
                                &data( local_subdomain_id, idx_i, 0, idx_j ),
                                recv_buffer( idx_i, idx_j, 0 ),
                                reduction );
                        } );
                }
                else
                {
                    Kokkos::parallel_for(
                        "add_boundary_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy(
                            { 0, 0, 0 },
                            { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ), recv_buffer.extent( 2 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j, const int d ) {
                            detail::reduction_function(
                                &data( local_subdomain_id, idx_i, 0, idx_j, d ),
                                recv_buffer( idx_i, idx_j, d ),
                                reduction );
                        } );
                }

                break;

            case grid::BoundaryFace::F_1YR:
                if constexpr ( is_scalar )
                {
                    Kokkos::parallel_for(
                        "add_boundary_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                            detail::reduction_function(
                                &data( local_subdomain_id, data.extent( 1 ) - 1, idx_i, idx_j ),
                                recv_buffer( recv_buffer.extent( 0 ) - 1 - idx_i, idx_j, 0 ),
                                reduction );
                        } );
                }
                else
                {
                    Kokkos::parallel_for(
                        "add_boundary_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy(
                            { 0, 0, 0 },
                            { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ), recv_buffer.extent( 2 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j, const int d ) {
                            detail::reduction_function(
                                &data( local_subdomain_id, data.extent( 1 ) - 1, idx_i, idx_j, d ),
                                recv_buffer( recv_buffer.extent( 0 ) - 1 - idx_i, idx_j, d ),
                                reduction );
                        } );
                }

                break;
            case grid::BoundaryFace::F_X1R:
                if constexpr ( is_scalar )
                {
                    Kokkos::parallel_for(
                        "add_boundary_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy( { 0, 0 }, { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j ) {
                            detail::reduction_function(
                                &data( local_subdomain_id, idx_i, data.extent( 2 ) - 1, idx_j ),
                                recv_buffer( recv_buffer.extent( 0 ) - 1 - idx_i, idx_j, 0 ),
                                reduction );
                        } );
                }
                else
                {
                    Kokkos::parallel_for(
                        "add_boundary_face_" + to_string( local_face_boundary ),
                        Kokkos::MDRangePolicy(
                            { 0, 0, 0 },
                            { recv_buffer.extent( 0 ), recv_buffer.extent( 1 ), recv_buffer.extent( 2 ) } ),
                        KOKKOS_LAMBDA( const int idx_i, const int idx_j, const int d ) {
                            detail::reduction_function(
                                &data( local_subdomain_id, idx_i, data.extent( 2 ) - 1, idx_j, d ),
                                recv_buffer( recv_buffer.extent( 0 ) - 1 - idx_i, idx_j, d ),
                                reduction );
                        } );
                }

                break;
            default:
                throw std::runtime_error( "Recv (unpack) not implemented for the passed boundary face." );
            }
        }
    }

    Kokkos::fence();
}

/// @brief Executes packing, sending, receiving, and unpacking operations for the shell.
///
/// @note THIS MAY COME WITH A PERFORMANCE PENALTY.
///       This function (re-)allocates send and receive buffers for each call, which could be inefficient.
///       Use only where performance does not matter (e.g. in tests).
///       Better: reuse the buffers for subsequent send-recv calls through overloads of this function.
///
/// Essentially just calls `pack_send_and_recv_local_subdomain_boundaries()` and `unpack_and_reduce_local_subdomain_boundaries()`.
template < typename ScalarType >
void send_recv(
    const grid::shell::DistributedDomain& domain,
    grid::Grid4DDataScalar< ScalarType >& grid,
    const CommunicationReduction          reduction = CommunicationReduction::SUM )
{
    SubdomainNeighborhoodSendRecvBuffer< ScalarType > send_buffers( domain );
    SubdomainNeighborhoodSendRecvBuffer< ScalarType > recv_buffers( domain );

    shell::pack_send_and_recv_local_subdomain_boundaries( domain, grid, send_buffers, recv_buffers );
    shell::unpack_and_reduce_local_subdomain_boundaries( domain, grid, recv_buffers, reduction );
}

/// @brief Executes packing, sending, receiving, and unpacking operations for the shell.
///
/// Send and receive buffers must be passed. This is the preferred way to execute communication since the buffers
/// can be reused.
///
/// Essentially just calls `pack_send_and_recv_local_subdomain_boundaries()` and `unpack_and_reduce_local_subdomain_boundaries()`.
template < typename ScalarType >
void send_recv(
    const grid::shell::DistributedDomain&              domain,
    grid::Grid4DDataScalar< ScalarType >&              grid,
    SubdomainNeighborhoodSendRecvBuffer< ScalarType >& send_buffers,
    SubdomainNeighborhoodSendRecvBuffer< ScalarType >& recv_buffers,
    const CommunicationReduction                       reduction = CommunicationReduction::SUM )
{
    shell::pack_send_and_recv_local_subdomain_boundaries( domain, grid, send_buffers, recv_buffers );
    shell::unpack_and_reduce_local_subdomain_boundaries( domain, grid, recv_buffers, reduction );
}

} // namespace terra::communication::shell