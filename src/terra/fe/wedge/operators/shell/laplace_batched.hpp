
#pragma once

#include "../../quadrature/quadrature.hpp"
#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class LaplaceBatched
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

    using view4d_t    = grid::Grid4DDataScalar< ScalarType >;
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = typename team_policy::member_type;

    int nx_e_, ny_e_, nr_e_; // elements counts
    int total_elements_;     // total elements across all subdomains
    int batch_size_;
    int batches_; // number of batches (ceil(total_elements / batch_size))
    int team_size_;
    int elems_per_sub_; // nx_elems*ny_elems*nz_elems

    int scratch_num_nodes_coeffs_;
    int scratch_num_unit_sphere_;
    int scratch_num_radii_;
    int scratch_total_bytes_;

    const int num_nodes_per_hex_ = 8; // nodes per hex - we'll update both wedges in the same thread

    using ScratchPadViewT = Kokkos::View< ScalarType*, Kokkos::DefaultExecutionSpace::scratch_memory_space >;

  public:
    // constructor
    LaplaceBatched(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        bool                                     treat_boundary,
        bool                                     diagonal,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        // TODO(?): this is handled below - we simply switch between atomic add and atomic store later.
        //          we may then skip this assign call.
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "LaplaceSimple: src/dst mismatch" );
        }

        /*

        Element batching
        ----------------

        This kernel updates one element per thread using scratch memory.
        So on team of size B in Kokkos (warp in CUDA, B = 32 threads) updates a batch of (up to) B elements.

        To optimize memory access, we'll use the scratch memory (think GPU cache).
        We'll load relevant data (src coefficients, coordinates) into the scratch memory such that all threads of a team
        can quickly access that memory.

        Scratch memory layout
        ---------------------

        Possible data (for all elements, for all nodes per element):

        [ src | dst | coords_x | coords_y | coords_z | radii | ... ]

        One of these areas looks like this:

        e.g., src (you can replace "src" with "coords_x" etc.):
        [
            // hex corner 0
            src_corner_0_el_0, src_corner_0_el_1, src_corner_0_el_2, ..., src_corner_0_el_(batch-1),
            // hex corner 1
            src_corner_1_el_0, src_corner_1_el_1, src_corner_1_el_2, ..., src_corner_1_el_(batch-1),

            // ...

            // hex corner 7
            src_corner_7_el_0, src_corner_7_el_1, src_corner_7_el_2, ..., src_corner_7_el_(batch-1),
        ]


        For the coords we need less memory so we can save a couple of rows.
        Instead of storing per corner, we store the x, y, z unit sphere coords only for the four corners,
        and the radii at the top and bottom.:

        [
            // x unit sphere, vertex 0
            coords_x_0_el_0, ..., coords_x_0_el_(batch-1),
            // ...
            coords_x_3_el_0, ..., coords_x_3_el_(batch-1),

            // same for y, and z. Then:
            radii_bot_el_0, ..., radii_bot_el_(batch-1),
            radii_top_el_0, ..., radii_top_el_(batch-1),
        ]



        All of this should (hopefully) ensure that no bank conflict (or not many) occurs (at least in single precision).
        Each column (for warp-sized batches) should be located on the same bank.
        If each thread in a warp reads corner idx j then all threads read from a different bank.

        **/

        // Create the team policy: one team per batch.

        nx_e_           = src_.extent( 1 ) - 1;
        ny_e_           = src_.extent( 2 ) - 1;
        nr_e_           = src_.extent( 3 ) - 1;
        total_elements_ = static_cast< int >( src_.extent( 0 ) ) * nx_e_ * ny_e_ * nr_e_;
        elems_per_sub_  = nx_e_ * ny_e_ * nr_e_;

        // Warp-sized team.
        // TODO: We might want to increase this number but then might need add padding columns?!
        batch_size_ = std::min( 32, total_elements_ );
        batches_    = ( total_elements_ + batch_size_ - 1 ) / batch_size_;
        Kokkos::TeamPolicy<> policy( batches_, Kokkos::AUTO );
        team_size_ = policy.team_size();

        scratch_num_nodes_coeffs_ = batch_size_ * num_nodes_per_hex_;
        scratch_num_unit_sphere_  = batch_size_ * 4; // 4 per element
        scratch_num_radii_        = batch_size_ * 1; // one per element (we'll have two arrays)

        scratch_total_bytes_ = ScratchPadViewT::shmem_size(
            2 * scratch_num_nodes_coeffs_ + 3 * scratch_num_unit_sphere_ + 2 * scratch_num_radii_ );

        // Set the per-team scratch size (level 0 scratch mem).
        policy.set_scratch_size( 0, Kokkos::PerTeam( scratch_total_bytes_ ) );

        // std::cout << "LaplaceBatched: total elements:           " << total_elements_ << std::endl;
        // std::cout << "LaplaceBatched: batch size:               " << batch_size_ << std::endl;
        // std::cout << "LaplaceBatched: scratch total bytes:      " << scratch_total_bytes_ << std::endl;
        // std::cout << "LaplaceBatched: scratch num nodes coeffs: " << scratch_num_nodes_coeffs_ << std::endl;
        // std::cout << "LaplaceBatched: scratch num unit sphere:  " << scratch_num_unit_sphere_ << std::endl;
        // std::cout << "LaplaceBatched: scratch num radii:        " << scratch_num_radii_ << std::endl;
        // std::cout << "LaplaceBatched: batches:                  " << batches_ << std::endl;
        // std::cout << "LaplaceBatched: elems per sub:            " << elems_per_sub_ << std::endl;
        // std::cout << "LaplaceBatched: team size:                " << team_size_ << std::endl;

        // run
        Kokkos::parallel_for( "batched_laplace", policy, *this );
        Kokkos::fence();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_and_send_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::recv_unpack_and_add_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    // decode a global linear element index -> (s, ie, je, ke)
    KOKKOS_INLINE_FUNCTION
    void decode_elem_index( int global_elem_idx, int& s, int& ie, int& je, int& ke ) const
    {
        s       = global_elem_idx / elems_per_sub_;
        int rem = global_elem_idx % elems_per_sub_;
        ie      = rem / ( nx_e_ * ny_e_ );
        rem     = rem % ( nx_e_ * ny_e_ );
        je      = rem / nx_e_;
        ke      = rem % nx_e_;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const member_type& team ) const
    {
        const int batch_id = (int) team.league_rank(); // each league rank == one batch

        // ScratchPadViewT src_local( team.team_scratch( 0 ), scratch_num_nodes_coeffs_ );

        // pointer into per-team scratch memory
        // ScalarType* raw_shmem = static_cast< ScalarType* >( team.team_scratch( 0 ).get_shmem( scratch_total_bytes_ ) );

#if 0
        ScalarType* src_local = reinterpret_cast< ScalarType* >( raw_shmem );
        ScalarType* dst_local = reinterpret_cast< ScalarType* >( raw_shmem + scratch_num_nodes_coeffs_ );

        ScalarType* coords_x_local = reinterpret_cast< ScalarType* >( raw_shmem + 2 * scratch_num_nodes_coeffs_ );
        ScalarType* coords_y_local =
            reinterpret_cast< ScalarType* >( raw_shmem + 2 * scratch_num_nodes_coeffs_ + scratch_num_unit_sphere_ );
        ScalarType* coords_z_local =
            reinterpret_cast< ScalarType* >( raw_shmem + 2 * scratch_num_nodes_coeffs_ + 2 * scratch_num_unit_sphere_ );

        ScalarType* radii_bot_local =
            reinterpret_cast< ScalarType* >( raw_shmem + 2 * scratch_num_nodes_coeffs_ + 3 * scratch_num_unit_sphere_ );
        ScalarType* radii_top_local = reinterpret_cast< ScalarType* >(
            raw_shmem + 2 * scratch_num_nodes_coeffs_ + 3 * scratch_num_unit_sphere_ + scratch_num_radii_ );
#endif

        ScalarType* src_local = static_cast< ScalarType* >(
            team.team_shmem().get_shmem( scratch_num_nodes_coeffs_ * sizeof( ScalarType ) ) );
        ScalarType* dst_local = static_cast< ScalarType* >(
            team.team_shmem().get_shmem( scratch_num_nodes_coeffs_ * sizeof( ScalarType ) ) );

        ScalarType* coords_x_local = static_cast< ScalarType* >(
            team.team_shmem().get_shmem( scratch_num_unit_sphere_ * sizeof( ScalarType ) ) );
        ScalarType* coords_y_local = static_cast< ScalarType* >(
            team.team_shmem().get_shmem( scratch_num_unit_sphere_ * sizeof( ScalarType ) ) );
        ScalarType* coords_z_local = static_cast< ScalarType* >(
            team.team_shmem().get_shmem( scratch_num_unit_sphere_ * sizeof( ScalarType ) ) );

        ScalarType* radii_bot_local =
            static_cast< ScalarType* >( team.team_shmem().get_shmem( scratch_num_radii_ * sizeof( ScalarType ) ) );
        ScalarType* radii_top_local =
            static_cast< ScalarType* >( team.team_shmem().get_shmem( scratch_num_radii_ * sizeof( ScalarType ) ) );

        const auto batch_size     = batch_size_;
        const auto total_elements = total_elements_;

        const auto start           = batch_id * batch_size;
        const auto end             = ( start + batch_size < total_elements_ ? start + batch_size : total_elements_ );
        const auto this_batch_size = end - start;

        // Stage 1: load nodal values into shared memory in parallel
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team, this_batch_size ), KOKKOS_LAMBDA( const int idx ) {
                const int elem_global = batch_id * batch_size + idx;

                if ( elem_global >= total_elements )
                {
                    return;
                }

                // Compute global element index.
                int subdomain_idx, x, y, r;
                decode_elem_index( elem_global, subdomain_idx, x, y, r );

                constexpr int offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
                constexpr int offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
                constexpr int offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

                for ( int node_j = 0; node_j < num_nodes_per_hex_; ++node_j )
                {
                    // Compute local node index.
                    const int local_node_index = node_j * batch_size + idx;

                    src_local[local_node_index] =
                        src_( subdomain_idx, x + offset_x[node_j], y + offset_y[node_j], r + offset_r[node_j] );
                }

                for ( int node_j = 0; node_j < 4; ++node_j )
                {
                    // Compute local node index.
                    const int local_node_index = node_j * batch_size + idx;

                    coords_x_local[local_node_index] =
                        grid_( subdomain_idx, x + offset_x[node_j], y + offset_y[node_j], 0 );
                    coords_y_local[local_node_index] =
                        grid_( subdomain_idx, x + offset_x[node_j], y + offset_y[node_j], 1 );
                    coords_z_local[local_node_index] =
                        grid_( subdomain_idx, x + offset_x[node_j], y + offset_y[node_j], 2 );
                }

                radii_bot_local[idx] = radii_( subdomain_idx, r );
                radii_top_local[idx] = radii_( subdomain_idx, r + 1 );
            } );

        // Ensure all loads complete before compute
        team.team_barrier();

        // Quadrature points.
        constexpr auto num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarType, 3 > quad_points[num_quad_points];
        ScalarType                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        // Stage 2: Each thread (up to team_size) handles one element in the batch
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team, this_batch_size ), KOKKOS_LAMBDA( const int e ) {
                for ( int i = 0; i < num_nodes_per_hex_; ++i )
                {
                    const int local_node_index  = i * batch_size + e;
                    dst_local[local_node_index] = 0.0;
                }

                dense::Vec< ScalarType, 3 > wedge_surface_coords[num_wedges_per_hex_cell][num_nodes_per_wedge_surface];

                wedge_surface_coords[0][0]( 0 ) = coords_x_local[0 * batch_size + e];
                wedge_surface_coords[0][0]( 1 ) = coords_y_local[0 * batch_size + e];
                wedge_surface_coords[0][0]( 2 ) = coords_z_local[0 * batch_size + e];

                wedge_surface_coords[0][1]( 0 ) = coords_x_local[1 * batch_size + e];
                wedge_surface_coords[0][1]( 1 ) = coords_y_local[1 * batch_size + e];
                wedge_surface_coords[0][1]( 2 ) = coords_z_local[1 * batch_size + e];

                wedge_surface_coords[0][2]( 0 ) = coords_x_local[2 * batch_size + e];
                wedge_surface_coords[0][2]( 1 ) = coords_y_local[2 * batch_size + e];
                wedge_surface_coords[0][2]( 2 ) = coords_z_local[2 * batch_size + e];

                wedge_surface_coords[1][0]( 0 ) = coords_x_local[3 * batch_size + e];
                wedge_surface_coords[1][0]( 1 ) = coords_y_local[3 * batch_size + e];
                wedge_surface_coords[1][0]( 2 ) = coords_z_local[3 * batch_size + e];

                wedge_surface_coords[1][1]( 0 ) = coords_x_local[2 * batch_size + e];
                wedge_surface_coords[1][1]( 1 ) = coords_y_local[2 * batch_size + e];
                wedge_surface_coords[1][1]( 2 ) = coords_z_local[2 * batch_size + e];

                wedge_surface_coords[1][2]( 0 ) = coords_x_local[1 * batch_size + e];
                wedge_surface_coords[1][2]( 1 ) = coords_y_local[1 * batch_size + e];
                wedge_surface_coords[1][2]( 2 ) = coords_z_local[1 * batch_size + e];

                const int elem_global = batch_id * batch_size + e;
                int       local_subdomain_id, x_cell, y_cell, r_cell;
                decode_elem_index( elem_global, local_subdomain_id, x_cell, y_cell, r_cell );

                ScalarType r_1 = radii_bot_local[e];
                ScalarType r_2 = radii_top_local[e];

                constexpr int hex_node[2][6] = { { 0, 1, 2, 4, 5, 6 }, { 3, 2, 1, 7, 6, 5 } };

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    for ( int q = 0; q < num_quad_points; q++ )
                    {
                        const auto quad_point  = quad_points[q];
                        const auto quad_weight = quad_weights[q];

                        // 1. Compute Jacobian and inverse at this quadrature point.
                        const auto J                = jac( wedge_surface_coords[wedge], r_1, r_2, quad_points[q] );
                        const auto det              = Kokkos::abs( J.det() );
                        const auto J_inv_transposed = J.inv().transposed();

                        // 2. Compute physical gradients for all nodes at this quadrature point.
                        dense::Vec< ScalarType, 3 > grad_phy[num_nodes_per_wedge];
                        for ( int k = 0; k < num_nodes_per_wedge; k++ )
                        {
                            grad_phy[k] = J_inv_transposed * grad_shape( k, quad_point );
                        }

                        // 3. Compute ∇u at this quadrature point.
                        dense::Vec< ScalarType, 3 > grad_u;
                        grad_u.fill( 0.0 );
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            const int local_node_index = hex_node[wedge][j] * batch_size + e;

                            grad_u = grad_u + src_local[local_node_index] * grad_phy[j];
                        }

                        // 4. Add the test function contributions.
                        for ( int i = 0; i < num_nodes_per_wedge; i++ )
                        {
                            const int local_node_index = hex_node[wedge][i] * batch_size + e;
                            dst_local[local_node_index] += quad_weight * grad_phy[i].dot( grad_u ) * det;
                        }
                    }
                }
            } ); // TeamThreadRange for compute

        // Ensure all stores complete before scatter.
        team.team_barrier();

        // Stage 3: Scatter back from scratch mem.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( team, this_batch_size ), KOKKOS_LAMBDA( const int idx ) {
                const int elem_global = batch_id * batch_size + idx;

                if ( elem_global >= total_elements_ )
                {
                    return;
                }

                for ( int node_j = 0; node_j < num_nodes_per_hex_; ++node_j )
                {
                    constexpr int offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
                    constexpr int offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
                    constexpr int offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

                    // Compute global node index.
                    int subdomain_idx, x, y, r;
                    decode_elem_index( elem_global, subdomain_idx, x, y, r );

                    // Compute local node index.
                    const int local_node_index = node_j * batch_size + idx;

                    Kokkos::atomic_add(
                        &dst_( subdomain_idx, x + offset_x[node_j], y + offset_y[node_j], r + offset_r[node_j] ),
                        dst_local[local_node_index] );
                }
            } );

        team.team_barrier();
    }
};

static_assert( linalg::OperatorLike< LaplaceBatched< float > > );
static_assert( linalg::OperatorLike< LaplaceBatched< double > > );

} // namespace terra::fe::wedge::operators::shell