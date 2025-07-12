
#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class Mass
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< double >;
    using DstVectorType = linalg::VectorQ1Scalar< double >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< double, 3 > grid_;
    grid::Grid2DDataScalar< double > radii_;

    bool diagonal_;

    communication::shell::SubdomainNeighborhoodSendBuffer< double > send_buffers_;
    communication::shell::SubdomainNeighborhoodRecvBuffer< double > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    Mass(
        const grid::shell::DistributedDomain&   domain,
        const grid::Grid3DDataVec< double, 3 >& grid,
        const grid::Grid2DDataScalar< double >& radii,
        const bool                              diagonal )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , diagonal_( diagonal )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    void apply_impl( const SrcVectorType& src, DstVectorType& dst, int level )
    {
        assign( dst, 0, level );

        src_ = src.grid_data( level );
        dst_ = dst.grid_data( level );

        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );

        std::vector< std::array< int, 11 > > expected_recvs_metadata;
        std::vector< MPI_Request >           expected_recvs_requests;

        communication::shell::pack_and_send_local_subdomain_boundaries(
            domain_, dst_, send_buffers_, expected_recvs_requests, expected_recvs_metadata );
        communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
            domain_, dst_, recv_buffers_, expected_recvs_requests, expected_recvs_metadata );
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // First all the r-independent stuff.
        // Gather surface points for each wedge.
        constexpr int num_wedges = 2;

        // Extract vertex positions of quad
        // (0, 0), (1, 0), (0, 1), (1, 1).
        dense::Vec< double, 3 > quad_surface_coords[2][2];

        for ( int x = x_cell; x <= x_cell + 1; x++ )
        {
            for ( int y = y_cell; y <= y_cell + 1; y++ )
            {
                for ( int d = 0; d < 3; d++ )
                {
                    quad_surface_coords[x - x_cell][y - y_cell]( d ) = grid_( local_subdomain_id, x, y, d );
                }
            }
        }

        // Sort coords for the two wedge surfaces.
        dense::Vec< double, 3 > wedge_phy_surf[num_wedges][3] = {};

        wedge_phy_surf[0][0] = quad_surface_coords[0][0];
        wedge_phy_surf[0][1] = quad_surface_coords[1][0];
        wedge_phy_surf[0][2] = quad_surface_coords[0][1];

        wedge_phy_surf[1][0] = quad_surface_coords[1][1];
        wedge_phy_surf[1][1] = quad_surface_coords[0][1];
        wedge_phy_surf[1][2] = quad_surface_coords[1][0];

        // Compute lateral part of Jacobian.

        constexpr auto nq = quad_felippa_3x2_nq;
        constexpr auto qp = quad_felippa_3x2_qp;
        constexpr auto qw = quad_felippa_3x2_qw;

        double det_jac_lat[num_wedges][nq] = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int q = 0; q < nq; q++ )
            {
                const auto jac_lat = wedge::jac_lat(
                    wedge_phy_surf[wedge][0],
                    wedge_phy_surf[wedge][1],
                    wedge_phy_surf[wedge][2],
                    qp[q]( 0 ),
                    qp[q]( 1 ) );

                det_jac_lat[wedge][q] = Kokkos::abs( jac_lat.det() );
            }
        }

        constexpr int num_nodes_per_wedge = 6;

        // Let's now gather all the shape functions and gradients we need.
        double shape_lat[num_wedges][num_nodes_per_wedge][nq] = {};
        double shape_rad[num_wedges][num_nodes_per_wedge][nq] = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    shape_lat[wedge][node_idx][q] = wedge::shape_lat( qp[q]( 0 ), qp[q]( 1 ) )( node_idx % 3 );
                    shape_rad[wedge][node_idx][q] = wedge::shape_rad( qp[q]( 2 ) )( node_idx / 3 );
                }
            }
        }

        // Only now we introduce radially dependent terms.
        const double r_1 = radii_( local_subdomain_id, r_cell );
        const double r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // For now, compute the local element matrix. We'll improve that later.
        dense::Mat< double, 6, 6 > A[num_wedges] = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int q = 0; q < nq; q++ )
            {
                const double r      = forward_map_rad( r_1, r_2, qp[q]( 2 ) );
                const double grad_r = grad_forward_map_rad( r_1, r_2 );

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const double shape_i = shape_lat[wedge][i][q] * shape_rad[wedge][i][q];
                        const double shape_j = shape_lat[wedge][j][q] * shape_rad[wedge][j][q];
                        A[wedge]( i, j ) += qw[q] * ( shape_i * shape_j * r * r * grad_r * det_jac_lat[wedge][q] );
                    }
                }
            }
        }

        if ( diagonal_ )
        {
            for ( int wedge = 0; wedge < num_wedges; wedge++ )
            {
                for ( int i = 0; i < 6; i++ )
                {
                    for ( int j = 0; j < 6; j++ )
                    {
                        if ( i != j )
                        {
                            A[wedge]( i, j ) = 0.0;
                        }
                    }
                }
            }
        }

        dense::Vec< double, 6 > src[num_wedges];

        src[0]( 0 ) = src_( local_subdomain_id, x_cell, y_cell, r_cell );
        src[0]( 1 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
        src[0]( 2 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
        src[0]( 3 ) = src_( local_subdomain_id, x_cell, y_cell, r_cell + 1 );
        src[0]( 4 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );
        src[0]( 5 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );

        src[1]( 0 ) = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell );
        src[1]( 1 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell );
        src[1]( 2 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell );
        src[1]( 3 ) = src_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 );
        src[1]( 4 ) = src_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 );
        src[1]( 5 ) = src_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 );

        dense::Vec< double, 6 > dst[num_wedges];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell ), dst[0]( 0 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), dst[0]( 1 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), dst[0]( 2 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell, r_cell + 1 ), dst[0]( 3 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), dst[0]( 4 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), dst[0]( 5 ) );

        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell ), dst[1]( 0 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell ), dst[1]( 1 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell ), dst[1]( 2 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell + 1, r_cell + 1 ), dst[1]( 3 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell, y_cell + 1, r_cell + 1 ), dst[1]( 4 ) );
        Kokkos::atomic_add( &dst_( local_subdomain_id, x_cell + 1, y_cell, r_cell + 1 ), dst[1]( 5 ) );
    }
};

static_assert( linalg::OperatorLike< Mass< double > > );

} // namespace terra::fe::wedge::operators::shell