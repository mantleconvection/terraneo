
#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/kernel_helpers.hpp"
#include "fe/wedge/quadrature.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class Laplace
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< double >;
    using DstVectorType = linalg::VectorQ1Scalar< double >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< double, 3 > grid_;
    grid::Grid2DDataScalar< double > radii_;

    bool treat_boundary_;
    bool diagonal_;

    communication::shell::SubdomainNeighborhoodSendBuffer< double > send_buffers_;
    communication::shell::SubdomainNeighborhoodRecvBuffer< double > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    Laplace(
        const grid::shell::DistributedDomain&   domain,
        const grid::Grid3DDataVec< double, 3 >& grid,
        const grid::Grid2DDataScalar< double >& radii,
        bool                                    treat_boundary,
        bool                                    diagonal )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , treat_boundary_( treat_boundary )
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

        dense::Vec< double, 3 > wedge_phy_surf[num_wedges][3] = {};
        wedge_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Compute lateral part of Jacobian.

        constexpr auto nq = quad_felippa_1x1_nq;
        constexpr auto qp = quad_felippa_1x1_qp;
        constexpr auto qw = quad_felippa_1x1_qw;

        dense::Mat< double, 3, 3 > jac_lat_inv_t[num_wedges][nq] = {};
        double                     det_jac_lat[num_wedges][nq]   = {};

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

                jac_lat_inv_t[wedge][q] = jac_lat.inv().transposed();
            }
        }

        constexpr int num_nodes_per_wedge = 6;

        dense::Vec< double, 3 > g_rad[num_wedges][num_nodes_per_wedge][nq] = {};
        dense::Vec< double, 3 > g_lat[num_wedges][num_nodes_per_wedge][nq] = {};

        for ( int wedge = 0; wedge < num_wedges; wedge++ )
        {
            for ( int node_idx = 0; node_idx < num_nodes_per_wedge; node_idx++ )
            {
                for ( int q = 0; q < nq; q++ )
                {
                    g_rad[wedge][node_idx][q] =
                        jac_lat_inv_t[wedge][q] *
                        dense::Vec< double, 3 >{
                            grad_shape_lat_xi_wedge_node( node_idx ) * shape_rad_wedge_node( node_idx, qp[q] ),
                            grad_shape_lat_eta_wedge_node( node_idx ) * shape_rad_wedge_node( node_idx, qp[q] ),
                            0.0 };

                    g_lat[wedge][node_idx][q] =
                        jac_lat_inv_t[wedge][q] *
                        dense::Vec< double, 3 >{ 0.0, 0.0, shape_lat_wedge_node( node_idx, qp[q] ) };
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
                const double r = fe::wedge::forward_map_rad( r_1, r_2, qp[q]( 2 ) );
                // TODO: we can precompute that per quadrature point to avoid the division.
                const double r_inv = 1.0 / r;

                const double grad_r = fe::wedge::grad_forward_map_rad( r_1, r_2 );
                // TODO: we can precompute that per quadrature point to avoid the division.
                const double grad_r_inv = 1.0 / grad_r;

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const dense::Vec< double, 3 > grad_i =
                            r_inv * g_rad[wedge][i][q] +
                            grad_shape_rad_wedge_node( i ) * grad_r_inv * g_lat[wedge][i][q];
                        const dense::Vec< double, 3 > grad_j =
                            r_inv * g_rad[wedge][j][q] +
                            grad_shape_rad_wedge_node( j ) * grad_r_inv * g_lat[wedge][j][q];

                        A[wedge]( i, j ) += qw[q] * ( grad_i.dot( grad_j ) * r * r * grad_r * det_jac_lat[wedge][q] );
                    }
                }
            }
        }

        if ( treat_boundary_ )
        {
            for ( int wedge = 0; wedge < num_wedges; wedge++ )
            {
                dense::Mat< double, 6, 6 > boundary_mask;
                boundary_mask.fill( 1.0 );
                if ( r_cell == 0 )
                {
                    // Inner boundary (CMB).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i < 3 || j < 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
                {
                    // Outer boundary (surface).
                    for ( int i = 0; i < 6; i++ )
                    {
                        for ( int j = 0; j < 6; j++ )
                        {
                            if ( i != j && ( i >= 3 || j >= 3 ) )
                            {
                                boundary_mask( i, j ) = 0.0;
                            }
                        }
                    }
                }

                A[wedge].hadamard_product( boundary_mask );
            }
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        dense::Vec< double, 6 > src[num_wedges];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

        dense::Vec< double, 6 > dst[num_wedges];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
    }
};

static_assert( linalg::OperatorLike< Laplace< double > > );

} // namespace terra::fe::wedge::operators::shell