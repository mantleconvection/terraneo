
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

    void apply_impl(
        const SrcVectorType&                    src,
        DstVectorType&                          dst,
        int                                     level,
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        if ( operator_apply_mode == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0, level );
        }

        src_ = src.grid_data( level );
        dst_ = dst.grid_data( level );

        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );

        Kokkos::fence();

        if ( operator_communication_mode == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            std::vector< std::array< int, 11 > > expected_recvs_metadata;
            std::vector< MPI_Request >           expected_recvs_requests;

            communication::shell::pack_and_send_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, expected_recvs_requests, expected_recvs_metadata );
            communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
                domain_, dst_, recv_buffers_, expected_recvs_requests, expected_recvs_metadata );
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // First all the r-independent stuff.
        // Gather surface points for each wedge.

        dense::Vec< double, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Compute lateral part of Jacobian.

        constexpr auto num_quad_points = quad_felippa_3x2_num_quad_points;

        dense::Vec< double, 3 > quad_points[num_quad_points];
        double                  quad_weights[num_quad_points];

        quad_felippa_3x2_quad_points( quad_points );
        quad_felippa_3x2_quad_weights( quad_weights );

        double det_jac_lat[num_wedges_per_hex_cell][num_quad_points] = {};

        jacobian_lat_determinant( det_jac_lat, wedge_phy_surf, quad_points );

        // Only now we introduce radially dependent terms.
        const double r_1 = radii_( local_subdomain_id, r_cell );
        const double r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // For now, compute the local element matrix. We'll improve that later.
        dense::Mat< double, 6, 6 > A[num_wedges_per_hex_cell] = {};

        const double grad_r = grad_forward_map_rad( r_1, r_2 );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                const double r = forward_map_rad( r_1, r_2, quad_points[q]( 2 ) );

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const double shape_i = shape_lat( i, quad_points[q] ) * shape_rad( i, quad_points[q] );
                        const double shape_j = shape_lat( j, quad_points[q] ) * shape_rad( j, quad_points[q] );

                        A[wedge]( i, j ) +=
                            quad_weights[q] * ( shape_i * shape_j * r * r * grad_r * det_jac_lat[wedge][q] );
                    }
                }
            }
        }

        if ( diagonal_ )
        {
            A[0] = A[0].diagonal();
            A[1] = A[1].diagonal();
        }

        dense::Vec< double, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

        dense::Vec< double, 6 > dst[num_wedges_per_hex_cell];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
    }
};

static_assert( linalg::OperatorLike< Mass< double > > );

} // namespace terra::fe::wedge::operators::shell