
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
#include "util/timer.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class Gradient
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Vec< ScalarT, 3 >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_fine_;
    grid::shell::DistributedDomain domain_coarse_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_fine_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    bool treat_boundary_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, 3 > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, 3 > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataVec< ScalarType, 3 > dst_;

  public:
    Gradient(
        const grid::shell::DistributedDomain&    domain_fine,
        const grid::shell::DistributedDomain&    domain_coarse,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid_fine,
        const grid::Grid2DDataScalar< ScalarT >& radii_fine,
        bool                                     treat_boundary,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_fine_( domain_fine )
    , domain_coarse_( domain_coarse )
    , grid_fine_( grid_fine )
    , radii_( radii_fine )
    , treat_boundary_( treat_boundary )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain_fine )
    , recv_buffers_( domain_fine )
    {}

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "gradient_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        util::Timer timer_kernel( "gradient_kernel" );
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_fine_ ), *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "gradient_comm" );

            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_fine_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_fine_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Gather surface points for each wedge.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_fine_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // Quadrature points.
        constexpr auto num_quad_points = quadrature::quad_felippa_1x1_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_1x1_quad_points( quad_points );
        quadrature::quad_felippa_1x1_quad_weights( quad_weights );

        const int fine_radial_wedge_index = r_cell % 2;

        // Compute the local element matrix.
        dense::Mat< ScalarT, 18, 6 > A[num_wedges_per_hex_cell] = {};

        for ( int q = 0; q < num_quad_points; q++ )
        {
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const int fine_lateral_wedge_index = fine_lateral_wedge_idx( x_cell, y_cell, wedge );

                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto grad_i = grad_shape( i, quad_points[q] );

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto shape_j =
                            shape_coarse( j, fine_radial_wedge_index, fine_lateral_wedge_index, quad_points[q] );

                        for ( int d = 0; d < 3; d++ )
                        {
                            A[wedge]( d * 6 + i, j ) +=
                                quad_weights[q] * ( -( ( J_inv_transposed * grad_i )(d) *shape_j ) * det );
                        }
                    }
                }
            }
        }

        if ( treat_boundary_ )
        {
            // we are killing rows here
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                dense::Mat< ScalarT, 18, 6 > boundary_mask;
                boundary_mask.fill( 1.0 );
                if ( r_cell == 0 )
                {
                    // Inner boundary (CMB).
                    for ( int d = 0; d < 3; d++ )
                    {
                        for ( int i = 0; i < 6; i++ )
                        {
                            for ( int j = 0; j < 6; j++ )
                            {
                                if ( i < 3 )
                                {
                                    boundary_mask( 6 * d + i, j ) = 0.0;
                                }
                            }
                        }
                    }
                }

                if ( r_cell + 1 == radii_.extent( 1 ) - 1 )
                {
                    // Outer boundary (surface).
                    for ( int d = 0; d < 3; d++ )
                    {
                        for ( int i = 0; i < 6; i++ )
                        {
                            for ( int j = 0; j < 6; j++ )
                            {
                                if ( i >= 3 )
                                {
                                    boundary_mask( 6 * d + i, j ) = 0.0;
                                }
                            }
                        }
                    }
                }

                A[wedge].hadamard_product( boundary_mask );
            }
        }

        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell / 2, y_cell / 2, r_cell / 2, src_ );

        dense::Vec< ScalarT, 18 > dst[num_wedges_per_hex_cell];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        for ( int d = 0; d < 3; d++ )
        {
            dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
            dst_d[0] = dst[0].template slice< 6 >( d * 6 );
            dst_d[1] = dst[1].template slice< 6 >( d * 6 );

            atomically_add_local_wedge_vector_coefficients(
                dst_, local_subdomain_id, x_cell, y_cell, r_cell, d, dst_d );
        }
    }
};

static_assert( linalg::OperatorLike< Gradient< double > > );

} // namespace terra::fe::wedge::operators::shell