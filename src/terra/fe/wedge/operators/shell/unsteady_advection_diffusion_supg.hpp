
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

template < typename ScalarT, int VelocityVecDim = 3 >
class UnsteadyAdvectionDiffusionSUPG
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using DstVectorType = linalg::VectorQ1Scalar< ScalarT >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 > grid_;
    grid::Grid2DDataScalar< ScalarT > radii_;

    linalg::VectorQ1Vec< ScalarT, VelocityVecDim > velocity_;

    ScalarT diffusivity_;
    ScalarT dt_;

    bool    treat_boundary_;
    bool    diagonal_;
    ScalarT mass_scaling_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT > recv_buffers_;

    grid::Grid4DDataScalar< ScalarType >              src_;
    grid::Grid4DDataScalar< ScalarType >              dst_;
    grid::Grid4DDataVec< ScalarType, VelocityVecDim > vel_grid_;

  public:
    UnsteadyAdvectionDiffusionSUPG(
        const grid::shell::DistributedDomain&                 domain,
        const grid::Grid3DDataVec< ScalarT, 3 >&              grid,
        const grid::Grid2DDataScalar< ScalarT >&              radii,
        const linalg::VectorQ1Vec< ScalarT, VelocityVecDim >& velocity,
        const ScalarT                                         diffusivity,
        const ScalarT                                         dt,
        bool                                                  treat_boundary,
        bool                                                  diagonal,
        ScalarT                                               mass_scaling,
        linalg::OperatorApplyMode                             operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode                     operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , velocity_( velocity )
    , diffusivity_( diffusivity )
    , dt_( dt )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , mass_scaling_( mass_scaling )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {}

    ScalarT&       dt() { return dt_; }
    const ScalarT& dt() const { return dt_; }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_      = src.grid_data();
        dst_      = dst.grid_data();
        vel_grid_ = velocity_.grid_data();

        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // Gather surface points for each wedge.
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        // Quadrature points.
        constexpr auto num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

        dense::Vec< ScalarT, 3 > quad_points[num_quad_points];
        ScalarT                  quad_weights[num_quad_points];

        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );

        // Interpolating velocity into quad points.

        dense::Vec< ScalarT, VelocityVecDim > vel_interp[num_wedges_per_hex_cell][num_quad_points];
        dense::Vec< ScalarT, 6 >              vel_coeffs[VelocityVecDim][num_wedges_per_hex_cell];

        for ( int d = 0; d < VelocityVecDim; d++ )
        {
            extract_local_wedge_vector_coefficients(
                vel_coeffs[d], local_subdomain_id, x_cell, y_cell, r_cell, d, vel_grid_ );
        }

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            for ( int q = 0; q < num_quad_points; q++ )
            {
                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    for ( int d = 0; d < VelocityVecDim; d++ )
                    {
                        vel_interp[wedge][q]( d ) += vel_coeffs[d][wedge]( d ) * shape_i;
                    }
                }
            }
        }

        // Let's compute the streamline diffusivity.

        ScalarT streamline_diffusivity[num_wedges_per_hex_cell];

        // Far from accurate but for now assume h = r.
        const auto h = r_2 - r_1;

        // constants
        const ScalarT eps_vel = ScalarT( 1e-12 );
        const ScalarT Pe_tol  = ScalarT( 1e-8 );

        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
        {
            ScalarT tau_accum = 0.0;
            ScalarT waccum    = 0.0;

            for ( int q = 0; q < num_quad_points; ++q )
            {
                // get velocity at this quad point
                const auto&   uq         = vel_interp[wedge][q];
                const ScalarT vel_norm_q = uq.norm();

                // characteristic length at quad point: if h is cell-based this is fine
                // otherwise compute a local h_q. Here we use h (cell-size).
                ScalarT tau_q = 0.0;

                if ( vel_norm_q > eps_vel )
                {
                    const ScalarT Pe_q = vel_norm_q * h / ( 2.0 * diffusivity_ );

                    if ( Pe_q > Pe_tol )
                    {
                        // coth(Pe) - 1/Pe = 1/tanh(Pe) - 1/Pe
                        const ScalarT coth_minus = ScalarT( 1.0 ) / Kokkos::tanh( Pe_q ) - ScalarT( 1.0 ) / Pe_q;
                        tau_q                    = ( h / ( 2.0 * vel_norm_q ) ) * coth_minus;
                    }
                    else
                    {
                        // diffusion dominated, tau -> limit (expand coth series) ~ h^2/(12*kappa) for small Pe,
                        // but it's fine to set small tau (or compute series). We'll set small tau to 0 here.
                        tau_q = 0.0;
                    }

                    // optional: clamp tau to avoid huge values
                    const ScalarT tau_max = ScalarT( 10.0 ) * h / Kokkos::max( vel_norm_q, eps_vel );
                    if ( tau_q > tau_max )
                    {
                        tau_q = tau_max;
                    }
                }
                else
                {
                    tau_q = 0.0;
                }

                // quadrature weight for this point (if you have weights)
                const ScalarT wq = quad_weights[q]; // if not available, use 1.0
                tau_accum += tau_q * wq;
                waccum += wq;
            }

            // final cell/wedge tau: volume-weighted average
            ScalarT tau_cell              = ( waccum > 0.0 ) ? ( tau_accum / waccum ) : 0.0;
            streamline_diffusivity[wedge] = tau_cell;
        }

        // Compute the local element matrix.
        dense::Mat< ScalarT, 6, 6 > A[num_wedges_per_hex_cell] = {};

        for ( int q = 0; q < num_quad_points; q++ )
        {
            const auto w = quad_weights[q];

            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                const auto J                = jac( wedge_phy_surf[wedge], r_1, r_2, quad_points[q] );
                const auto det              = Kokkos::abs( J.det() );
                const auto J_inv_transposed = J.inv().transposed();

                const auto vel = vel_interp[wedge][q];

                for ( int i = 0; i < num_nodes_per_wedge; i++ )
                {
                    const auto shape_i = shape( i, quad_points[q] );
                    const auto grad_i  = J_inv_transposed * grad_shape( i, quad_points[q] );

                    for ( int j = 0; j < num_nodes_per_wedge; j++ )
                    {
                        const auto shape_j = shape( j, quad_points[q] );
                        const auto grad_j  = J_inv_transposed * grad_shape( j, quad_points[q] );

                        const auto mass      = shape_i * shape_j;
                        const auto diffusion = diffusivity_ * ( grad_i ).dot( grad_j );
                        const auto advection = ( vel.dot( grad_j ) ) * shape_i;
                        const auto streamline_diffusion =
                            streamline_diffusivity[wedge] * ( vel.dot( grad_j ) ) * ( vel.dot( grad_i ) );

                        A[wedge]( i, j ) +=
                            w * ( mass_scaling_ * mass + dt_ * ( diffusion + advection + streamline_diffusion ) ) * det;
                    }
                }
            }
        }

        if ( treat_boundary_ )
        {
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                dense::Mat< ScalarT, 6, 6 > boundary_mask;
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

        dense::Vec< ScalarT, 6 > src[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( src, local_subdomain_id, x_cell, y_cell, r_cell, src_ );

        dense::Vec< ScalarT, 6 > dst[num_wedges_per_hex_cell];

        dst[0] = A[0] * src[0];
        dst[1] = A[1] * src[1];

        atomically_add_local_wedge_scalar_coefficients( dst_, local_subdomain_id, x_cell, y_cell, r_cell, dst );
    }
};

static_assert( linalg::OperatorLike< UnsteadyAdvectionDiffusionSUPG< double > > );

} // namespace terra::fe::wedge::operators::shell