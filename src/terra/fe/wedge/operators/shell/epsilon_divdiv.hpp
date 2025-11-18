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

template < typename ScalarT, int VecDim = 3 >
class EpsilonDivDiv
{
  public:
    using SrcVectorType           = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using DstVectorType           = linalg::VectorQ1Vec< ScalarT, VecDim >;
    using ScalarType              = ScalarT;
    static constexpr int LocalMatrixDim = 18;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, LocalMatrixDim, LocalMatrixDim, 2 >;

  private:
    bool apply_stored_lmatrices_ =
        false; // set to make apply_impl() load and use the stored local matrices for the operator application

    Grid4DDataLocalMatrices lmatrices_;

    grid::shell::DistributedDomain domain_;

    grid::Grid3DDataVec< ScalarT, 3 >    grid_;
    grid::Grid2DDataScalar< ScalarT >    radii_;
    grid::Grid4DDataScalar< ScalarType > k_;

    bool treat_boundary_;
    bool diagonal_;

    linalg::OperatorApplyMode         operator_apply_mode_;
    linalg::OperatorCommunicationMode operator_communication_mode_;

    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > send_buffers_;
    communication::shell::SubdomainNeighborhoodSendRecvBuffer< ScalarT, VecDim > recv_buffers_;

    grid::Grid4DDataVec< ScalarType, VecDim > src_;
    grid::Grid4DDataVec< ScalarType, VecDim > dst_;

    // Quadrature points.
    const int num_quad_points = quadrature::quad_felippa_3x2_num_quad_points;

    dense::Vec< ScalarT, 3 > quad_points[quadrature::quad_felippa_3x2_num_quad_points];
    ScalarT                  quad_weights[quadrature::quad_felippa_3x2_num_quad_points];

  public:
    EpsilonDivDiv(
        const grid::shell::DistributedDomain&    domain,
        const grid::Grid3DDataVec< ScalarT, 3 >& grid,
        const grid::Grid2DDataScalar< ScalarT >& radii,
        const grid::Grid4DDataScalar< ScalarT >& k,
        bool                                     treat_boundary,
        bool                                     diagonal,
        linalg::OperatorApplyMode                operator_apply_mode = linalg::OperatorApplyMode::Replace,
        linalg::OperatorCommunicationMode        operator_communication_mode =
            linalg::OperatorCommunicationMode::CommunicateAdditively )
    : domain_( domain )
    , grid_( grid )
    , radii_( radii )
    , k_( k )
    , treat_boundary_( treat_boundary )
    , diagonal_( diagonal )
    , operator_apply_mode_( operator_apply_mode )
    , operator_communication_mode_( operator_communication_mode )
    // TODO: we can reuse the send and recv buffers and pass in from the outside somehow
    , send_buffers_( domain )
    , recv_buffers_( domain )
    {
        quadrature::quad_felippa_3x2_quad_points( quad_points );
        quadrature::quad_felippa_3x2_quad_weights( quad_weights );
    }

    void set_operator_apply_and_communication_modes(
        const linalg::OperatorApplyMode         operator_apply_mode,
        const linalg::OperatorCommunicationMode operator_communication_mode )
    {
        operator_apply_mode_         = operator_apply_mode;
        operator_communication_mode_ = operator_communication_mode;
    }

    /// @brief S/Getter for diagonal member
    void set_diagonal( bool v ) { diagonal_ = v; }

    /// @brief Getter for coefficient
    const grid::Grid4DDataScalar< ScalarType >& k_grid_data() { return k_; }

    /// @brief Getter for domain member
    grid::shell::DistributedDomain& get_domain() { return domain_; }

    /// @brief Getter for radii member
    grid::Grid2DDataScalar< ScalarT >& get_radii() { return radii_; }

    /// @brief Getter for grid member
    grid::Grid3DDataVec< ScalarT, 3 >& get_grid() { return grid_; }

    /// @brief allocates memory for the local matrices
    void allocate_local_matrix_memory()
    {
        if ( lmatrices_.data() == nullptr )
        {
            lmatrices_ = Grid4DDataLocalMatrices(
                "EpsilonDivDiv::lmatrices_",
                domain_.subdomains().size(),
                domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain_.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                domain_.domain_info().subdomain_num_nodes_radially() - 1 );
        }
    }

    /// @brief Set the local matrix stored in the operator
    KOKKOS_INLINE_FUNCTION
    void set_local_matrix(
        const int                                        local_subdomain_id,
        const int                                        x_cell,
        const int                                        y_cell,
        const int                                        r_cell,
        const int                                        wedge,
        const dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim >& mat ) const
    {
        assert( lmatrices_.data() != nullptr );
        for ( int i = 0; i < LocalMatrixDim; ++i )
        {
            for ( int j = 0; j < LocalMatrixDim; ++j )
            {
                lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge )( i, j ) = mat( i, j );
            }
        }
    }

    /// @brief Retrives the local matrix
    /// if there is stored local matrices, the desired local matrix is loaded and returned
    /// if not, the local matrix is assembled on-the-fly
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > get_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        if ( lmatrices_.data() != nullptr )
        {
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > ijslice;
            for ( int i = 0; i < LocalMatrixDim; ++i )
            {
                for ( int j = 0; j < LocalMatrixDim; ++j )
                {
                    ijslice( i, j ) = lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, wedge )( i, j );
                }
            }
            return ijslice;
        }
        else
        {
            return assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, wedge );
        }
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        util::Timer timer_apply( "vector_laplace_apply" );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( src_.extent( 0 ) != dst_.extent( 0 ) || src_.extent( 1 ) != dst_.extent( 1 ) ||
             src_.extent( 2 ) != dst_.extent( 2 ) || src_.extent( 3 ) != dst_.extent( 3 ) )
        {
            throw std::runtime_error( "EpsilonDivDiv: src/dst mismatch" );
        }

        if ( src_.extent( 1 ) != grid_.extent( 1 ) || src_.extent( 2 ) != grid_.extent( 2 ) )
        {
            throw std::runtime_error( "EpsilonDivDiv: src/dst mismatch" );
        }

        util::Timer timer_kernel( "epsilon_divdiv_kernel" );
        Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_ ), *this );
        Kokkos::fence();
        timer_kernel.stop();

        if ( operator_communication_mode_ == linalg::OperatorCommunicationMode::CommunicateAdditively )
        {
            util::Timer timer_comm( "epsilon_divdiv_comm" );

            communication::shell::pack_send_and_recv_local_subdomain_boundaries(
                domain_, dst_, send_buffers_, recv_buffers_ );
            communication::shell::unpack_and_reduce_local_subdomain_boundaries( domain_, dst_, recv_buffers_ );
        }
    }

    /// @brief: For both trial and test space this function sets up a vector:
    /// each vector element holds the symmetric gradient (a 3x3 matrix) of the shape function of the corresponding dof
    /// (if dimi == dimj, these are the same and we are on the diagonal of the vectorial diffusion operator)
    /// Additionally, we compute the scalar factor for the numerical integral comp: determinant of the jacobian,
    /// evaluation of the coefficient k on the element and the quadrature weight of the current quad-point.

    /// The idea of this function is that the two vectors can be:
    /// - accumulated to the result of the local matvec with 2 * num_nodes_per_wedge complexity
    ///   by scaling the dot product of the trial vec and local src dofs with each element of the test vec
    ///   (and adding to the dst dofs, this is the fused local matvec).
    /// - propagated to the local matrix by an outer product of the two vectors
    ///   (without applying it to dofs). This is e.g. required to assemble the finest grid local
    ///   matrix on-the-fly during GCA/Galerkin coarsening.

    ///
    KOKKOS_INLINE_FUNCTION void assemble_trial_test_vecs(
        const int                               wedge,
        const dense::Vec< ScalarType, VecDim >& quad_point,
        const ScalarType                        quad_weight,
        const ScalarT                           r_1,
        const ScalarT                           r_2,
        dense::Vec< ScalarT, 3 > ( *wedge_phy_surf )[3],
        const dense::Vec< ScalarT, 6 >*           k_local_hex,
        const int                                 dimi,
        const int                                 dimj,
        dense::Mat< ScalarType, VecDim, VecDim >* sym_grad_i,
        dense::Mat< ScalarType, VecDim, VecDim >* sym_grad_j,
        ScalarType&                               jdet_keval_quadweight ) const
    {
        dense::Mat< ScalarType, VecDim, VecDim >       J       = jac( wedge_phy_surf[wedge], r_1, r_2, quad_point );
        const auto                                     det     = J.det();
        const auto                                     abs_det = Kokkos::abs( det );
        const dense::Mat< ScalarType, VecDim, VecDim > J_inv_transposed = J.inv_transposed( det );

        // dot of coeff dofs and element-local shape functions to evaluate the coefficent on the current element
        ScalarType k_eval = 0.0;
        for ( int k = 0; k < num_nodes_per_wedge; k++ )
        {
            k_eval += shape( k, quad_point ) * k_local_hex[wedge]( k );
        }

        for ( int k = 0; k < num_nodes_per_wedge; k++ )
        {
            sym_grad_i[k] = symmetric_grad( J_inv_transposed, quad_point, k, dimi );
            sym_grad_j[k] = symmetric_grad( J_inv_transposed, quad_point, k, dimj );
        }
        jdet_keval_quadweight = quad_weight * k_eval * abs_det;
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_cell, const int y_cell, const int r_cell ) const
    {
        // If we have stored lmatrices, use them.
        // It's the user's responsibility to write meaningful matrices via set_lmatrix()
        // We probably never want to assemble lmatrices with DCA and store,
        // so GCA should be the actor storing matrices.
        if ( lmatrices_.data() != nullptr )
        {
            // Compute the local element matrix.
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A[num_wedges_per_hex_cell] = { 0 };
            A[0] =// assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
            lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 0 );
            A[1] = //assemble_local_matrix( local_subdomain_id, x_cell, y_cell, r_cell, 1 );
            lmatrices_( local_subdomain_id, x_cell, y_cell, r_cell, 1 );

            // BCs are applied by GCA ... to be discussed for free-slip

            if ( diagonal_ )
            {
                A[0] = A[0].diagonal();
                A[1] = A[1].diagonal();
            }

            dense::Vec< ScalarT, 18 > src[num_wedges_per_hex_cell];
            for ( int dimj = 0; dimj < 3; dimj++ )
            {
                dense::Vec< ScalarT, 6 > src_d[num_wedges_per_hex_cell];
                extract_local_wedge_vector_coefficients(
                    src_d, local_subdomain_id, x_cell, y_cell, r_cell, dimj, src_ );

                for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                {
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        src[wedge]( dimj * num_nodes_per_wedge + i ) = src_d[wedge]( i );
                    }
                }
            }

            dense::Vec< ScalarT, 18 > dst[num_wedges_per_hex_cell];

            dst[0] = A[0] * src[0];
            dst[1] = A[1] * src[1];

            for ( int dimi = 0; dimi < 3; dimi++ )
            {
                dense::Vec< ScalarT, 6 > dst_d[num_wedges_per_hex_cell];
                dst_d[0] = dst[0].template slice< 6 >( dimi * num_nodes_per_wedge );
                dst_d[1] = dst[1].template slice< 6 >( dimi * num_nodes_per_wedge );

                atomically_add_local_wedge_vector_coefficients(
                    dst_, local_subdomain_id, x_cell, y_cell, r_cell, dimi, dst_d );
            }
        }
        else
        {
            // Gather surface points for each wedge.
            dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
            wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

            // Gather wedge radii.
            const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
            const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

            dense::Vec< ScalarT, 6 > k_local_hex[num_wedges_per_hex_cell];
            extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

            constexpr int hex_offset_x[8] = { 0, 1, 0, 1, 0, 1, 0, 1 };
            constexpr int hex_offset_y[8] = { 0, 0, 1, 1, 0, 0, 1, 1 };
            constexpr int hex_offset_r[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

            // FE dimensions: velocity coupling components of epsilon operator

            for ( int dimi = 0; dimi < 3; ++dimi )
            {
                for ( int dimj = 0; dimj < 3; ++dimj )
                {
                    if ( diagonal_ and dimi != dimj )
                        continue;

                    ScalarType src_local_hex[8] = { 0 };
                    ScalarType dst_local_hex[8] = { 0 };

                    for ( int i = 0; i < 8; i++ )
                    {
                        src_local_hex[i] = src_(
                            local_subdomain_id,
                            x_cell + hex_offset_x[i],
                            y_cell + hex_offset_y[i],
                            r_cell + hex_offset_r[i],
                            dimi );
                    }

                    // spatial dimensions: quadrature points and wedge
                    for ( int q = 0; q < num_quad_points; q++ )
                    {
                        for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
                        {
                            dense::Mat< ScalarType, VecDim, VecDim > sym_grad_i[num_nodes_per_wedge];
                            dense::Mat< ScalarType, VecDim, VecDim > sym_grad_j[num_nodes_per_wedge];
                            ScalarType                               jdet_keval_quadweight = 0;

                            assemble_trial_test_vecs(
                                wedge,
                                quad_points[q],
                                quad_weights[q],
                                r_1,
                                r_2,
                                wedge_phy_surf,
                                k_local_hex,
                                dimi,
                                dimj,
                                sym_grad_i,
                                sym_grad_j,
                                jdet_keval_quadweight );

                            fused_local_mv(
                                src_local_hex,
                                dst_local_hex,
                                wedge,
                                jdet_keval_quadweight,
                                sym_grad_i,
                                sym_grad_j,
                                dimi,
                                dimj,
                                r_cell );
                        }
                    }

                    for ( int i = 0; i < 8; i++ )
                    {
                        Kokkos::atomic_add(
                            &dst_(
                                local_subdomain_id,
                                x_cell + hex_offset_x[i],
                                y_cell + hex_offset_y[i],
                                r_cell + hex_offset_r[i],
                                dimj ),
                            dst_local_hex[i] );
                    }
                }
            }
        }
    }

    /// @brief assemble the local matrix and return it for a given element, wedge, and vectorial component
    /// (determined by dimi, dimj)
    KOKKOS_INLINE_FUNCTION
    dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > assemble_local_matrix(
        const int local_subdomain_id,
        const int x_cell,
        const int y_cell,
        const int r_cell,
        const int wedge ) const
    {
        // Gather surface points for each wedge.
        // TODO gather this for only 1 wedge
        dense::Vec< ScalarT, 3 > wedge_phy_surf[num_wedges_per_hex_cell][num_nodes_per_wedge_surface] = {};
        wedge_surface_physical_coords( wedge_phy_surf, grid_, local_subdomain_id, x_cell, y_cell );

        // Gather wedge radii.
        const ScalarT r_1 = radii_( local_subdomain_id, r_cell );
        const ScalarT r_2 = radii_( local_subdomain_id, r_cell + 1 );

        dense::Vec< ScalarT, 6 > k_local_hex[num_wedges_per_hex_cell];
        extract_local_wedge_scalar_coefficients( k_local_hex, local_subdomain_id, x_cell, y_cell, r_cell, k_ );

        // Compute the local element matrix.
        dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > A = { 0 };
        for ( int dimi = 0; dimi < 3; ++dimi )
        {
            for ( int dimj = 0; dimj < 3; ++dimj )
            {
                // spatial dimensions: quadrature points and wedge
                for ( int q = 0; q < num_quad_points; q++ )
                {
                    dense::Mat< ScalarType, VecDim, VecDim > sym_grad_i[num_nodes_per_wedge];
                    dense::Mat< ScalarType, VecDim, VecDim > sym_grad_j[num_nodes_per_wedge];
                    ScalarType                               jdet_keval_quadweight = 0;
                    assemble_trial_test_vecs(
                        wedge,
                        quad_points[q],
                        quad_weights[q],
                        r_1,
                        r_2,
                        wedge_phy_surf,
                        k_local_hex,
                        dimi,
                        dimj,
                        sym_grad_i,
                        sym_grad_j,
                        jdet_keval_quadweight );

                    // propagate on local matrix by outer product of test and trial vecs
                    for ( int i = 0; i < num_nodes_per_wedge; i++ )
                    {
                        for ( int j = 0; j < num_nodes_per_wedge; j++ )
                        {
                            A( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) += jdet_keval_quadweight *
                                         ( 2 * sym_grad_j[j].double_contract( sym_grad_i[i] ) -
                                           2.0 / 3.0 * sym_grad_j[j]( dimj, dimj ) * sym_grad_i[i]( dimi, dimi ) );
                            // for the div, we just extract the component from the gradient vector
                        }
                    }
                }
            }
        }

        if ( treat_boundary_ )
        {
            dense::Mat< ScalarT, LocalMatrixDim, LocalMatrixDim > boundary_mask;
            boundary_mask.fill( 1.0 );

            for ( int dimi = 0; dimi < 3; ++dimi )
            {
                for ( int dimj = 0; dimj < 3; ++dimj )
                {
                    if ( r_cell == 0 )
                    {
                        // Inner boundary (CMB).
                        for ( int i = 0; i < 6; i++ )
                        {
                            for ( int j = 0; j < 6; j++ )
                            {
                                // on diagonal components of the vectorial diffusion operator, we exclude the diagonal entries from elimination
                                if ( ( dimi == dimj && i != j && ( i < 3 || j < 3 ) ) or
                                     ( dimi != dimj && ( i < 3 || j < 3 ) ) )
                                {
                                    boundary_mask( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) =
                                        0.0;
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
                                // on diagonal components of the vectorial diffusion operator, we exclude the diagonal entries from elimination
                                if ( ( dimi == dimj && i != j && ( i >= 3 || j >= 3 ) ) or
                                     ( dimi != dimj && ( i >= 3 || j >= 3 ) ) )
                                {
                                    boundary_mask( i + dimi * num_nodes_per_wedge, j + dimj * num_nodes_per_wedge ) =
                                        0.0;
                                }
                            }
                        }
                    }
                }
            }
            A.hadamard_product( boundary_mask );
        }

        return A;
    }

    // executes the fused local matvec on an element, given the assembled trial and test vectors
    KOKKOS_INLINE_FUNCTION void fused_local_mv(
        ScalarType                      src_local_hex[8],
        ScalarType                      dst_local_hex[8],
        const int                       wedge,
        const ScalarType                jdet_keval_quadweight,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_i,
        dense::Mat< ScalarType, 3, 3 >* sym_grad_j,
        const int                       dimi,
        const int                       dimj,
        int                             r_cell ) const
    {
        constexpr int offset_x[2][6] = { { 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 1, 0, 1 } };
        constexpr int offset_y[2][6] = { { 0, 0, 1, 0, 0, 1 }, { 1, 1, 0, 1, 1, 0 } };
        constexpr int offset_r[2][6] = { { 0, 0, 0, 1, 1, 1 }, { 0, 0, 0, 1, 1, 1 } };

        dense::Mat< ScalarType, 3, 3 > grad_u;
        ScalarType                     divu = 0.0;
        grad_u.fill( 0.0 );

        const bool at_cmb        = r_cell == 0;
        const bool at_surface    = r_cell + 1 == radii_.extent( 1 ) - 1;
        int        cmb_shift     = 0;
        int        surface_shift = 0;

        // Compute ∇u at this quadrature point.
        if ( !diagonal_ )
        {
            if ( treat_boundary_ && at_cmb )
            {
                // at the core-mantle boundary, we exclude dofs that are lower-indexed than the dof on the boundary
                cmb_shift = 3;
            }
            else if ( treat_boundary_ && at_surface )
            {
                // at the surface boundary, we exclude dofs that are higher-indexed than the dof on the boundary
                surface_shift = 3;
            }

            // accumulate the element-local gradient/divergence of the trial function (loop over columns of local matrix/local dofs)
            // by dot of trial vec and src dofs
            for ( int i = 0 + cmb_shift; i < num_nodes_per_wedge - surface_shift; i++ )
            {
                grad_u =
                    grad_u +
                    sym_grad_i[i] * src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];
                divu += sym_grad_i[i]( dimi, dimi ) *
                        src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];
            }

            // Add the test function contributions.
            // for each row of the local matrix (test-functions):
            // multiply trial part (fully assembled for the current element from loop above) with test part corresponding to the current row/dof
            // += due to contributions from other elements
            for ( int j = 0 + cmb_shift; j < num_nodes_per_wedge - surface_shift; j++ )
            {
                dst_local_hex[4 * offset_r[wedge][j] + 2 * offset_y[wedge][j] + offset_x[wedge][j]] +=
                    jdet_keval_quadweight * ( 2 * ( sym_grad_j[j] ).double_contract( grad_u ) -
                                              2.0 / 3.0 * sym_grad_j[j]( dimj, dimj ) * divu );
                // for the div, we just extract the component from the gradient vector
            }
        }

        // Dirichlet DoFs are only to be eliminated on diagonal blocks of epsilon
        if ( diagonal_ || ( dimi == dimj && ( treat_boundary_ && ( at_cmb || at_surface ) ) ) )
        {
            // for the diagonal elements at the boundary, we switch the shifts
            for ( int i = 0 + surface_shift; i < num_nodes_per_wedge - cmb_shift; i++ )
            {
                const auto grad_u_diag =
                    sym_grad_i[i] * src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];
                const auto div_u_diag =
                    sym_grad_i[i]( dimi, dimi ) *
                    src_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]];

                dst_local_hex[4 * offset_r[wedge][i] + 2 * offset_y[wedge][i] + offset_x[wedge][i]] +=
                    jdet_keval_quadweight * ( 2 * ( sym_grad_j[i] ).double_contract( grad_u_diag ) -
                                              2.0 / 3.0 * sym_grad_j[i]( dimj, dimj ) * div_u_diag );
            }
        }
    }
};

static_assert( linalg::GCACapable< EpsilonDivDiv< float > > );
static_assert( linalg::GCACapable< EpsilonDivDiv< double > > );

} // namespace terra::fe::wedge::operators::shell
