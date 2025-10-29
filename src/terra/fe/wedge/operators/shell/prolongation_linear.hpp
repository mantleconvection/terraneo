

#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/shell/grid_transfer_linear.hpp"
#include "grid/grid_types.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

template < typename ScalarT >
class ProlongationLinear
{
  public:
    using SrcVectorType           = linalg::VectorQ1Scalar< double >;
    using DstVectorType           = linalg::VectorQ1Scalar< double >;
    using ScalarType              = ScalarT;
    using Grid4DDataLocalMatrices = terra::grid::Grid4DDataMatrices< ScalarType, 6, 6, 2 >;

  private:
    bool storeLMatrices_ =
        false; // set to let apply_impl() know, that it should store the local matrices after assembling them
    bool applyStoredLMatrices_ =
        false; // set to make apply_impl() load and use the stored LMatrices for the operator application

    Grid4DDataLocalMatrices LMatrices_;

    grid::Grid3DDataVec< ScalarType, 3 > grid_fine_;
    grid::Grid2DDataScalar< ScalarType > radii_fine_;
    grid::shell::DistributedDomain       domain_fine_;
    grid::shell::DistributedDomain       domain_coarse_;

    linalg::OperatorApplyMode operator_apply_mode_;

    grid::Grid4DDataScalar< ScalarType > src_;
    grid::Grid4DDataScalar< ScalarType > dst_;

  public:
    explicit ProlongationLinear(
        const grid::shell::DistributedDomain&       domain_fine,
        const grid::shell::DistributedDomain&       domain_coarse,
        const grid::Grid3DDataVec< ScalarType, 3 >& grid_fine,
        const grid::Grid2DDataScalar< ScalarType >& radii_fine,
        linalg::OperatorApplyMode                   operator_apply_mode = linalg::OperatorApplyMode::Replace )
    : grid_fine_( grid_fine )
    , radii_fine_( radii_fine )
    , operator_apply_mode_( operator_apply_mode )
    , domain_fine_( domain_fine )
    , domain_coarse_( domain_coarse )
    {}

    void storeLMatrices()
    {
        storeLMatrices_ = true;
        if ( LMatrices_.data() == nullptr )
        {
            LMatrices_ = Grid4DDataLocalMatrices(
                "ProlongationLinear::LMatrices",
                domain_fine_.subdomains().size(),
                domain_fine_.domain_info().subdomain_num_nodes_per_side_laterally(),
                domain_fine_.domain_info().subdomain_num_nodes_per_side_laterally(),
                domain_fine_.domain_info().subdomain_num_nodes_radially() );
            Kokkos::parallel_for( "matvec", grid::shell::local_domain_md_range_policy_cells( domain_coarse_ ), *this );
            Kokkos::fence();
        }
        storeLMatrices_ = false;
    }

    void apply_impl( const SrcVectorType& src, DstVectorType& dst )
    {
        if ( storeLMatrices_ or applyStoredLMatrices_ )
            assert( LMatrices_.data() != nullptr );

        if ( operator_apply_mode_ == linalg::OperatorApplyMode::Replace )
        {
            assign( dst, 0 );
        }

        src_ = src.grid_data();
        dst_ = dst.grid_data();

        if ( dst_.extent( 1 ) != grid_fine_.extent( 1 ) )
        {
            throw std::runtime_error(
                "Prolongation: dst and grid_fine must have the same number of cells in the x direction." );
        }

        if ( dst_.extent( 3 ) != radii_fine_.extent( 1 ) )
        {
            throw std::runtime_error(
                "Prolongation: dst and radii_fine must have the same number of cells in the r direction." );
        }

        if ( src_.extent( 0 ) != dst_.extent( 0 ) )
        {
            throw std::runtime_error( "Prolongation: src and dst must have the same number of subdomains." );
        }

        for ( int i = 1; i <= 3; i++ )
        {
            if ( 2 * ( src_.extent( i ) - 1 ) != dst_.extent( i ) - 1 )
            {
                throw std::runtime_error( "Prolongation: src and dst must have a compatible number of cells." );
            }
        }

        // Looping over the fine grid.
        Kokkos::parallel_for(
            "matvec",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                {
                    dst_.extent( 0 ),
                    dst_.extent( 1 ),
                    dst_.extent( 2 ),
                    dst_.extent( 3 ),
                } ),
            *this );

        Kokkos::fence();
    }

    /// @brief Computes indices of vertices associated to a wedge in a hex cell.
    /// @param coarse_hex_idx  [in] global index of the hex cell
    /// @param wedge  [in] coarse wedge index (local index 0 or 1)
    /// @param wedge_local_vertex_indices  [out] global indices of the vertices located on the fine wedge
    KOKKOS_INLINE_FUNCTION void wedge_vertex_indices(
        dense::Vec< int, 3 > coarse_hex_idx,
        int                  wedge,
        dense::Vec< int, 3 > ( &wedge_local_vertex_indices )[6] ) const
    {
        if ( wedge == 0 )
        {
            wedge_local_vertex_indices[0] = coarse_hex_idx;
            wedge_local_vertex_indices[1] = coarse_hex_idx + dense::Vec< int, 3 >( { 1, 0, 0 } );
            wedge_local_vertex_indices[2] = coarse_hex_idx + dense::Vec< int, 3 >( { 0, 1, 0 } );
            wedge_local_vertex_indices[3] = coarse_hex_idx + dense::Vec< int, 3 >( { 0, 0, 1 } );
            wedge_local_vertex_indices[4] = coarse_hex_idx + dense::Vec< int, 3 >( { 1, 0, 1 } );
            wedge_local_vertex_indices[5] = coarse_hex_idx + dense::Vec< int, 3 >( { 0, 1, 1 } );
        }
        else
        {
            wedge_local_vertex_indices[0] = coarse_hex_idx + dense::Vec< int, 3 >( { 1, 1, 0 } );
            wedge_local_vertex_indices[1] = coarse_hex_idx + dense::Vec< int, 3 >( { 0, 1, 0 } );
            wedge_local_vertex_indices[2] = coarse_hex_idx + dense::Vec< int, 3 >( { 1, 0, 0 } );
            wedge_local_vertex_indices[3] = coarse_hex_idx + dense::Vec< int, 3 >( { 1, 1, 1 } );
            wedge_local_vertex_indices[4] = coarse_hex_idx + dense::Vec< int, 3 >( { 0, 1, 1 } );
            wedge_local_vertex_indices[5] = coarse_hex_idx + dense::Vec< int, 3 >( { 1, 0, 1 } );
        }
    }

    KOKKOS_INLINE_FUNCTION void
        operator()( const int local_subdomain_id, const int x_idx, const int y_idx, const int r_idx ) const
    {
        dense::Vec< int, 3 > fine_hex_shifts[8] = {
            { 0, 0, 0 },
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 1, 1, 0 },
            { 0, 0, 1 },
            { 1, 0, 1 },
            { 0, 1, 1 },
            { 1, 1, 1 },
        };

        /*if ( x_idx % 2 == 0 && y_idx % 2 == 0 && r_idx % 2 == 0 )
            {
                const auto x_coarse = x_idx / 2;
                const auto y_coarse = y_idx / 2;
                const auto r_coarse = r_idx / 2;

                dst_( local_subdomain_id, x_idx, y_idx, r_idx ) +=
                    src_( local_subdomain_id, x_coarse, y_coarse, r_coarse );

                return;
            }

            const auto r_coarse_bot = r_idx < dst_.extent( 3 ) - 1 ? r_idx / 2 : r_idx / 2 - 1;
            const auto r_coarse_top = r_coarse_bot + 1;

            if ( x_idx % 2 == 0 && y_idx % 2 == 0 )
            {
                const auto x_coarse = x_idx / 2;
                const auto y_coarse = y_idx / 2;

                const auto weights = wedge::shell::prolongation_linear_weights(
                    dense::Vec< int, 4 >{ local_subdomain_id, x_idx, y_idx, r_idx },
                    dense::Vec< int, 4 >{ local_subdomain_id, x_coarse, y_coarse, r_coarse_bot },
                    grid_fine_,
                    radii_fine_ );

                dst_( local_subdomain_id, x_idx, y_idx, r_idx ) +=
                    weights( 0 ) * src_( local_subdomain_id, x_coarse, y_coarse, r_coarse_bot ) +
                    weights( 1 ) * src_( local_subdomain_id, x_coarse, y_coarse, r_coarse_top );

                return;
            }

            int x_coarse_0 = -1;
            int x_coarse_1 = -1;

            int y_coarse_0 = -1;
            int y_coarse_1 = -1;

            if ( x_idx % 2 == 0 )
            {
                // "Vertical" edge.
                x_coarse_0 = x_idx / 2;
                x_coarse_1 = x_idx / 2;

                y_coarse_0 = y_idx / 2;
                y_coarse_1 = y_idx / 2 + 1;
            }
            else if ( y_idx % 2 == 0 )
            {
                // "Horizontal" edge.
                x_coarse_0 = x_idx / 2;
                x_coarse_1 = x_idx / 2 + 1;

                y_coarse_0 = y_idx / 2;
                y_coarse_1 = y_idx / 2;
            }
            else
            {
                // "Diagonal" edge.
                x_coarse_0 = x_idx / 2 + 1;
                x_coarse_1 = x_idx / 2;

                y_coarse_0 = y_idx / 2;
                y_coarse_1 = y_idx / 2 + 1;
            }

            const auto weights = wedge::shell::prolongation_linear_weights(
                dense::Vec< int, 4 >{ local_subdomain_id, x_idx, y_idx, r_idx },
                dense::Vec< int, 4 >{ local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_bot },
                dense::Vec< int, 4 >{ local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_bot },
                grid_fine_,
                radii_fine_ );

            dst_( local_subdomain_id, x_idx, y_idx, r_idx ) +=
                weights( 0 ) * src_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_bot ) +
                weights( 0 ) * src_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_bot ) +
                weights( 1 ) * src_( local_subdomain_id, x_coarse_0, y_coarse_0, r_coarse_top ) +
                weights( 1 ) * src_( local_subdomain_id, x_coarse_1, y_coarse_1, r_coarse_top );
        }
        else
        {
        */
        dense::Vec< int, 3 > coarse_hex_idx      = { x_idx, y_idx, r_idx };
        dense::Vec< int, 3 > coarse_hex_idx_fine = { 2 * x_idx, 2 * y_idx, 2 * r_idx };
        dense::Vec< int, 3 > wedge_local_vertex_indices_fine[6];
        dense::Vec< int, 3 > wedge_local_vertex_indices_coarse[6];

        // loop finer hexes of our coarse hex
        for ( int fine_hex_lidx = 0; fine_hex_lidx < 8; fine_hex_lidx++ )
        {
            auto fine_hex_idx = coarse_hex_idx_fine + fine_hex_shifts[fine_hex_lidx];

            // two wedges per fine hex
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                dense::Mat< ScalarT, 6, 6 > P = { 0 };

                // obtain vertex indices of the current fine wedge
                wedge_vertex_indices( fine_hex_idx, wedge, wedge_local_vertex_indices_fine );

                // compute local (fully-assembled!) interpolation matrices mapping from the coarse DoFs in the hex to the current fine wedge DoFs
                if ( !applyStoredLMatrices_ )
                {
                    // loop destination of the interpolation (row dim of P): fine DoFs
                    for ( int fine_dof_lidx = 0; fine_dof_lidx < num_nodes_per_wedge; fine_dof_lidx++ )
                    {
                        auto fine_dof_idx = wedge_local_vertex_indices_fine[fine_dof_lidx];

                        // fine dof is on coarse dof
                        if ( fine_dof_idx( 0 ) % 2 == 0 && fine_dof_idx( 1 ) % 2 == 0 && fine_dof_idx( 2 ) % 2 == 0 )
                        {
                            // local index of destination fine DoF == local index of source coarse DoF
                            P( fine_dof_lidx, fine_dof_lidx ) = 1.0;
                            continue;
                        }

                        // else: need radial direction bot (>=) and top (<=) of current fine DoF
                        const auto r_idx_coarse_bot = fine_dof_idx( 2 ) < dst_.extent( 3 ) - 1 ?
                                                          fine_dof_idx( 2 ) / 2 :
                                                          fine_dof_idx( 2 ) / 2 - 1;
                        const auto r_idx_coarse_top = r_idx_coarse_bot + 1;

                        // fine dof is radially aligned: x and y index match with coarse DoFs
                        // interpolate on the line in radial direction (coarse DoF bot -- fine DoF -- coarse DoF top)
                        if ( fine_dof_idx( 0 ) % 2 == 0 && fine_dof_idx( 1 ) % 2 == 0 )
                        {
                            // x, y on coarse, so we can just divide by 2 to obtain coarse indices
                            const auto fine_dof_x_idx_coarse = fine_dof_idx( 0 ) / 2;
                            const auto fine_dof_y_idx_coarse = fine_dof_idx( 1 ) / 2;

                            // actualy weight computation
                            const auto weights = wedge::shell::prolongation_linear_weights(
                                dense::Vec< int, 4 >{
                                    local_subdomain_id, fine_dof_idx( 0 ), fine_dof_idx( 1 ), fine_dof_idx( 2 ) },
                                dense::Vec< int, 4 >{
                                    local_subdomain_id,
                                    fine_dof_x_idx_coarse,
                                    fine_dof_y_idx_coarse,
                                    r_idx_coarse_bot },
                                grid_fine_,
                                radii_fine_ );

                            // is hurts but we only do it once for assembling local Ps so its fine
                            // local indices of coarse DoFs can be determined analytically
                            if ( fine_dof_lidx == 2 or fine_dof_lidx == 5 )
                            {
                                P( fine_dof_lidx, 2 ) = weights( 0 );
                                P( fine_dof_lidx, 5 ) = weights( 1 );
                            }
                            else if ( fine_dof_lidx == 0 or fine_dof_lidx == 3 )
                            {
                                P( fine_dof_lidx, 0 ) = weights( 0 );
                                P( fine_dof_lidx, 3 ) = weights( 1 );
                            }
                            else if ( fine_dof_lidx == 1 or fine_dof_lidx == 4 )
                            {
                                P( fine_dof_lidx, 1 ) = weights( 0 );
                                P( fine_dof_lidx, 4 ) = weights( 1 );
                            }
                            continue;
                        }

                        // else: we interpolate fine DoF from the plane of 4 coarse DoFs that contains the fine DoF

                        // for the two botting coarse DoFs
                        int x0_idx_coarse = -1;
                        int x1_idx_coarse = -1;
                        int y0_idx_coarse = -1;
                        int y1_idx_coarse = -1;

                        // local indices of the 4 coarse DoFs in the plane
                        int coarse_lindices[4] = { -1 };

                        if ( fine_dof_idx( 0 ) % 2 == 0 )
                        {
                            // "Vertical" edge.
                            x0_idx_coarse = fine_dof_idx( 0 ) / 2;
                            x1_idx_coarse = fine_dof_idx( 0 ) / 2;

                            y0_idx_coarse = fine_dof_idx( 1 ) / 2;
                            y1_idx_coarse = fine_dof_idx( 1 ) / 2 + 1;

                            coarse_lindices[0] = 0;
                            coarse_lindices[1] = 2;
                            coarse_lindices[2] = 4;
                            coarse_lindices[3] = 5;
                        }
                        else if ( fine_dof_idx( 1 ) % 2 == 0 )
                        {
                            // "Horizontal" edge.
                            x0_idx_coarse = fine_dof_idx( 0 ) / 2;
                            x1_idx_coarse = fine_dof_idx( 0 ) / 2 + 1;

                            y0_idx_coarse      = fine_dof_idx( 1 ) / 2;
                            y1_idx_coarse      = fine_dof_idx( 1 ) / 2;
                            coarse_lindices[0] = 0;
                            coarse_lindices[1] = 1;
                            coarse_lindices[2] = 3;
                            coarse_lindices[3] = 4;
                        }
                        else
                        {
                            // "Diagonal" edge.
                            x0_idx_coarse = fine_dof_idx( 0 ) / 2 + 1;
                            x1_idx_coarse = fine_dof_idx( 0 ) / 2;

                            y0_idx_coarse      = fine_dof_idx( 1 ) / 2;
                            y1_idx_coarse      = fine_dof_idx( 1 ) / 2 + 1;
                            coarse_lindices[0] = 1;
                            coarse_lindices[1] = 2;
                            coarse_lindices[2] = 5;
                            coarse_lindices[3] = 6;
                        }

                        const auto weights = wedge::shell::prolongation_linear_weights(
                            dense::Vec< int, 4 >{
                                local_subdomain_id, fine_dof_idx( 0 ), fine_dof_idx( 1 ), fine_dof_idx( 2 ) },
                            dense::Vec< int, 4 >{ local_subdomain_id, x0_idx_coarse, y0_idx_coarse, r_idx_coarse_bot },
                            dense::Vec< int, 4 >{ local_subdomain_id, x1_idx_coarse, y1_idx_coarse, r_idx_coarse_bot },
                            grid_fine_,
                            radii_fine_ );

                        P( fine_dof_lidx, coarse_lindices[0] ) = weights( 0 );
                        P( fine_dof_lidx, coarse_lindices[1] ) = weights( 0 );
                        P( fine_dof_lidx, coarse_lindices[2] ) = weights( 1 );
                        P( fine_dof_lidx, coarse_lindices[3] ) = weights( 1 );
                    }
                }
                else
                {
                    // load LMatrix for the current local fine wedge
                    P = LMatrices_(
                        local_subdomain_id, fine_hex_idx( 0 ), fine_hex_idx( 1 ), fine_hex_idx( 2 ), wedge );
                }

                if ( storeLMatrices_ )
                {
                    // write LMatrix for the current local fine wedge to mem
                    LMatrices_( local_subdomain_id, fine_hex_idx( 0 ), fine_hex_idx( 1 ), fine_hex_idx( 2 ), wedge ) =
                        P;
                }
                else
                {
                    // apply local interpolation to local DoFs
                    dense::Vec< ScalarT, 6 > src;
                    dense::Vec< ScalarT, 6 > dst;
                    if ( wedge == 0 )
                    {
                        src( 0 ) = src_( local_subdomain_id, x_idx, y_idx, r_idx );
                        src( 1 ) = src_( local_subdomain_id, x_idx + 1, y_idx, r_idx );
                        src( 2 ) = src_( local_subdomain_id, x_idx, y_idx + 1, r_idx );
                        src( 3 ) = src_( local_subdomain_id, x_idx, y_idx, r_idx + 1 );
                        src( 4 ) = src_( local_subdomain_id, x_idx + 1, y_idx, r_idx + 1 );
                        src( 5 ) = src_( local_subdomain_id, x_idx, y_idx + 1, r_idx + 1 );

                        dst = P * src;

                        // since the local interpolations are fully assembled, this is not an additive process but we assign
                        // there are redundant assigns from multiple elements connecting two DoFs
                        // the connection values should be the same for both matrices
                        // in practice, the interpolation values should not be stored except to compute Galerkin coarse-grid operators
                        // afterwards, these matrices should be erased from memory
                        dst_( local_subdomain_id, fine_hex_idx( 0 ), fine_hex_idx( 1 ), fine_hex_idx( 2 ) ) = dst( 0 );
                        dst_( local_subdomain_id, fine_hex_idx( 0 ) + 1, fine_hex_idx( 1 ), fine_hex_idx( 2 ) ) =
                            dst( 1 );
                        dst_( local_subdomain_id, fine_hex_idx( 0 ), fine_hex_idx( 1 ) + 1, fine_hex_idx( 2 ) ) =
                            dst( 2 );
                        dst_( local_subdomain_id, fine_hex_idx( 0 ), fine_hex_idx( 1 ), fine_hex_idx( 2 ) + 1 ) =
                            dst( 3 );
                        dst_( local_subdomain_id, fine_hex_idx( 0 ) + 1, fine_hex_idx( 1 ), fine_hex_idx( 2 ) + 1 ) =
                            dst( 4 );
                        dst_( local_subdomain_id, fine_hex_idx( 0 ), fine_hex_idx( 1 ) + 1, fine_hex_idx( 2 ) + 1 ) =
                            dst( 5 );
                    }
                    else
                    {
                        src( 0 ) = src_( local_subdomain_id, x_idx + 1, y_idx + 1, r_idx );
                        src( 1 ) = src_( local_subdomain_id, x_idx + 1, y_idx, r_idx );
                        src( 2 ) = src_( local_subdomain_id, x_idx, y_idx + 1, r_idx );
                        src( 3 ) = src_( local_subdomain_id, x_idx + 1, y_idx + 1, r_idx + 1 );
                        src( 4 ) = src_( local_subdomain_id, x_idx + 1, y_idx, r_idx + 1 );
                        src( 5 ) = src_( local_subdomain_id, x_idx, y_idx + 1, r_idx + 1 );

                        dst = P * src;
                        dst_( local_subdomain_id, fine_hex_idx( 0 ) + 1, fine_hex_idx( 1 ) + 1, fine_hex_idx( 2 ) ) =
                            dst( 0 );
                        dst_( local_subdomain_id, fine_hex_idx( 0 ) + 1, fine_hex_idx( 1 ), fine_hex_idx( 2 ) ) =
                            dst( 1 );
                        dst_( local_subdomain_id, fine_hex_idx( 0 ), fine_hex_idx( 1 ) + 1, fine_hex_idx( 2 ) ) =
                            dst( 2 );
                        dst_(
                            local_subdomain_id, fine_hex_idx( 0 ) + 1, fine_hex_idx( 1 ) + 1, fine_hex_idx( 2 ) + 1 ) =
                            dst( 3 );
                        dst_( local_subdomain_id, fine_hex_idx( 0 ) + 1, fine_hex_idx( 1 ), fine_hex_idx( 2 ) + 1 ) =
                            dst( 4 );
                        dst_( local_subdomain_id, fine_hex_idx( 0 ), fine_hex_idx( 1 ) + 1, fine_hex_idx( 2 ) + 1 ) =
                            dst( 5 );
                    }
                }
            }
        }
    }
};
} // namespace terra::fe::wedge::operators::shell