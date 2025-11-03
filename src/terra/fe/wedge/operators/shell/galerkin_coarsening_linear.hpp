

#pragma once

#include "communication/shell/communication.hpp"
#include "dense/vec.hpp"
#include "fe/wedge/shell/grid_transfer_linear.hpp"
#include "grid/grid_types.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"

namespace terra::fe::wedge::operators::shell {

/// @brief: Galerkin coarse approximation (GCA). 
/// TwoGridGCA takes a coarser and a finer operator. Each thread assembles a 
/// coarse-grid gca matrix in the coarser operator on a single hex. To do this, it loops 
/// the finer hexes of the coarse hex and its respective wedges. It computes the interpolation 
/// matrix P mapping from coarse wedge to the current fine wedge, computes the 
/// triple-product P^TAP with the fine-operator local matrix A and adds the resulting gca matrix 
/// up for all fine wedges comprising the coarse wedge. Finally, it stores the result in the 
/// wedge-wise matrix storage of the coarse operator.
template < typename ScalarT, typename Operator >
class TwoGridGCA
{
  public:
    using SrcVectorType = linalg::VectorQ1Scalar< double >;
    using DstVectorType = linalg::VectorQ1Scalar< double >;
    using ScalarType    = ScalarT;

  private:
    grid::shell::DistributedDomain&    domain_fine_;
    grid::Grid3DDataVec< ScalarT, 3 >& grid_fine_;
    grid::Grid2DDataScalar< ScalarT >& radii_fine_;
    grid::Grid2DDataScalar< ScalarT >& radii_coarse_;
    Operator&                          fine_op_;
    Operator&                          coarse_op_;
    bool                               treat_boundary_;

  public:
    explicit TwoGridGCA( Operator fine_op, Operator coarse_op, bool treat_boundary = true )
    : domain_fine_( fine_op.get_domain() )
    , fine_op_( fine_op )
    , coarse_op_( coarse_op )
    , grid_fine_( fine_op.get_grid() )
    , radii_fine_( fine_op.get_radii() )
    , radii_coarse_( coarse_op.get_radii() )
    , treat_boundary_( treat_boundary )

    {
        // this probably cant not happen
        if ( coarse_op.get_domain().subdomains().size() != domain_fine_.subdomains().size() )
        {
            throw std::runtime_error( "Prolongation: src and dst must have a compatible number of subdomains." );
        }

        if ( 2 * ( coarse_op.get_domain().domain_info().subdomain_num_nodes_per_side_laterally() - 1 ) !=
             domain_fine_.domain_info().subdomain_num_nodes_per_side_laterally() - 1 )
        {
            throw std::runtime_error( "Prolongation: src and dst must have a compatible number of lateral cells." );
        }
        if ( 2 * ( coarse_op.get_domain().domain_info().subdomain_num_nodes_radially() - 1 ) !=
             domain_fine_.domain_info().subdomain_num_nodes_radially() - 1 )
        {
            throw std::runtime_error( "Prolongation: src and dst must have a compatible number of radial cells." );
        }

        // Looping over the coarse grid.
        Kokkos::parallel_for(
            "gca_coarsening",
            Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
                { 0, 0, 0, 0 },
                {
                    static_cast< long long >( coarse_op.get_domain().subdomains().size() ),
                    coarse_op.get_domain().domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                    coarse_op.get_domain().domain_info().subdomain_num_nodes_per_side_laterally() - 1,
                    coarse_op.get_domain().domain_info().subdomain_num_nodes_radially() - 1,
                } ),
            *this );

        Kokkos::fence();
    }

    /// @brief Computes indices of vertices associated to a wedge in a hex cell.
    /// @param coarse_hex_idx  [in] global index of the hex cell
    /// @param wedge  [in] wedge index (local index 0 or 1)
    /// @param wedge_local_vertex_indices  [out] global indices of the vertices located on the wedge
    KOKKOS_INLINE_FUNCTION void wedge_vertex_indices(
        dense::Vec< int, 4 > hex_idx,
        int                  wedge,
        dense::Vec< int, 4 > ( &wedge_local_vertex_indices )[6] ) const
    {
        if ( wedge == 0 )
        {
            wedge_local_vertex_indices[0] = hex_idx;
            wedge_local_vertex_indices[1] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 0 } );
            wedge_local_vertex_indices[2] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 0 } );
            wedge_local_vertex_indices[3] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 0, 1 } );
            wedge_local_vertex_indices[4] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 1 } );
            wedge_local_vertex_indices[5] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 1 } );
        }
        else
        {
            wedge_local_vertex_indices[0] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 1, 0 } );
            wedge_local_vertex_indices[1] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 0 } );
            wedge_local_vertex_indices[2] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 0 } );
            wedge_local_vertex_indices[3] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 1, 1 } );
            wedge_local_vertex_indices[4] = hex_idx + dense::Vec< int, 4 >( { 0, 0, 1, 1 } );
            wedge_local_vertex_indices[5] = hex_idx + dense::Vec< int, 4 >( { 0, 1, 0, 1 } );
        }
    }

    KOKKOS_INLINE_FUNCTION void operator()(
        const int local_subdomain_id,
        const int x_coarse_idx,
        const int y_coarse_idx,
        const int r_coarse_idx ) const
    {
        dense::Vec< int, 4 > fine_hex_shifts[8] = {
            { 0, 0, 0, 0 },
            { 0, 1, 0, 0 },
            { 0, 0, 1, 0 },
            { 0, 1, 1, 0 },
            { 0, 0, 0, 1 },
            { 0, 1, 0, 1 },
            { 0, 0, 1, 1 },
            { 0, 1, 1, 1 },
        };

        dense::Vec< int, 4 > coarse_hex_idx      = { local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx };
        dense::Vec< int, 4 > coarse_hex_idx_fine = {
            local_subdomain_id, 2 * x_coarse_idx, 2 * y_coarse_idx, 2 * r_coarse_idx };

        dense::Mat< ScalarT, 6, 6 > A_coarse[num_wedges_per_hex_cell] = { { 0 }, { 0 } };
        // loop finer hexes of our coarse hex
        for ( int fine_hex_lidx = 0; fine_hex_lidx < 8; fine_hex_lidx++ )
        {
            auto fine_hex_idx = coarse_hex_idx_fine + fine_hex_shifts[fine_hex_lidx];

            // two wedges per fine hex
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                dense::Mat< ScalarT, 6, 6 > P = { 0 };

                // obtain vertex indices of the current fine wedge
                dense::Vec< int, 4 > wedge_local_vertex_indices_fine[6];
                wedge_vertex_indices( fine_hex_idx, wedge, wedge_local_vertex_indices_fine );

                // compute local (fully-assembled!) interpolation matrices mapping from the coarse DoFs in the hex to the current fine wedge DoFs

                // loop destination of the interpolation (row dim of P): fine DoFs
                for ( int fine_dof_lidx = 0; fine_dof_lidx < num_nodes_per_wedge; fine_dof_lidx++ )
                {
                    auto fine_dof_idx = wedge_local_vertex_indices_fine[fine_dof_lidx];

                    // fine dof is on coarse dof
                    if ( fine_dof_idx( 1 ) % 2 == 0 && fine_dof_idx( 2 ) % 2 == 0 && fine_dof_idx( 3 ) % 2 == 0 )
                    {
                        // local index of destination fine DoF == local index of source coarse DoF
                        P( fine_dof_lidx, fine_dof_lidx ) = 1.0;
                        continue;
                    }

                    // else: need radial direction bot (>=) and top (<=) of current fine DoF
                    const auto r_idx_coarse_bot = fine_dof_idx( 3 ) < radii_fine_.extent( 1 ) - 1 ?
                                                      fine_dof_idx( 3 ) / 2 :
                                                      fine_dof_idx( 3 ) / 2 - 1;
                    const auto r_idx_coarse_top = r_idx_coarse_bot + 1;

                    // fine dof is radially aligned: x and y index match with coarse DoFs
                    // interpolate on the line in radial direction (coarse DoF bot -- fine DoF -- coarse DoF top)
                    if ( fine_dof_idx( 1 ) % 2 == 0 && fine_dof_idx( 2 ) % 2 == 0 )
                    {
                        // x, y on coarse, so we can just divide by 2 to obtain coarse indices
                        const auto fine_dof_x_idx_coarse = fine_dof_idx( 1 ) / 2;
                        const auto fine_dof_y_idx_coarse = fine_dof_idx( 2 ) / 2;

                        // actual weight computation
                        const auto weights = wedge::shell::prolongation_linear_weights(
                            dense::Vec< int, 4 >{
                                local_subdomain_id, fine_dof_idx( 1 ), fine_dof_idx( 2 ), fine_dof_idx( 3 ) },
                            dense::Vec< int, 4 >{
                                local_subdomain_id, fine_dof_x_idx_coarse, fine_dof_y_idx_coarse, r_idx_coarse_bot },
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
                    int coarse_dof_lindices[4] = { -1 };

                    if ( fine_dof_idx( 1 ) % 2 == 0 )
                    {
                        // "Vertical" edge.
                        x0_idx_coarse = fine_dof_idx( 1 ) / 2;
                        x1_idx_coarse = fine_dof_idx( 1 ) / 2;

                        y0_idx_coarse = fine_dof_idx( 2 ) / 2;
                        y1_idx_coarse = fine_dof_idx( 2 ) / 2 + 1;

                        coarse_dof_lindices[0] = 0;
                        coarse_dof_lindices[1] = 2;
                        coarse_dof_lindices[2] = 3;
                        coarse_dof_lindices[3] = 5;
                    }
                    else if ( fine_dof_idx( 2 ) % 2 == 0 )
                    {
                        // "Horizontal" edge.
                        x0_idx_coarse = fine_dof_idx( 1 ) / 2;
                        x1_idx_coarse = fine_dof_idx( 1 ) / 2 + 1;

                        y0_idx_coarse = fine_dof_idx( 2 ) / 2;
                        y1_idx_coarse = fine_dof_idx( 2 ) / 2;

                        coarse_dof_lindices[0] = 0;
                        coarse_dof_lindices[1] = 1;
                        coarse_dof_lindices[2] = 3;
                        coarse_dof_lindices[3] = 4;
                    }
                    else
                    {
                        // "Diagonal" edge.
                        x0_idx_coarse = fine_dof_idx( 1 ) / 2 + 1;
                        x1_idx_coarse = fine_dof_idx( 1 ) / 2;

                        y0_idx_coarse = fine_dof_idx( 2 ) / 2;
                        y1_idx_coarse = fine_dof_idx( 2 ) / 2 + 1;

                        coarse_dof_lindices[0] = 1;
                        coarse_dof_lindices[1] = 2;
                        coarse_dof_lindices[2] = 4;
                        coarse_dof_lindices[3] = 5;
                    }

                    const auto weights = wedge::shell::prolongation_linear_weights(
                        dense::Vec< int, 4 >{
                            local_subdomain_id, fine_dof_idx( 1 ), fine_dof_idx( 2 ), fine_dof_idx( 3 ) },
                        dense::Vec< int, 4 >{ local_subdomain_id, x0_idx_coarse, y0_idx_coarse, r_idx_coarse_bot },
                        dense::Vec< int, 4 >{ local_subdomain_id, x1_idx_coarse, y1_idx_coarse, r_idx_coarse_bot },
                        grid_fine_,
                        radii_fine_ );

                    P( fine_dof_lidx, coarse_dof_lindices[0] ) = weights( 0 );
                    P( fine_dof_lidx, coarse_dof_lindices[1] ) = weights( 0 );
                    P( fine_dof_lidx, coarse_dof_lindices[2] ) = weights( 1 );
                    P( fine_dof_lidx, coarse_dof_lindices[3] ) = weights( 1 );
                }

                dense::Mat< ScalarT, 6, 6 > A_fine = fine_op_.get_lmatrix(
                    local_subdomain_id, fine_hex_idx( 1 ), fine_hex_idx( 2 ), fine_hex_idx( 3 ), wedge );

                // core part: assemble local gca matrix by mapping from coarse wedge to current fine wedge,
                // applying the corresponding local operator and mapping back.
                auto PAP = P.transposed() * A_fine * P;

                // correctly add to gca coarsened matrix
                // depending on the fine hex and wedge, we are located on the coarse 0 or 1 wedge
                // and need to add to the corresponding coarse matrix
                if ( ( wedge == 0 && ( fine_hex_lidx == 0 || fine_hex_lidx == 1 || fine_hex_lidx == 2 ||
                                       fine_hex_lidx == 4 || fine_hex_lidx == 5 || fine_hex_lidx == 6 ) ) or
                     ( wedge == 1 && ( fine_hex_lidx == 0 || fine_hex_lidx == 4 ) ) )
                {
                    A_coarse[0] += PAP;
                }
                else if (
                    ( wedge == 1 && ( fine_hex_lidx == 1 || fine_hex_lidx == 2 || fine_hex_lidx == 3 ||
                                      fine_hex_lidx == 5 || fine_hex_lidx == 6 || fine_hex_lidx == 7 ) ) or
                    ( wedge == 0 && ( fine_hex_lidx == 3 || fine_hex_lidx == 7 ) ) )
                {
                    A_coarse[1] += PAP;
                }
                else
                {
                    Kokkos::abort( "Unexpected path." );
                }
            }
        }

        if ( treat_boundary_ )
        {
            for ( int wedge = 0; wedge < num_wedges_per_hex_cell; wedge++ )
            {
                dense::Mat< ScalarT, 6, 6 > boundary_mask;
                boundary_mask.fill( 1.0 );
                if ( r_coarse_idx == 0 )
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

                if ( r_coarse_idx + 1 == radii_coarse_.extent( 1 ) - 1 )
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

                A_coarse[wedge].hadamard_product( boundary_mask );
            }
        }

        // store coarse matrices
        coarse_op_.get_lmatrix( local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx, 0 ) = A_coarse[0];
        coarse_op_.get_lmatrix( local_subdomain_id, x_coarse_idx, y_coarse_idx, r_coarse_idx, 1 ) = A_coarse[1];
    }
};
} // namespace terra::fe::wedge::operators::shell