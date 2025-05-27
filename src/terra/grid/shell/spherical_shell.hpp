#pragma once

#include <cmath>
#include <stdexcept>

#include "../grid_types.hpp"
#include "../terra/kokkos/kokkos_wrapper.hpp"
#include "dense/vec.hpp"

namespace terra::grid::shell {

std::vector< double > uniform_shell_radii( double r_min, double r_max, int num_shells );

class SubdomainInfo
{
  public:
    SubdomainInfo( int diamond_id, int subdomain_x, int subdomain_y, int subdomain_r )
    : diamond_id_( diamond_id )
    , subdomain_x_( subdomain_x )
    , subdomain_y_( subdomain_y )
    , subdomain_r_( subdomain_r )
    {}

    int diamond_id() const { return diamond_id_; }
    int subdomain_x() const { return subdomain_x_; }
    int subdomain_y() const { return subdomain_y_; }
    int subdomain_r() const { return subdomain_r_; }

  private:
    /// Diamond that subdomain is part of.
    int diamond_id_;

    /// Subdomain index in lateral x-direction (local to the diamond).
    int subdomain_x_;

    /// Subdomain index in lateral y-direction (local to the diamond).
    int subdomain_y_;

    /// Subdomain index in radial direction.
    int subdomain_r_;
};

class DomainInfo
{
  public:
    DomainInfo( int global_lateral_refinement_level, double r_min, double r_max, int num_uniform_layers )
    : global_lateral_refinement_level_( global_lateral_refinement_level )
    , radii_( uniform_shell_radii( r_min, r_max, num_uniform_layers + 1 ) )
    , diamond_subdomain_refinement_level_( 0 )
    , num_subdomains_in_radial_direction_( 1 )
    {
        const int num_layers = num_uniform_layers;
        if ( num_layers % num_subdomains_in_radial_direction_ != 0 )
        {
            throw std::invalid_argument(
                "Number of layers must be divisible by number of subdomains in radial direction." );
        }
    }

    int global_lateral_refinement_level() const { return global_lateral_refinement_level_; }

    const std::vector< double >& radii() const { return radii_; }

    int num_subdomains_per_diamond_side() const { return 1 << diamond_subdomain_refinement_level_; }

    int num_subdomains_in_radial_direction() const { return num_subdomains_in_radial_direction_; }

    int subdomain_num_nodes_per_side_laterally() const
    {
        const int num_cells_per_diamond_side   = 1 << global_lateral_refinement_level();
        const int num_cells_per_subdomain_side = num_cells_per_diamond_side / num_subdomains_per_diamond_side();
        const int num_nodes_per_subdomain_side = num_cells_per_subdomain_side + 1;
        return num_nodes_per_subdomain_side;
    }

    int subdomain_num_nodes_radially() const
    {
        const int num_layers               = radii_.size() - 1;
        const int num_layers_per_subdomain = num_layers / num_subdomains_in_radial_direction_;
        return num_layers_per_subdomain + 1;
    }

    std::vector< SubdomainInfo > all_subdomains() const
    {
        std::vector< SubdomainInfo > subdomains;
        for ( int diamond_id = 0; diamond_id < 10; diamond_id++ )
        {
            for ( int x = 0; x < num_subdomains_per_diamond_side(); x++ )
            {
                for ( int y = 0; y < num_subdomains_per_diamond_side(); y++ )
                {
                    for ( int r = 0; r < num_subdomains_in_radial_direction_; r++ )
                    {
                        SubdomainInfo subdomain( diamond_id, x, y, r );
                        subdomains.push_back( subdomain );
                    }
                }
            }
        }
        return subdomains;
    }

  private:
    /// Number of times each diamond is refined laterally in each direction.
    int global_lateral_refinement_level_;

    /// Shell radii.
    std::vector< double > radii_;

    /// Number of subdomain partitioning steps (for parallel partitioning) in each direction of the diamond (at least 0).
    int diamond_subdomain_refinement_level_;

    /// Number of subdomains (for parallel partitioning) in the radial direction (at least 1).
    int num_subdomains_in_radial_direction_;
};

Grid3DDataVec< double, 3 > subdomain_unit_sphere_single_shell_coords(
    const DomainInfo&                   domain_info,
    const std::vector< SubdomainInfo >& subdomain_infos );

Grid2DDataScalar< double >
    subdomain_shell_radii( const DomainInfo& domain_info, const std::vector< SubdomainInfo >& subdomain_infos );

KOKKOS_INLINE_FUNCTION dense::Vec< double, 3 > coords(
    const int                         subdomain,
    const int                         x,
    const int                         y,
    const int                         r,
    const Grid3DDataVec< double, 3 >& subdomain_unit_sphere_coords,
    const Grid2DDataScalar< double >& subdomain_shell_radii )
{
    dense::Vec< double, 3 > coords;
    coords( 0 ) = subdomain_unit_sphere_coords( subdomain, x, y, 0 );
    coords( 1 ) = subdomain_unit_sphere_coords( subdomain, x, y, 1 );
    coords( 2 ) = subdomain_unit_sphere_coords( subdomain, x, y, 2 );
    return coords * subdomain_shell_radii( subdomain, r );
}

} // namespace terra::grid::shell
