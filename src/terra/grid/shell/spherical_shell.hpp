#pragma once

#include <cmath>
#include <stdexcept>

#include "../grid_types.hpp"
#include "../terra/kokkos/kokkos_wrapper.hpp"
#include "dense/vec.hpp"
#include "mpi/mpi.hpp"

namespace terra::grid::shell {

std::vector< double > uniform_shell_radii( double r_min, double r_max, int num_shells );

class SubdomainInfo
{
  public:
    SubdomainInfo()
    : diamond_id_( -1 )
    , subdomain_x_( -1 )
    , subdomain_y_( -1 )
    , subdomain_r_( -1 )
    {}

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

    bool operator<( const SubdomainInfo& other ) const
    {
        return std::tie( diamond_id_, subdomain_r_, subdomain_y_, subdomain_x_ ) <
               std::tie( other.diamond_id_, other.subdomain_r_, other.subdomain_y_, other.subdomain_x_ );
    }

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

inline std::ostream& operator<<( std::ostream& os, const SubdomainInfo& si )
{
    os << "Diamond ID: " << si.diamond_id();
    return os;
}

class DomainInfo
{
  public:
    DomainInfo() = default;

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

class SubdomainNeighborhood
{
  public:
    using NeighborSubdomainTupleVertex = std::tuple< SubdomainInfo, BoundaryVertex, mpi::MPIRank >;
    using NeighborSubdomainTupleEdge   = std::tuple< SubdomainInfo, BoundaryEdge, mpi::MPIRank >;
    using NeighborSubdomainTupleFace   = std::tuple< SubdomainInfo, BoundaryFace, mpi::MPIRank >;

    SubdomainNeighborhood() = default;

    SubdomainNeighborhood( const DomainInfo& domain_info, const SubdomainInfo& subdomain_info )
    {
        setup_neighborhood( domain_info, subdomain_info );
    }

    const std::map< BoundaryVertex, std::vector< NeighborSubdomainTupleVertex > >& neighborhood_vertex() const
    {
        return neighborhood_vertex_;
    }

    const std::map< BoundaryEdge, std::vector< NeighborSubdomainTupleEdge > >& neighborhood_edge() const
    {
        return neighborhood_edge_;
    }

    const std::map< BoundaryFace, NeighborSubdomainTupleFace >& neighborhood_face() const { return neighborhood_face_; }

  private:
    void setup_neighborhood( const DomainInfo& domain_info, const SubdomainInfo& subdomain_info )
    {
        if ( domain_info.num_subdomains_per_diamond_side() != 1 ||
             domain_info.num_subdomains_in_radial_direction() != 1 )
        {
            throw std::logic_error( "Neighborhood setup only implemented for full diamonds." );
        }

        if ( mpi::num_processes() != 1 )
        {
            throw std::logic_error( "Parallel neighborhood setup not yet supported." );
        }

        // Setup faces.
        const int diamond_id = subdomain_info.diamond_id();

        // Node equivalences: part one - communication between diamonds at the same poles

        // d_0( 0, :, r ) = d_1( :, 0, r )
        // d_1( 0, :, r ) = d_2( :, 0, r )
        // d_2( 0, :, r ) = d_3( :, 0, r )
        // d_3( 0, :, r ) = d_4( :, 0, r )
        // d_4( 0, :, r ) = d_0( :, 0, r )

        // d_5( 0, :, r ) = d_6( :, 0, r )
        // d_6( 0, :, r ) = d_7( :, 0, r )
        // d_7( 0, :, r ) = d_8( :, 0, r )
        // d_8( 0, :, r ) = d_9( :, 0, r )
        // d_9( 0, :, r ) = d_5( :, 0, r )

        // Node equivalences: part two - communication between diamonds at different poles

        // d_0( :, end, r ) = d_5( end, :, r )
        // d_1( :, end, r ) = d_6( end, :, r )
        // d_2( :, end, r ) = d_7( end, :, r )
        // d_3( :, end, r ) = d_8( end, :, r )
        // d_4( :, end, r ) = d_9( end, :, r )

        // d_5( :, end, r ) = d_1( end, :, r )
        // d_6( :, end, r ) = d_2( end, :, r )
        // d_7( :, end, r ) = d_3( end, :, r )
        // d_8( :, end, r ) = d_4( end, :, r )
        // d_9( :, end, r ) = d_0( end, :, r )

        switch ( diamond_id )
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
            // Part I
            neighborhood_face_[BoundaryFace::F_0YR] = {
                SubdomainInfo( ( diamond_id + 1 ) % 5, 0, 0, 0 ), BoundaryFace::F_X0R, 0 };
            neighborhood_face_[BoundaryFace::F_X0R] = {
                SubdomainInfo( ( diamond_id + 4 ) % 5, 0, 0, 0 ), BoundaryFace::F_0YR, 0 };

            // Part II
            neighborhood_face_[BoundaryFace::F_X1R] = {
                SubdomainInfo( diamond_id + 5, 0, 0, 0 ), BoundaryFace::F_1YR, 0 };
            neighborhood_face_[BoundaryFace::F_1YR] = {
                SubdomainInfo( ( diamond_id + 4 ) % 5 + 5, 0, 0, 0 ), BoundaryFace::F_X1R, 0 };
            break;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
            // Part I
            neighborhood_face_[BoundaryFace::F_0YR] = {
                SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, 0, 0, 0 ), BoundaryFace::F_X0R, 0 };
            neighborhood_face_[BoundaryFace::F_X0R] = {
                SubdomainInfo( ( diamond_id - 1 ) % 5 + 5, 0, 0, 0 ), BoundaryFace::F_0YR, 0 };

            // Part II
            neighborhood_face_[BoundaryFace::F_X1R] = {
                SubdomainInfo( ( diamond_id - 4 ) % 5, 0, 0, 0 ), BoundaryFace::F_1YR, 0 };
            neighborhood_face_[BoundaryFace::F_1YR] = {
                SubdomainInfo( diamond_id - 5, 0, 0, 0 ), BoundaryFace::F_X1R, 0 };
            break;
        default:
            throw std::logic_error( "Invalid diamond id." );
        }

        // Now only the edges at the poles that are not already part of the faces remain.

        switch ( diamond_id )
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
            // North Pole.
            neighborhood_edge_[BoundaryEdge::E_00R] = {
                { SubdomainInfo( ( diamond_id + 2 ) % 5, 0, 0, 0 ), BoundaryEdge::E_00R, 0 },
                { SubdomainInfo( ( diamond_id + 3 ) % 5, 0, 0, 0 ), BoundaryEdge::E_00R, 0 } };
            break;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
            // South Pole.
            neighborhood_edge_[BoundaryEdge::E_00R] = {
                { SubdomainInfo( ( diamond_id + 2 ) % 5 + 5, 0, 0, 0 ), BoundaryEdge::E_00R, 0 },
                { SubdomainInfo( ( diamond_id + 3 ) % 5 + 5, 0, 0, 0 ), BoundaryEdge::E_00R, 0 } };
            break;
        default:
            throw std::logic_error( "Invalid diamond id." );
        }
    }

    std::map< BoundaryVertex, std::vector< NeighborSubdomainTupleVertex > > neighborhood_vertex_;
    std::map< BoundaryEdge, std::vector< NeighborSubdomainTupleEdge > >     neighborhood_edge_;
    std::map< BoundaryFace, NeighborSubdomainTupleFace >                    neighborhood_face_;
};

class DistributedDomain
{
  public:
    using LocalSubdomainIdx = int;

    static DistributedDomain create_uniform_single_subdomain(
        const int    lateral_refinement_level,
        const int    radial_refinement_level,
        const real_t r_min,
        const real_t r_max )
    {
        DistributedDomain domain;
        domain.domain_info_ = DomainInfo( lateral_refinement_level, r_min, r_max, 1 << radial_refinement_level );
        int idx             = 0;
        for ( const auto& subdomain : domain.domain_info_.all_subdomains() )
        {
            domain.subdomains_[subdomain] = { idx, SubdomainNeighborhood( domain.domain_info_, subdomain ) };
            idx++;
        }
        return domain;
    }

    const DomainInfo& domain_info() const { return domain_info_; }
    const std::map< SubdomainInfo, std::tuple< LocalSubdomainIdx, SubdomainNeighborhood > >& subdomains() const
    {
        return subdomains_;
    }

  private:
    DistributedDomain() = default;

    DomainInfo                                                                        domain_info_;
    std::map< SubdomainInfo, std::tuple< LocalSubdomainIdx, SubdomainNeighborhood > > subdomains_;
};

inline Grid4DDataScalar< double >
    allocate_scalar_grid( const std::string label, const DistributedDomain& distributed_domain )
{
    return Grid4DDataScalar< double >(
        label,
        distributed_domain.subdomains().size(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_radially() );
}

inline Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >
    local_domain_md_range_policy_nodes( const DistributedDomain& distributed_domain )
{
    return Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
        { 0, 0, 0, 0 },
        { static_cast< long long >( distributed_domain.subdomains().size() ),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
          distributed_domain.domain_info().subdomain_num_nodes_radially() } );
}

inline Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >
    local_domain_md_range_policy_cells( const DistributedDomain& distributed_domain )
{
    return Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
        { 0, 0, 0, 0 },
        { static_cast< long long >( distributed_domain.subdomains().size() ),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
          distributed_domain.domain_info().subdomain_num_nodes_radially() - 1 } );
}

Grid3DDataVec< double, 3 > subdomain_unit_sphere_single_shell_coords( const DistributedDomain& domain );

Grid2DDataScalar< double > subdomain_shell_radii( const DistributedDomain& domain );

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
