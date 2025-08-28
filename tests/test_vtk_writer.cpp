#include <fstream>
#include <iomanip>
#include <optional>

#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/visualization/vtk.hpp"
#include "util/init.hpp"

struct SomeInterpolator
{
    terra::grid::Grid3DDataVec< double, 3 > shell_coords_;
    terra::grid::Grid2DDataScalar< double > radii_;
    terra::grid::Grid4DDataScalar< double > scalar_data_;

    SomeInterpolator(
        const terra::grid::Grid3DDataVec< double, 3 >& shell_coords,
        const terra::grid::Grid2DDataScalar< double >& radii,
        const terra::grid::Grid4DDataScalar< double >& scalar_data )
    : shell_coords_( shell_coords )
    , radii_( radii )
    , scalar_data_( scalar_data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int subdomain, const int x, const int y, const int r ) const
    {
        const terra::dense::Vec< double, 3 > coords =
            terra::grid::shell::coords( subdomain, x, y, r, shell_coords_, radii_ );

        const double value = coords( 0 ) * Kokkos::sin( coords( 1 ) ) * Kokkos::cos( coords( 2 ) );

        scalar_data_( subdomain, x, y, r ) = value;
    }
};

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    constexpr int    lateral_refinement_level = 4;
    constexpr int    radial_refinement_level  = 4;
    constexpr double r_min                    = 0.5;
    constexpr double r_max                    = 1.0;

    const auto domain = terra::grid::shell::DistributedDomain::create_uniform_single_subdomain(
        lateral_refinement_level,
        radial_refinement_level,
        r_min,
        r_max,
        terra::grid::shell::subdomain_to_rank_distribute_full_diamonds );

    const auto subdomain_shell_coords = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto subdomain_radii        = terra::grid::shell::subdomain_shell_radii< double >( domain );

    auto data = terra::grid::shell::allocate_scalar_grid< double >( "scalar_data", domain );

    Kokkos::parallel_for(
        "some_interpolation",
        terra::grid::shell::local_domain_md_range_policy_nodes( domain ),
        SomeInterpolator( subdomain_shell_coords, subdomain_radii, data ) );

    terra::visualization::VTKOutput< double > vtk( subdomain_shell_coords, subdomain_radii, true );
    vtk.add_scalar_field( data );
    vtk.write( "my_fancy_vtk.vtu" );

    return 0;
}