#include <fstream> // For VTK output example
#include <iomanip>
#include <iostream>
#include <optional>

#include "../src/terra/grid/shell/spherical_shell.hpp"
#include "../src/terra/vtk/vtk.hpp"
#include "kernels/common/interpolation.hpp"
#include "kernels/common/vector_operations.hpp"
#include "terra/point_3d.hpp"

struct SomeInterpolator
{
    terra::grid::Grid3DDataVec< double, 3 > shell_coords_;
    terra::grid::Grid2DDataScalar< double > radii_;
    terra::grid::Grid4DDataScalar< double > scalar_data_;

    SomeInterpolator(
        terra::grid::Grid3DDataVec< double, 3 > shell_coords,
        terra::grid::Grid2DDataScalar< double > radii,
        terra::grid::Grid4DDataScalar< double > scalar_data )
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
    Kokkos::initialize( argc, argv );
    {
        const int    lateral_refinement_level = 0;
        const double r_min                    = 0.5;
        const double r_max                    = 1.0;
        const int    num_shells               = 5;
        const int    num_layers               = num_shells - 1;

        terra::grid::shell::DomainInfo domain_info( lateral_refinement_level, r_min, r_max, num_layers );
        const auto                     subdomain_infos = domain_info.all_subdomains();

        const auto subdomain_shell_coords =
            terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain_info, subdomain_infos );
        const auto subdomain_radii = terra::grid::shell::subdomain_shell_radii( domain_info, subdomain_infos );

        terra::vtk::VTKOutput vtk( subdomain_shell_coords, subdomain_radii, "my_fancy_vtk.vtu", true );

        terra::grid::Grid4DDataScalar< double > data(
            "scalar_data",
            subdomain_infos.size(),
            domain_info.subdomain_num_nodes_per_side_laterally(),
            domain_info.subdomain_num_nodes_per_side_laterally(),
            domain_info.subdomain_num_nodes_radially() );

        Kokkos::parallel_for(
            "some_interpolation",
            Kokkos::MDRangePolicy(
                { 0, 0, 0, 0 }, { data.extent( 0 ), data.extent( 1 ), data.extent( 2 ), data.extent( 3 ) } ),
            SomeInterpolator( subdomain_shell_coords, subdomain_radii, data ) );

        vtk.add_scalar_field( data.label(), data );

        vtk.write();
    }
    Kokkos::finalize();

    return 0;
}