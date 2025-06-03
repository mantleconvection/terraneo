#include <fstream> // For VTK output example
#include <iomanip>
#include <iostream>
#include <optional>

#include "../src/terra/grid/shell/spherical_shell.hpp"
#include "../src/terra/vtk/vtk.hpp"
#include "terra/point_3d.hpp"

struct SubdomainInterpolator
{
    using PointType  = terra::real_t;
    using ScalarType = terra::real_t;

    using SubdomainShellCoordsType = terra::grid::Grid2DDataVec< PointType, 3 >;
    using ScalarGridDataType       = terra::grid::Grid3DDataVec< ScalarType, 3 >;

    terra::grid::ThickSphericalShellSubdomainGrid grid_;
    ScalarGridDataType                            grid_data_;

    SubdomainInterpolator(
        const terra::grid::ThickSphericalShellSubdomainGrid& grid,
        const ScalarGridDataType&                            grid_data )
    : grid_( grid )
    , grid_data_( grid_data )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int x, const int y, const int r ) const
    {
        grid_data_( x, y, r, 0 ) = grid_.unit_sphere_coords()( x, y, 0 );
        grid_data_( x, y, r, 1 ) = grid_.unit_sphere_coords()( x, y, 1 );
        grid_data_( x, y, r, 2 ) = grid_.shell_radii()( r );
    }
};

int main( int argc, char** argv )
{
    Kokkos::initialize( argc, argv );
    {
        const int global_refinements = 2;
        const int diamond_id         = 0;

        terra::grid::ThickSphericalShellSubdomainGrid grid(
            global_refinements, diamond_id, 1, 0, 0, { 0.5, 0.75, 1.0 } );

        Kokkos::MDRangePolicy rangePolicy( { 0, 0, 0 }, { grid.size_x(), grid.size_y(), grid.size_r() } );

        terra::grid::Grid3DDataVec< double, 3 > data( "data", grid.size_x(), grid.size_y(), grid.size_r() );

        Kokkos::parallel_for( "parfor", rangePolicy, SubdomainInterpolator( grid, data ) );

        terra::vtk::write_rectilinear_to_triangular_vtu(
            grid.unit_sphere_coords(), "mesh_fulllll.vtu", terra::vtk::DiagonalSplitType::BACKWARD_SLASH );

        terra::vtk::write_vtk_xml_quad_mesh( "quadds.vtu", grid.unit_sphere_coords() );

        terra::vtk::write_surface_radial_extruded_to_wedge_vtu(
            grid.unit_sphere_coords(),
            grid.shell_radii(),
            std::optional( data ),
            "data_name",
            "shell.vtu",
            terra::vtk::DiagonalSplitType::BACKWARD_SLASH );
    }
    Kokkos::finalize();

    return 0;
}