#include <fstream> // For VTK output example
#include <iomanip>
#include <iostream>

#include "terra/point_3d.hpp"
#include "terra/spherical_shell_mesh.hpp"
#include "terra/vtk.hpp"
#include "vtk.hpp"

int main( int argc, char** argv )
{
    Kokkos::initialize( argc, argv );
    {
        const int global_refinements = 8;
        const int n_subdomains       = 4;

        for ( int i = 0; i < 1; ++i )
        {
            auto mesh_view_full      = terra::setup_coords_subdomain( i, global_refinements, 1, 0, 0 );
            auto mesh_view_partial   = terra::setup_coords_subdomain( i, global_refinements, n_subdomains, 0, 0 );
            auto mesh_view_partial_2 = terra::setup_coords_subdomain( i, global_refinements, n_subdomains, 1, 0 );
            auto mesh_view_partial_3 = terra::setup_coords_subdomain( i, global_refinements, n_subdomains, 0, 1 );
            auto mesh_view_partial_4 = terra::setup_coords_subdomain( i, global_refinements, n_subdomains, 1, 1 );
            terra::write_vtk_xml_quad_mesh(
                "mesh_view_full_" + std::to_string( i ) + ".vtu",
                mesh_view_full,
                terra::VtkElementType::QUADRATIC_QUAD );
            terra::write_vtk_xml_quad_mesh(
                "mesh_view_partial_" + std::to_string( i ) + ".vtu",
                mesh_view_partial,
                terra::VtkElementType::QUADRATIC_QUAD );
            terra::write_vtk_xml_quad_mesh(
                "mesh_view_partial_2_" + std::to_string( i ) + ".vtu",
                mesh_view_partial_2,
                terra::VtkElementType::QUADRATIC_QUAD );
            terra::write_vtk_xml_quad_mesh(
                "mesh_view_partial_3_" + std::to_string( i ) + ".vtu",
                mesh_view_partial_3,
                terra::VtkElementType::QUADRATIC_QUAD );
            terra::write_vtk_xml_quad_mesh(
                "mesh_view_partial_4_" + std::to_string( i ) + ".vtu",
                mesh_view_partial_4,
                terra::VtkElementType::QUADRATIC_QUAD );
        }
    }
    Kokkos::finalize();

    return 0;
}