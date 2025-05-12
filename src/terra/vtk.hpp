#pragma once

namespace terra {

#include <Kokkos_Core.hpp>
#include <cmath>     // For std::floor
#include <fstream>   // For file output (std::ofstream)
#include <iomanip>   // For std::fixed, std::setprecision
#include <stdexcept> // For error handling (std::runtime_error)
#include <string>    // For filenames (std::string)
#include <vector>    // Can be useful for intermediate storage if needed

// Define VTK cell type IDs for clarity
constexpr int VTK_QUAD           = 9;
constexpr int VTK_QUADRATIC_QUAD = 23;

// Enum to specify the desired element type
enum class VtkElementType
{
    LINEAR_QUAD,
    QUADRATIC_QUAD
};

/**
 * @brief Writes a 2D grid of vertices stored in a Kokkos View to a VTK XML
 *        Unstructured Grid file (.vtu) representing a quadrilateral mesh
 *        (linear or quadratic).
 *
 * @param filename The path to the output VTK file (.vtu).
 * @param vertices A Kokkos View containing the vertex coordinates.
 *                 Assumed dimensions: (Nx, Ny, 3).
 *                 vertices(i, j, 0) = X coordinate of point (i, j)
 *                 vertices(i, j, 1) = Y coordinate of point (i, j)
 *                 vertices(i, j, 2) = Z coordinate of point (i, j)
 *                 Nx = vertices.extent(0), Ny = vertices.extent(1)
 *                 For QUADRATIC_QUAD, Nx and Ny must be odd and >= 3.
 * @param elementType Specifies whether to write linear or quadratic elements.
 */
void write_vtk_xml_quad_mesh(
    const std::string&                  filename,
    const Kokkos::View< double** [3] >& vertices,
    VtkElementType                      elementType = VtkElementType::LINEAR_QUAD )
{
    // 1. Get Dimensions and Validate
    const size_t nx         = vertices.extent( 0 );
    const size_t ny         = vertices.extent( 1 );
    const size_t num_points = nx * ny;

    size_t  num_elements       = 0;
    size_t  points_per_element = 0;
    uint8_t vtk_cell_type_id   = 0; // Use uint8_t for VTK types array

    if ( elementType == VtkElementType::LINEAR_QUAD )
    {
        if ( nx < 2 || ny < 2 )
        {
            throw std::runtime_error( "XML: Cannot create linear quads from a mesh smaller than 2x2 points." );
        }
        num_elements       = ( nx - 1 ) * ( ny - 1 );
        points_per_element = 4;
        vtk_cell_type_id   = static_cast< uint8_t >( VTK_QUAD );
    }
    else if ( elementType == VtkElementType::QUADRATIC_QUAD )
    {
        if ( nx < 3 || ny < 3 )
        {
            throw std::runtime_error( "XML: Cannot create quadratic quads from a mesh smaller than 3x3 points." );
        }
        if ( nx % 2 == 0 || ny % 2 == 0 )
        {
            throw std::runtime_error(
                "XML: For QUADRATIC_QUAD elements using the 'every second node' scheme, Nx and Ny must be odd." );
        }
        size_t num_quad_elems_x = ( nx - 1 ) / 2;
        size_t num_quad_elems_y = ( ny - 1 ) / 2;
        num_elements            = num_quad_elems_x * num_quad_elems_y;
        points_per_element      = 8;
        vtk_cell_type_id        = static_cast< uint8_t >( VTK_QUADRATIC_QUAD );
    }
    else
    {
        throw std::runtime_error( "XML: Unsupported VtkElementType." );
    }

    if ( num_elements == 0 && num_points > 0 )
    {
        // Handle cases like 2x1 or 3x1 grids where no elements can be formed
        // Write points but zero cells
        std::cout << "Warning: Input dimensions result in zero elements. Writing points only." << std::endl;
    }
    else if ( num_elements == 0 && num_points == 0 )
    {
        throw std::runtime_error( "XML: Input dimensions result in zero points and zero elements." );
    }

    // 2. Ensure data is accessible on the Host
    auto h_vertices = Kokkos::create_mirror_view( vertices );
    Kokkos::deep_copy( h_vertices, vertices );
    Kokkos::fence();

    // 3. Open the output file stream
    std::ofstream ofs( filename );
    if ( !ofs.is_open() )
    {
        throw std::runtime_error( "XML: Could not open file for writing: " + filename );
    }
    ofs << std::fixed << std::setprecision( 8 ); // Precision for coordinates

    // --- 4. Write VTK XML Header ---
    ofs << "<?xml version=\"1.0\"?>\n";
    // Use Float64 for coordinates (double), Int64 for connectivity/offsets (safer for large meshes)
    ofs << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    ofs << "  <UnstructuredGrid>\n";
    // Piece contains the main mesh data. NumberOfPoints/Cells must be correct.
    ofs << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfCells=\"" << num_elements << "\">\n";

    // --- 5. Write Points ---
    ofs << "      <Points>\n";
    // DataArray for coordinates: Float64, 3 components (XYZ)
    ofs << "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for ( size_t i = 0; i < nx; ++i )
    {
        for ( size_t j = 0; j < ny; ++j )
        {
            ofs << "          " << h_vertices( i, j, 0 ) << " " << h_vertices( i, j, 1 ) << " " << h_vertices( i, j, 2 )
                << "\n";
        }
    }
    ofs << "        </DataArray>\n";
    ofs << "      </Points>\n";

    // --- 6. Write Cells (Connectivity, Offsets, Types) ---
    ofs << "      <Cells>\n";

    // 6.a. Connectivity Array (flat list of point indices for all cells)
    // Use Int64 for indices to be safe with large meshes (size_t can exceed Int32)
    std::vector< int64_t > connectivity;                       // Use std::vector temporarily or write directly
    connectivity.reserve( num_elements * points_per_element ); // Pre-allocate roughly

    // 6.b. Offsets Array (index in connectivity where each cell ENDS)
    std::vector< int64_t > offsets;
    offsets.reserve( num_elements );
    int64_t current_offset = 0; // VTK offsets are cumulative

    // 6.c. Types Array (VTK cell type ID for each cell)
    std::vector< uint8_t > types;
    types.reserve( num_elements );

    // --- Populate Connectivity, Offsets, and Types ---
    if ( elementType == VtkElementType::LINEAR_QUAD )
    {
        for ( size_t i = 0; i < nx - 1; ++i )
        {
            for ( size_t j = 0; j < ny - 1; ++j )
            {
                // Calculate the 0-based indices
                int64_t p0_idx = static_cast< int64_t >( i * ny + j );
                int64_t p1_idx = static_cast< int64_t >( ( i + 1 ) * ny + j );
                int64_t p2_idx = static_cast< int64_t >( ( i + 1 ) * ny + ( j + 1 ) );
                int64_t p3_idx = static_cast< int64_t >( i * ny + ( j + 1 ) );

                // Append connectivity
                connectivity.push_back( p0_idx );
                connectivity.push_back( p1_idx );
                connectivity.push_back( p2_idx );
                connectivity.push_back( p3_idx );

                // Update and append offset
                current_offset += points_per_element;
                offsets.push_back( current_offset );

                // Append type
                types.push_back( vtk_cell_type_id );
            }
        }
    }
    else
    { // QUADRATIC_QUAD
        for ( size_t i = 0; i < nx - 1; i += 2 )
        {
            for ( size_t j = 0; j < ny - 1; j += 2 )
            {
                // Calculate indices (casting to int64_t for the array)
                int64_t p0_idx  = static_cast< int64_t >( i * ny + j );
                int64_t p1_idx  = static_cast< int64_t >( ( i + 2 ) * ny + j );
                int64_t p2_idx  = static_cast< int64_t >( ( i + 2 ) * ny + ( j + 2 ) );
                int64_t p3_idx  = static_cast< int64_t >( i * ny + ( j + 2 ) );
                int64_t m01_idx = static_cast< int64_t >( ( i + 1 ) * ny + j );
                int64_t m12_idx = static_cast< int64_t >( ( i + 2 ) * ny + ( j + 1 ) );
                int64_t m23_idx = static_cast< int64_t >( ( i + 1 ) * ny + ( j + 2 ) );
                int64_t m30_idx = static_cast< int64_t >( i * ny + ( j + 1 ) );

                // Append connectivity (VTK order: corners then midsides)
                connectivity.push_back( p0_idx );
                connectivity.push_back( p1_idx );
                connectivity.push_back( p2_idx );
                connectivity.push_back( p3_idx );
                connectivity.push_back( m01_idx );
                connectivity.push_back( m12_idx );
                connectivity.push_back( m23_idx );
                connectivity.push_back( m30_idx );

                // Update and append offset
                current_offset += points_per_element;
                offsets.push_back( current_offset );

                // Append type
                types.push_back( vtk_cell_type_id );
            }
        }
    }

    // --- Write the populated arrays to the file ---
    // Connectivity
    ofs << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    ofs << "          "; // Indentation
    for ( size_t i = 0; i < connectivity.size(); ++i )
    {
        ofs << connectivity[i] << ( ( i + 1 ) % 12 == 0 ? "\n          " : " " ); // Newline every 12 values
    }
    ofs << "\n        </DataArray>\n"; // Add newline before closing tag if needed

    // Offsets
    ofs << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    ofs << "          ";
    for ( size_t i = 0; i < offsets.size(); ++i )
    {
        ofs << offsets[i] << ( ( i + 1 ) % 12 == 0 ? "\n          " : " " );
    }
    ofs << "\n        </DataArray>\n";

    // Types
    ofs << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    ofs << "          ";
    for ( size_t i = 0; i < types.size(); ++i )
    {
        // Need to cast uint8_t to int for printing as number, not char
        ofs << static_cast< int >( types[i] ) << ( ( i + 1 ) % 20 == 0 ? "\n          " : " " );
    }
    ofs << "\n        </DataArray>\n";

    ofs << "      </Cells>\n";

    // --- 7. Write Empty PointData and CellData (Good Practice) ---
    ofs << "      <PointData>\n";
    // Add <DataArray> tags here if you have data associated with points
    ofs << "      </PointData>\n";
    ofs << "      <CellData>\n";
    // Add <DataArray> tags here if you have data associated with cells
    ofs << "      </CellData>\n";

    // --- 8. Write VTK XML Footer ---
    ofs << "    </Piece>\n";
    ofs << "  </UnstructuredGrid>\n";
    ofs << "</VTKFile>\n";

    // 9. Close the file
    ofs.close();
    if ( !ofs )
    {
        throw std::runtime_error( "XML: Error occurred during writing or closing file: " + filename );
    }
}

} // namespace terra