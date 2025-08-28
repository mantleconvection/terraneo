
#pragma once

#include <fstream>

#include "mpi/mpi.hpp"
#include "util/xml.hpp"

namespace terra::visualization {

/// @brief XDMF output for visualization with software like Paraview.
///
/// Interprets data as block-structured wedge-element meshes (like the thick spherical shell is).
///
/// The mesh data has to be added upon construction.
/// None, one, or many scalar or vector-valued grids can be added.
///
/// Each write() call then writes out a corresponding .xmf file to the specified directory with an increasing index.
/// So for time-dependent runs, call write() in, say, every timestep.
///
/// The mesh is only written out ONCE during the first write() call and then re-used (referenced) by all files produced
/// by later write calls.
///
/// All added data grids must have different (Kokkos::View-)labels. Otherwise, the output will be corrupted.
///
/// The actually written data type can be selected regardless of the underlying data type of the allocated data for the
/// mesh points, topology, and each data grid. Defaults are selected via default parameters.
template < typename InputGridScalarType >
class XDMFOutput
{
  public:
    /// @brief Used to specify the output type when writing floating point data.
    ///
    /// Values are the number of bytes.
    enum class OutputTypeFloat : int
    {
        Float32 = 4,
        Float64 = 8,
    };

    /// @brief Used to specify the output type when writing (signed) integer data.
    ///
    /// Values are the number of bytes.
    enum class OutputTypeInt : int
    {
        Int32 = 4,
        Int64 = 8,
    };

    /// @brief Constructs an XDMFOutput object.
    ///
    /// All data will be written to the specified directory (it is a good idea to pass an empty directory).
    ///
    /// Does not write any data, yet.
    XDMFOutput(
        const std::string&                                   directory_path,
        const grid::Grid3DDataVec< InputGridScalarType, 3 >& coords_shell_device,
        const grid::Grid2DDataScalar< InputGridScalarType >& coords_radii_device,
        const OutputTypeFloat                                output_type_points       = OutputTypeFloat::Float32,
        const OutputTypeInt                                  output_type_connectivity = OutputTypeInt::Int64 )
    : directory_path_( directory_path )
    , coords_shell_device_( coords_shell_device )
    , coords_radii_device_( coords_radii_device )
    , output_type_points_( output_type_points )
    , output_type_connectivity_( output_type_connectivity )
    {
        if ( coords_shell_device.extent( 0 ) != coords_radii_device.extent( 0 ) )
        {
            Kokkos::abort( "XDMF: Number of subdomains of shell and radii coords does not match." );
        }
    }

    /// @brief Adds a new data grid to be written out.
    ///
    /// Does not write any data to file.
    template < typename InputScalarDataType >
    void
        add( const grid::Grid4DDataScalar< InputScalarDataType >& data,
             const OutputTypeFloat                                output_type = OutputTypeFloat::Float32 )
    {
        check_extents( data );
        if constexpr ( std::is_same_v< InputScalarDataType, double > )
        {
            device_data_views_scalar_double_.push_back( { data, output_type } );
        }
        else if constexpr ( std::is_same_v< InputScalarDataType, float > )
        {
            device_data_views_scalar_float_.push_back( { data, output_type } );
        }
        else
        {
            Kokkos::abort( "XDMF::add(): Grid data type not supported (yet)." );
        }
    }

    /// @brief Writes a "time step".
    ///
    /// Will write one .xmf file with the current counter as part of the name such that the files can be opened as a
    /// time series.
    ///
    /// The first write() call after construction will also write the mesh data (binary files) that will be referenced
    /// from later .xmf files.
    ///
    /// For each added data grid, one additional binary file is written. The data is copied to the host if required.
    /// The write() calls will allocate temporary storage on the host if host and device memory are not shared.
    /// Currently, for data grids, some host-side temporary buffers are kept (the sizes depend on the type of data
    /// added) to avoid frequent reallocation.
    void write()
    {
        using util::XML;

        const auto geometry_file_base = "geometry.bin";
        const auto topology_file_base = "topology.bin";

        const auto geometry_file_path = directory_path_ + "/" + geometry_file_base;
        const auto topology_file_path = directory_path_ + "/" + topology_file_base;

        const auto step_file_path = directory_path_ + "/step_" + std::to_string( write_counter_ ) + ".xmf";

        const int num_subdomains = coords_shell_device_.extent( 0 );
        const int nodes_x        = coords_shell_device_.extent( 1 );
        const int nodes_y        = coords_shell_device_.extent( 2 );
        const int nodes_r        = coords_radii_device_.extent( 1 );

        const auto number_of_nodes_local    = num_subdomains * nodes_x * nodes_y * nodes_r;
        const auto number_of_elements_local = num_subdomains * ( nodes_x - 1 ) * ( nodes_y - 1 ) * ( nodes_r - 1 ) * 2;

        if ( write_counter_ == 0 )
        {
            // Number of global nodes and elements.

            int num_nodes_elements_global[2] = { number_of_nodes_local, number_of_elements_local };
            MPI_Allreduce( MPI_IN_PLACE, &num_nodes_elements_global, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
            number_of_nodes_global_    = num_nodes_elements_global[0];
            number_of_elements_global_ = num_nodes_elements_global[1];

            // Check MPI write offset

            // To be populated:
            // First entry: number of nodes of processes before this
            // Second entry: number of elements of processes before this
            int offsets[2];

            int local_values[2] = { number_of_nodes_local, number_of_elements_local };

            // Compute the prefix sum (inclusive)
            MPI_Scan( &local_values, &offsets, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

            // Subtract the local value to get the sum of all values from processes with ranks < current rank
            number_of_nodes_offset_    = offsets[0] - local_values[0];
            number_of_elements_offset_ = offsets[1] - local_values[1];

            // Create directory on root

            // TODO

            // Add a README to the directory (what to keep, what the data contains, some notes on how to use paraview).

            // TODO

            // Write mesh binary data if first write

            // Node points.
            {
                std::stringstream geometry_stream;
                switch ( output_type_points_ )
                {
                case OutputTypeFloat::Float32:
                    write_geometry_binary_data< float >( geometry_stream );
                    break;
                case OutputTypeFloat::Float64:
                    write_geometry_binary_data< double >( geometry_stream );
                    break;
                default:
                    Kokkos::abort( "XDMF: Unknown output type for geometry." );
                }

                MPI_File fh;
                MPI_File_open(
                    MPI_COMM_WORLD, geometry_file_path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh );

                // Define the file view: each process writes its local data sequentially
                MPI_Offset disp = number_of_nodes_offset_ * 3 * static_cast< int >( output_type_points_ );
                MPI_File_set_view( fh, disp, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL );

                std::string geom_str = geometry_stream.str();

                // Write data collectively
                MPI_File_write_all(
                    fh, geom_str.data(), static_cast< int >( geom_str.size() ), MPI_CHAR, MPI_STATUS_IGNORE );

                // Close the file
                MPI_File_close( &fh );
            }

            // Connectivity/topology/elements (whatever you want to call it).
            {
                std::stringstream topology_stream;
                switch ( output_type_connectivity_ )
                {
                case OutputTypeInt::Int32:
                    write_topology_binary_data< int32_t >( topology_stream, number_of_nodes_offset_ );
                    break;
                case OutputTypeInt::Int64:
                    write_topology_binary_data< int64_t >( topology_stream, number_of_nodes_offset_ );
                    break;
                default:
                    Kokkos::abort( "XDMF: Unknown output type for topology." );
                }

                MPI_File fh;
                MPI_File_open(
                    MPI_COMM_WORLD, topology_file_path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh );

                // Define the file view: each process writes its local data sequentially
                MPI_Offset disp = 6 * number_of_elements_offset_ * static_cast< int >( output_type_connectivity_ );
                MPI_File_set_view( fh, disp, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL );

                std::string topo_str = topology_stream.str();

                // Write data collectively
                MPI_File_write_all(
                    fh, topo_str.data(), static_cast< int >( topo_str.size() ), MPI_CHAR, MPI_STATUS_IGNORE );

                // Close the file
                MPI_File_close( &fh );
            }
        }

        // Build XML skeleton.

        auto xml    = XML( "Xdmf", { { "Version", "2.0" } } );
        auto domain = XML( "Domain" );
        auto grid   = XML( "Grid", { { "Name", "Grid" }, { "GridType", "Uniform" } } );

        auto geometry =
            XML( "Geometry", { { "Type", "XYZ" } } )
                .add_child(
                    XML( "DataItem",
                         { { "Format", "Binary" },
                           { "DataType", "Float" },
                           { "Precision", std::to_string( static_cast< int >( output_type_points_ ) ) },
                           { "Endian", "Little" },
                           { "Dimensions", std::to_string( number_of_nodes_global_ ) + " " + std::to_string( 3 ) } },
                         geometry_file_base ) );

        grid.add_child( geometry );

        auto topology =
            XML( "Topology",
                 { { "Type", "Wedge" }, { "NumberOfElements", std::to_string( number_of_elements_global_ ) } } )
                .add_child(
                    XML( "DataItem",
                         { { "Format", "Binary" },
                           { "DataType", "Int" },
                           { "Precision", std::to_string( static_cast< int >( output_type_connectivity_ ) ) },
                           { "Endian", "Little" },
                           { "Dimensions", std::to_string( number_of_elements_global_ * 6 ) } },
                         topology_file_base ) );

        grid.add_child( topology );

        // Write attribute data for each attached grid.

        for ( const auto& [data, output_type] : device_data_views_scalar_float_ )
        {
            const auto attribute = write_scalar_attribute_file( data, output_type );
            grid.add_child( attribute );
        }

        for ( const auto& [data, output_type] : device_data_views_scalar_double_ )
        {
            const auto attribute = write_scalar_attribute_file( data, output_type );
            grid.add_child( attribute );
        }

        domain.add_child( grid );
        xml.add_child( domain );

        if ( mpi::rank() == 0 )
        {
            std::ofstream step_stream( step_file_path );
            step_stream << "<?xml version=\"1.0\" ?>\n";
            step_stream << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
            step_stream << xml.to_string();
            step_stream.close();
        }

        write_counter_++;
    }

  private:
    template < typename GridDataType >
    void check_extents( const GridDataType& data )
    {
        if ( data.extent( 0 ) != coords_radii_device_.extent( 0 ) )
        {
            Kokkos::abort( "XDMF: Number of subdomains of added data item does not match mesh." );
        }

        if ( data.extent( 1 ) != coords_shell_device_.extent( 1 ) )
        {
            Kokkos::abort( "XDMF: Dim x of added data item does not match mesh." );
        }

        if ( data.extent( 2 ) != coords_shell_device_.extent( 2 ) )
        {
            Kokkos::abort( "XDMF: Dim y of added data item does not match mesh." );
        }

        if ( data.extent( 3 ) != coords_radii_device_.extent( 1 ) )
        {
            Kokkos::abort( "XDMF: Dim r of added data item does not match mesh." );
        }
    }

    template < std::floating_point FloatingPointOutputType >
    void write_geometry_binary_data( std::stringstream& out )
    {
        // Copy mesh to host.
        // We assume the mesh is only written once so we throw away the host copies after this method returns.
        const typename grid::Grid3DDataVec< InputGridScalarType, 3 >::HostMirror coords_shell_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_shell_device_ );
        const typename grid::Grid2DDataScalar< InputGridScalarType >::HostMirror coords_radii_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, coords_radii_device_ );

        for ( int local_subdomain_id = 0; local_subdomain_id < coords_shell_host.extent( 0 ); local_subdomain_id++ )
        {
            for ( int r = 0; r < coords_radii_host.extent( 1 ); r++ )
            {
                for ( int y = 0; y < coords_shell_host.extent( 2 ); y++ )
                {
                    for ( int x = 0; x < coords_shell_host.extent( 1 ); x++ )
                    {
                        const auto c =
                            grid::shell::coords( local_subdomain_id, x, y, r, coords_shell_host, coords_radii_host );

                        for ( int d = 0; d < 3; d++ )
                        {
                            const auto cd = static_cast< FloatingPointOutputType >( c( d ) );
                            out.write( reinterpret_cast< const char* >( &cd ), sizeof( FloatingPointOutputType ) );
                        }
                    }
                }
            }
        }
    }

    template < std::integral IntegerOutputType >
    void write_topology_binary_data( std::stringstream& out, IntegerOutputType number_of_nodes_offset )
    {
        const int num_subdomains = coords_shell_device_.extent( 0 );
        const int nodes_x        = coords_shell_device_.extent( 1 );
        const int nodes_y        = coords_shell_device_.extent( 2 );
        const int nodes_r        = coords_radii_device_.extent( 1 );

        const int stride_0 = nodes_x * nodes_y * nodes_r;
        const int stride_1 = nodes_x * nodes_y;
        const int stride_2 = nodes_x;

        for ( int local_subdomain_id = 0; local_subdomain_id < num_subdomains; local_subdomain_id++ )
        {
            for ( int r = 0; r < nodes_r - 1; r++ )
            {
                for ( int y = 0; y < nodes_y - 1; y++ )
                {
                    for ( int x = 0; x < nodes_x - 1; x++ )
                    {
                        // Hex nodes
                        IntegerOutputType v[8];

                        v[0] = number_of_nodes_offset + local_subdomain_id * stride_0 + r * stride_1 + y * stride_2 + x;
                        v[1] = v[0] + 1;
                        v[2] = v[0] + nodes_x;
                        v[3] = v[0] + nodes_x + 1;

                        v[4] = number_of_nodes_offset + local_subdomain_id * stride_0 + ( r + 1 ) * stride_1 +
                               y * stride_2 + x;
                        v[5] = v[4] + 1;
                        v[6] = v[4] + nodes_x;
                        v[7] = v[4] + nodes_x + 1;

                        IntegerOutputType wedge_0[6] = { v[0], v[1], v[2], v[4], v[5], v[6] };
                        IntegerOutputType wedge_1[6] = { v[3], v[2], v[1], v[7], v[6], v[5] };

                        out.write( reinterpret_cast< const char* >( wedge_0 ), sizeof( IntegerOutputType ) * 6 );
                        out.write( reinterpret_cast< const char* >( wedge_1 ), sizeof( IntegerOutputType ) * 6 );
                    }
                }
            }
        }
    }

    template < typename ScalarTypeIn, typename ScalarTypeOut >
    void write_scalar_attribute_binary_data(
        const grid::Grid4DDataScalar< ScalarTypeIn >& device_data,
        std::stringstream&                            out )
    {
        // Copy data to host.
        if constexpr ( std::is_same_v< ScalarTypeIn, double > )
        {
            if ( !host_data_mirror_scalar_double_.has_value() )
            {
                host_data_mirror_scalar_double_ = Kokkos::create_mirror_view( Kokkos::HostSpace{}, device_data );
            }

            Kokkos::deep_copy( host_data_mirror_scalar_double_.value(), device_data );

            const auto& host_data = host_data_mirror_scalar_double_.value();

            for ( int local_subdomain_id = 0; local_subdomain_id < host_data.extent( 0 ); local_subdomain_id++ )
            {
                for ( int r = 0; r < host_data.extent( 3 ); r++ )
                {
                    for ( int y = 0; y < host_data.extent( 2 ); y++ )
                    {
                        for ( int x = 0; x < host_data.extent( 1 ); x++ )
                        {
                            const auto value = static_cast< ScalarTypeOut >( host_data( local_subdomain_id, x, y, r ) );
                            out.write( reinterpret_cast< const char* >( &value ), sizeof( ScalarTypeOut ) );
                        }
                    }
                }
            }
        }
        else if constexpr ( std::is_same_v< ScalarTypeIn, float > )
        {
            if ( !host_data_mirror_scalar_float_.has_value() )
            {
                host_data_mirror_scalar_float_ = Kokkos::create_mirror_view( Kokkos::HostSpace{}, device_data );
            }

            Kokkos::deep_copy( host_data_mirror_scalar_float_.value(), device_data );

            const auto& host_data = host_data_mirror_scalar_float_.value();

            for ( int local_subdomain_id = 0; local_subdomain_id < host_data.extent( 0 ); local_subdomain_id++ )
            {
                for ( int r = 0; r < host_data.extent( 3 ); r++ )
                {
                    for ( int y = 0; y < host_data.extent( 2 ); y++ )
                    {
                        for ( int x = 0; x < host_data.extent( 1 ); x++ )
                        {
                            const auto value = static_cast< ScalarTypeOut >( host_data( local_subdomain_id, x, y, r ) );
                            out.write( reinterpret_cast< const char* >( &value ), sizeof( ScalarTypeOut ) );
                        }
                    }
                }
            }
        }
        else
        {
            Kokkos::abort( "XDMF: Only double precision grids supported for scalar attributes." );
        }
    }

    template < typename GridData >
    util::XML write_scalar_attribute_file( const GridData& data, const OutputTypeFloat& output_type )
    {
        using InputType = typename GridData::value_type;

        const auto attribute_file_base = data.label() + "_" + std::to_string( write_counter_ ) + ".bin";
        const auto attribute_file_path = directory_path_ + "/" + attribute_file_base;

        {
            std::stringstream attribute_stream;
            switch ( output_type )
            {
            case OutputTypeFloat::Float32:
                write_scalar_attribute_binary_data< InputType, float >( data, attribute_stream );
                break;
            case OutputTypeFloat::Float64:
                write_scalar_attribute_binary_data< InputType, double >( data, attribute_stream );
                break;
            }

            MPI_File fh;
            MPI_File_open(
                MPI_COMM_WORLD, attribute_file_path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh );

            // Define the file view: each process writes its local data sequentially
            MPI_Offset disp = number_of_nodes_offset_ * static_cast< int >( output_type_points_ );
            MPI_File_set_view( fh, disp, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL );

            std::string attr_str = attribute_stream.str();

            // Write data collectively
            MPI_File_write_all(
                fh, attr_str.data(), static_cast< int >( attr_str.size() ), MPI_CHAR, MPI_STATUS_IGNORE );

            // Close the file
            MPI_File_close( &fh );
        }

        auto attribute =
            util::XML( "Attribute", { { "Name", data.label() }, { "AttributeType", "Scalar" }, { "Center", "Node" } } )
                .add_child(
                    util::XML(
                        "DataItem",
                        { { "Format", "Binary" },
                          { "DataType", "Float" },
                          { "Precision", std::to_string( static_cast< int >( output_type ) ) },
                          { "Endian", "Little" },
                          { "Dimensions", std::to_string( number_of_nodes_global_ ) } },
                        attribute_file_base ) );

        return attribute;
    }

    std::string directory_path_;

    grid::Grid3DDataVec< InputGridScalarType, 3 > coords_shell_device_;
    grid::Grid2DDataScalar< InputGridScalarType > coords_radii_device_;

    OutputTypeFloat output_type_points_;
    OutputTypeInt   output_type_connectivity_;

    // Collecting all views that are written on every write call.
    std::vector< std::pair< grid::Grid4DDataScalar< double >, OutputTypeFloat > > device_data_views_scalar_double_;
    std::vector< std::pair< grid::Grid4DDataScalar< float >, OutputTypeFloat > >  device_data_views_scalar_float_;

    // Just a single mirror for buffering during write.
    std::optional< grid::Grid4DDataScalar< double >::HostMirror > host_data_mirror_scalar_double_;
    std::optional< grid::Grid4DDataScalar< float >::HostMirror >  host_data_mirror_scalar_float_;

    int write_counter_ = 0;

    int number_of_nodes_offset_    = -1;
    int number_of_elements_offset_ = -1;

    int number_of_nodes_global_    = -1;
    int number_of_elements_global_ = -1;
};

} // namespace terra::visualization