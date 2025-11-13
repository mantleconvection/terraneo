
#pragma once

#include <fstream>

#include "mpi/mpi.hpp"
#include "util/filesystem.hpp"
#include "util/logging.hpp"
#include "util/result.hpp"
#include "util/xml.hpp"

namespace terra::io {

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
        const grid::shell::DistributedDomain&                distributed_domain,
        const grid::Grid3DDataVec< InputGridScalarType, 3 >& coords_shell_device,
        const grid::Grid2DDataScalar< InputGridScalarType >& coords_radii_device,
        const OutputTypeFloat                                output_type_points       = OutputTypeFloat::Float32,
        const OutputTypeInt                                  output_type_connectivity = OutputTypeInt::Int64 )
    : directory_path_( directory_path )
    , distributed_domain_( distributed_domain )
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

    /// @brief Adds a new scalar data grid to be written out.
    ///
    /// Does not write any data to file yet - call write() for writing the next time step.
    template < typename InputScalarDataType >
    void
        add( const grid::Grid4DDataScalar< InputScalarDataType >& data,
             const OutputTypeFloat                                output_type = OutputTypeFloat::Float32 )
    {
        if ( write_counter_ != 0 )
        {
            Kokkos::abort( "XDMF::add(): Cannot add data after write() has been called." );
        }

        check_extents( data );

        if ( is_label_taken( data.label() ) )
        {
            Kokkos::abort( ( "Cannot add data with label '" + data.label() +
                             "' - data with identical label has been added previously." )
                               .c_str() );
        }

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

    /// @brief Adds a new vector-valued data grid to be written out.
    ///
    /// Does not write any data to file yet - call write() for writing the next time step.
    template < typename InputScalarDataType, int VecDim >
    void
        add( const grid::Grid4DDataVec< InputScalarDataType, VecDim >& data,
             const OutputTypeFloat                                     output_type = OutputTypeFloat::Float32 )
    {
        if ( write_counter_ != 0 )
        {
            Kokkos::abort( "XDMF::add(): Cannot add data after write() has been called." );
        }

        check_extents( data );

        if ( is_label_taken( data.label() ) )
        {
            Kokkos::abort( ( "Cannot add data with label '" + data.label() +
                             "' - data with identical label has been added previously." )
                               .c_str() );
        }

        if constexpr ( std::is_same_v< InputScalarDataType, double > )
        {
            device_data_views_vec_double_.push_back( { data, output_type } );
        }
        else if constexpr ( std::is_same_v< InputScalarDataType, float > )
        {
            device_data_views_vec_float_.push_back( { data, output_type } );
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

            int num_nodes_elements_subdomains_global[3] = {
                number_of_nodes_local, number_of_elements_local, num_subdomains };
            MPI_Allreduce( MPI_IN_PLACE, &num_nodes_elements_subdomains_global, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
            number_of_nodes_global_      = num_nodes_elements_subdomains_global[0];
            number_of_elements_global_   = num_nodes_elements_subdomains_global[1];
            number_of_subdomains_global_ = num_nodes_elements_subdomains_global[2];

            // Check MPI write offset

            // To be populated:
            // First entry: number of nodes of processes before this
            // Second entry: number of elements of processes before this
            // Third entry: number of subdomains of processes before this
            int offsets[3];

            int local_values[3] = { number_of_nodes_local, number_of_elements_local, num_subdomains };

            // Compute the prefix sum (inclusive)
            MPI_Scan( &local_values, &offsets, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

            // Subtract the local value to get the sum of all values from processes with ranks < current rank
            number_of_nodes_offset_      = offsets[0] - local_values[0];
            number_of_elements_offset_   = offsets[1] - local_values[1];
            number_of_subdomains_offset_ = offsets[2] - local_values[2];

            // Create a directory on root

            util::prepare_empty_directory( directory_path_ );

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

            // Checkpoint metadata
            {
                write_checkpoint_metadata();
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

        for ( const auto& [data, output_type] : device_data_views_vec_float_ )
        {
            const auto attribute = write_vec_attribute_file( data, output_type );
            grid.add_child( attribute );
        }

        for ( const auto& [data, output_type] : device_data_views_vec_double_ )
        {
            const auto attribute = write_vec_attribute_file( data, output_type );
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

    bool is_label_taken( const std::string& label )
    {
        for ( auto [grid, _] : device_data_views_scalar_double_ )
        {
            if ( grid.label() == label )
            {
                return true;
            }
        }

        for ( auto [grid, _] : device_data_views_scalar_float_ )
        {
            if ( grid.label() == label )
            {
                return true;
            }
        }

        for ( auto [grid, _] : device_data_views_vec_double_ )
        {
            if ( grid.label() == label )
            {
                return true;
            }
        }

        for ( auto [grid, _] : device_data_views_vec_float_ )
        {
            if ( grid.label() == label )
            {
                return true;
            }
        }

        return false;
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

    template < typename ScalarTypeIn >
    util::XML write_scalar_attribute_file(
        const grid::Grid4DDataScalar< ScalarTypeIn >& data,
        const OutputTypeFloat&                        output_type )
    {
        const auto attribute_file_base = data.label() + "_" + std::to_string( write_counter_ ) + ".bin";
        const auto attribute_file_path = directory_path_ + "/" + attribute_file_base;

        {
            std::stringstream attribute_stream;
            switch ( output_type )
            {
            case OutputTypeFloat::Float32:
                write_scalar_attribute_binary_data< ScalarTypeIn, float >( data, attribute_stream );
                break;
            case OutputTypeFloat::Float64:
                write_scalar_attribute_binary_data< ScalarTypeIn, double >( data, attribute_stream );
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

    template < typename ScalarTypeIn, typename ScalarTypeOut, int VecDim >
    void write_vec_attribute_binary_data(
        const grid::Grid4DDataVec< ScalarTypeIn, VecDim >& device_data,
        std::stringstream&                                 out )
    {
        // Copy data to host.
        if constexpr ( std::is_same_v< ScalarTypeIn, double > )
        {
            if ( !host_data_mirror_vec_double_.has_value() )
            {
                host_data_mirror_vec_double_ = Kokkos::create_mirror_view( Kokkos::HostSpace{}, device_data );
            }

            Kokkos::deep_copy( host_data_mirror_vec_double_.value(), device_data );

            const auto& host_data = host_data_mirror_vec_double_.value();

            for ( int local_subdomain_id = 0; local_subdomain_id < host_data.extent( 0 ); local_subdomain_id++ )
            {
                for ( int r = 0; r < host_data.extent( 3 ); r++ )
                {
                    for ( int y = 0; y < host_data.extent( 2 ); y++ )
                    {
                        for ( int x = 0; x < host_data.extent( 1 ); x++ )
                        {
                            for ( int d = 0; d < VecDim; d++ )
                            {
                                const auto value =
                                    static_cast< ScalarTypeOut >( host_data( local_subdomain_id, x, y, r, d ) );
                                out.write( reinterpret_cast< const char* >( &value ), sizeof( ScalarTypeOut ) );
                            }
                        }
                    }
                }
            }
        }
        else if constexpr ( std::is_same_v< ScalarTypeIn, float > )
        {
            if ( !host_data_mirror_vec_float_.has_value() )
            {
                host_data_mirror_vec_float_ = Kokkos::create_mirror_view( Kokkos::HostSpace{}, device_data );
            }

            Kokkos::deep_copy( host_data_mirror_vec_float_.value(), device_data );

            const auto& host_data = host_data_mirror_vec_float_.value();

            for ( int local_subdomain_id = 0; local_subdomain_id < host_data.extent( 0 ); local_subdomain_id++ )
            {
                for ( int r = 0; r < host_data.extent( 3 ); r++ )
                {
                    for ( int y = 0; y < host_data.extent( 2 ); y++ )
                    {
                        for ( int x = 0; x < host_data.extent( 1 ); x++ )
                        {
                            for ( int d = 0; d < VecDim; d++ )
                            {
                                const auto value =
                                    static_cast< ScalarTypeOut >( host_data( local_subdomain_id, x, y, r, d ) );
                                out.write( reinterpret_cast< const char* >( &value ), sizeof( ScalarTypeOut ) );
                            }
                        }
                    }
                }
            }
        }
        else
        {
            Kokkos::abort( "XDMF: Only double precision grids supported for vector-valued attributes." );
        }
    }

    template < typename ScalarTypeIn, int VecDim >
    util::XML write_vec_attribute_file(
        const grid::Grid4DDataVec< ScalarTypeIn, VecDim >& data,
        const OutputTypeFloat&                             output_type )
    {
        const auto attribute_file_base = data.label() + "_" + std::to_string( write_counter_ ) + ".bin";
        const auto attribute_file_path = directory_path_ + "/" + attribute_file_base;

        {
            std::stringstream attribute_stream;
            switch ( output_type )
            {
            case OutputTypeFloat::Float32:
                write_vec_attribute_binary_data< ScalarTypeIn, float >( data, attribute_stream );
                break;
            case OutputTypeFloat::Float64:
                write_vec_attribute_binary_data< ScalarTypeIn, double >( data, attribute_stream );
                break;
            }

            MPI_File fh;
            MPI_File_open(
                MPI_COMM_WORLD, attribute_file_path.c_str(), MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh );

            // Define the file view: each process writes its local data sequentially
            MPI_Offset disp = VecDim * number_of_nodes_offset_ * static_cast< int >( output_type_points_ );
            MPI_File_set_view( fh, disp, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL );

            std::string attr_str = attribute_stream.str();

            // Write data collectively
            MPI_File_write_all(
                fh, attr_str.data(), static_cast< int >( attr_str.size() ), MPI_CHAR, MPI_STATUS_IGNORE );

            // Close the file
            MPI_File_close( &fh );
        }

        auto attribute =
            util::XML( "Attribute", { { "Name", data.label() }, { "AttributeType", "Vector" }, { "Center", "Node" } } )
                .add_child(
                    util::XML(
                        "DataItem",
                        { { "Format", "Binary" },
                          { "DataType", "Float" },
                          { "Precision", std::to_string( static_cast< int >( output_type ) ) },
                          { "Endian", "Little" },
                          { "Dimensions",
                            std::to_string( number_of_nodes_global_ ) + " " + std::to_string( VecDim ) } },
                        attribute_file_base ) );

        return attribute;
    }

    /// Format (required once per series):
    ///
    /// @code
    /// version:                                      i32
    /// num_subdomains_per_diamond_lateral_direction: i32
    /// num_subdomains_per_diamond_radial_direction:  i32
    /// subdomain_size_x:                             i32
    /// subdomain_size_y:                             i32
    /// subdomain_size_r:                             i32
    /// radii:                                        array: f64, entries: num_subdomains_per_diamond_radial_direction * (subdomain_size_r - 1) + 1
    /// num_grid_data_files:                          i32
    /// list (size = num_grid_data_files)
    /// [
    ///     grid_name_string_size_bytes:              i32
    ///     grid_name_string:                         grid_name_string_size_bytes * 8
    ///     scalar_data_type:                         i32 // 0: signed integer, 1: unsigned integer, 2: float
    ///     scalar_bytes:                             i32
    ///     vec_dim:                                  i32
    /// ]
    /// checkpoint_subdomain_ordering:                i32
    /// if (checkpoint_subdomain_ordering == 0) {
    ///     // The following list contains the encoded global_subdomain_ids (as i64) in the order in which the
    ///     // subdomains are written to the data files. To find out where a certain subdomain is located in the
    ///     // file, the starting offset must be set equal to the index of the global_subdomain_id in the list below,
    ///     // multiplied by the size of a single chunk of data per subdomain.
    ///     // That means that in the worst case, the entire list must be read to find the correct subdomain.
    ///     // However, this way it is easy to _write_ the metadata since we do not need to globally sort the subdomain
    ///     // positions in the parallel setting, and we get away with O(1) local memory usage during parsing.
    ///     // Lookup is O(n), though.
    ///     //
    ///     // An alternative would be to sort the list before writing the checkpoint and store the _position of the
    ///     // subdomain data_ instead of the global_subdomain_id. Since the global_subdomain_id is sortable, an
    ///     // implicit mapping from global_subdomain_id to this list's index can be accomplished.
    ///     // Then look up the position of the data in O(1).
    ///     // However, that would require reducing the entire list on one process which is in theory not scalable
    ///     // (plus the sorting approach is a tiny bit more complicated).
    ///     // Use a different value for checkpoint_subdomain_ordering in that case and extend the file format.
    ///     list (size = "num_global_subdomains" = 10 * num_subdomains_per_diamond_lateral_direction * num_subdomains_per_diamond_lateral_direction * num_subdomains_per_diamond_radial_direction)
    ///     [
    ///         global_subdomain_id: i64
    ///     ]
    /// }
    /// @endcode
    ///
    /// @note Must be called collectively by all processes.
    ///
    void write_checkpoint_metadata()
    {
        const auto checkpoint_metadata_file_path = directory_path_ + "/" + "checkpoint_metadata.bin";

        std::stringstream checkpoint_metadata_stream;

        auto write_i32 = [&]( const int32_t value ) {
            checkpoint_metadata_stream.write( reinterpret_cast< const char* >( &value ), sizeof( int32_t ) );
        };

        auto write_i64 = [&]( const int64_t value ) {
            checkpoint_metadata_stream.write( reinterpret_cast< const char* >( &value ), sizeof( int64_t ) );
        };

        auto write_f64 = [&]( const double value ) {
            checkpoint_metadata_stream.write( reinterpret_cast< const char* >( &value ), sizeof( double ) );
        };

        write_i32( 0 ); // version
        write_i32( distributed_domain_.domain_info().num_subdomains_per_diamond_side() );
        write_i32( distributed_domain_.domain_info().num_subdomains_in_radial_direction() );

        write_i32( coords_shell_device_.extent( 1 ) ); // subdomain_size_x
        write_i32( coords_shell_device_.extent( 2 ) ); // subdomain_size_y
        write_i32( coords_radii_device_.extent( 1 ) ); // subdomain_size_r

        for ( const auto r : distributed_domain_.domain_info().radii() )
        {
            write_f64( static_cast< double >( r ) );
        }

        write_i32(
            device_data_views_scalar_float_.size() + device_data_views_scalar_double_.size() +
            device_data_views_vec_float_.size() + device_data_views_vec_double_.size() );

        for ( const auto& [data, output_type] : device_data_views_scalar_float_ )
        {
            write_i32( data.label().size() );
            checkpoint_metadata_stream.write( data.label().data(), data.label().size() );
            write_i32( 2 ); // scalar_data_type
            if ( output_type == OutputTypeFloat::Float32 )
            {
                write_i32( sizeof( float ) );
            }
            else if ( output_type == OutputTypeFloat::Float64 )
            {
                write_i32( sizeof( double ) );
            }
            else
            {
                Kokkos::abort( "Invalid output type." );
            }

            write_i32( 1 ); // vec_dim
        }

        for ( const auto& [data, output_type] : device_data_views_scalar_double_ )
        {
            write_i32( data.label().size() );
            checkpoint_metadata_stream.write( data.label().data(), data.label().size() );
            write_i32( 2 ); // scalar_data_type
            if ( output_type == OutputTypeFloat::Float32 )
            {
                write_i32( sizeof( float ) );
            }
            else if ( output_type == OutputTypeFloat::Float64 )
            {
                write_i32( sizeof( double ) );
            }
            else
            {
                Kokkos::abort( "Invalid output type." );
            }
            write_i32( 1 ); // vec_dim
        }

        for ( const auto& [data, output_type] : device_data_views_vec_float_ )
        {
            write_i32( data.label().size() );
            checkpoint_metadata_stream.write( data.label().data(), data.label().size() );
            write_i32( 2 ); // scalar_data_type
            if ( output_type == OutputTypeFloat::Float32 )
            {
                write_i32( sizeof( float ) );
            }
            else if ( output_type == OutputTypeFloat::Float64 )
            {
                write_i32( sizeof( double ) );
            }
            else
            {
                Kokkos::abort( "Invalid output type." );
            }
            write_i32( data.extent( 4 ) ); // vec_dim
        }

        for ( const auto& [data, output_type] : device_data_views_vec_double_ )
        {
            write_i32( data.label().size() );
            checkpoint_metadata_stream.write( data.label().data(), data.label().size() );
            write_i32( 2 ); // scalar_data_type
            if ( output_type == OutputTypeFloat::Float32 )
            {
                write_i32( sizeof( float ) );
            }
            else if ( output_type == OutputTypeFloat::Float64 )
            {
                write_i32( sizeof( double ) );
            }
            else
            {
                Kokkos::abort( "Invalid output type." );
            }
            write_i32( data.extent( 4 ) ); // vec_dim
        }

        write_i32( 0 ); // checkpoint_subdomain_ordering

        // for ( int local_subdomain_id = 0; local_subdomain_id < coords_shell_device_.extent( 0 ); local_subdomain_id++ )
        // {
        //     write_i64( distributed_domain_.subdomain_info_from_local_idx( local_subdomain_id ).global_id() );
        // }

        MPI_File fh;
        MPI_File_open(
            MPI_COMM_WORLD,
            checkpoint_metadata_file_path.c_str(),
            MPI_MODE_CREATE | MPI_MODE_RDWR,
            MPI_INFO_NULL,
            &fh );

        // Define the file view: each process writes its local data sequentially
        MPI_Offset disp                       = 0;
        const auto offset_metadata_size_bytes = static_cast< int >( checkpoint_metadata_stream.str().size() );
        if ( mpi::rank() != 0 )
        {
            // We'll write the global metadata just via rank 0.
            // Next up: each process writes the global subdomain IDs of their local subdomains in parallel.
            // We have previously also filled the stream with the global metadata on non-root processes to facilitate
            // computing its length (we need to know where to start writing).
            // After computing its length and before writing the subdomain ids we clear the stream here.
            disp = offset_metadata_size_bytes + number_of_subdomains_offset_ * sizeof( int64_t );
            checkpoint_metadata_stream.clear();
            checkpoint_metadata_stream.str( "" );
        }

        // Writing the global subdomain ids of the local subdomains now in contiguous chunks per process.
        // Must be the order of the local_subdomain_id here to adhere to the ordering of the data output (which also
        // writes in that order).
        for ( int local_subdomain_id = 0; local_subdomain_id < coords_shell_device_.extent( 0 ); local_subdomain_id++ )
        {
            write_i64( distributed_domain_.subdomain_info_from_local_idx( local_subdomain_id ).global_id() );
        }

        MPI_File_set_view( fh, disp, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL );

        std::string checkpoint_metadata_str = checkpoint_metadata_stream.str();

        // Write data collectively
        MPI_File_write_all(
            fh,
            checkpoint_metadata_str.data(),
            static_cast< int >( checkpoint_metadata_str.size() ),
            MPI_CHAR,
            MPI_STATUS_IGNORE );

        // Close the file
        MPI_File_close( &fh );
    }

    std::string directory_path_;

    grid::shell::DistributedDomain distributed_domain_;

    grid::Grid3DDataVec< InputGridScalarType, 3 > coords_shell_device_;
    grid::Grid2DDataScalar< InputGridScalarType > coords_radii_device_;

    OutputTypeFloat output_type_points_;
    OutputTypeInt   output_type_connectivity_;

    // Collecting all views that are written on every write call.
    std::vector< std::pair< grid::Grid4DDataScalar< double >, OutputTypeFloat > > device_data_views_scalar_double_;
    std::vector< std::pair< grid::Grid4DDataScalar< float >, OutputTypeFloat > >  device_data_views_scalar_float_;

    std::vector< std::pair< grid::Grid4DDataVec< double, 3 >, OutputTypeFloat > > device_data_views_vec_double_;
    std::vector< std::pair< grid::Grid4DDataVec< float, 3 >, OutputTypeFloat > >  device_data_views_vec_float_;

    // Just a single mirror for buffering during write.
    std::optional< grid::Grid4DDataScalar< double >::HostMirror > host_data_mirror_scalar_double_;
    std::optional< grid::Grid4DDataScalar< float >::HostMirror >  host_data_mirror_scalar_float_;

    std::optional< grid::Grid4DDataVec< double, 3 >::HostMirror > host_data_mirror_vec_double_;
    std::optional< grid::Grid4DDataVec< float, 3 >::HostMirror >  host_data_mirror_vec_float_;

    int write_counter_ = 0;

    int number_of_nodes_offset_      = -1;
    int number_of_elements_offset_   = -1;
    int number_of_subdomains_offset_ = -1;

    int number_of_nodes_global_      = -1;
    int number_of_elements_global_   = -1;
    int number_of_subdomains_global_ = -1;
};

/// Captures the format of the checkpoint metadata.
struct CheckpointMetadata
{
    int32_t version{};
    int32_t num_subdomains_per_diamond_lateral_direction{};
    int32_t num_subdomains_per_diamond_radial_direction{};
    int32_t size_x{};
    int32_t size_y{};
    int32_t size_r{};

    std::vector< double > radii;

    struct GridDataFile
    {
        std::string grid_name_string;
        int32_t     scalar_data_type{};
        int32_t     scalar_bytes{};
        int32_t     vec_dim{};
    };

    std::vector< GridDataFile > grid_data_files;

    int32_t checkpoint_subdomain_ordering{};

    std::vector< int64_t > checkpoint_ordering_0_global_subdomain_ids;
};

[[nodiscard]] inline util::Result< CheckpointMetadata >
    read_checkpoint_metadata( const std::string& checkpoint_directory )
{
    const auto metadata_file = checkpoint_directory + "/" + "checkpoint_metadata.bin";

    if ( !std::filesystem::exists( metadata_file ) )
    {
        return { "Could not find file: " + metadata_file };
    }

    MPI_File fh;
    MPI_File_open( MPI_COMM_WORLD, metadata_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh );
    MPI_Offset filesize;
    MPI_File_get_size( fh, &filesize );

    std::vector< char > buffer( filesize );
    MPI_File_read_all( fh, buffer.data(), filesize, MPI_BYTE, MPI_STATUS_IGNORE );
    MPI_File_close( &fh );

    std::string data( buffer.data(), buffer.size() );

    std::istringstream in( data, std::ios::binary );

    auto read_i32 = [&]( int32_t& value ) {
        in.read( reinterpret_cast< char* >( &value ), sizeof( value ) );
        if ( !in )
        {
            return 1;
        }
        return 0;
    };

    auto read_i64 = [&]( int64_t& value ) {
        in.read( reinterpret_cast< char* >( &value ), sizeof( value ) );
        if ( !in )
        {
            return 1;
        }
        return 0;
    };

    auto read_f64 = [&]( double& value ) {
        in.read( reinterpret_cast< char* >( &value ), sizeof( value ) );
        if ( !in )
        {
            return 1;
        }
        return 0;
    };

    const std::string read_error = "Failed to read from input stream.";

    CheckpointMetadata metadata;

    if ( read_i32( metadata.version ) )
        return read_error;
    if ( read_i32( metadata.num_subdomains_per_diamond_lateral_direction ) )
        return read_error;
    if ( read_i32( metadata.num_subdomains_per_diamond_radial_direction ) )
        return read_error;
    if ( read_i32( metadata.size_x ) )
        return read_error;
    if ( read_i32( metadata.size_y ) )
        return read_error;
    if ( read_i32( metadata.size_r ) )
        return read_error;

    for ( int i = 0; i < metadata.num_subdomains_per_diamond_radial_direction * ( metadata.size_r - 1 ) + 1; i++ )
    {
        double r;
        if ( read_f64( r ) )
            return read_error;

        if ( i > 0 && r <= metadata.radii.back() )
        {
            return { "Radii are not sorted correctly in checkpoint." };
        }

        metadata.radii.push_back( r );
    }

    int32_t num_grid_data_files;
    if ( read_i32( num_grid_data_files ) )
        return read_error;

    metadata.grid_data_files.resize( num_grid_data_files );
    for ( auto& [grid_name_string, scalar_data_type, scalar_bytes, vec_dim] : metadata.grid_data_files )
    {
        int32_t grid_name_string_size_bytes;
        if ( read_i32( grid_name_string_size_bytes ) )
            return read_error;

        grid_name_string = std::string( grid_name_string_size_bytes, '\0' );
        in.read( &grid_name_string[0], grid_name_string_size_bytes );
        if ( !in )
            return read_error;

        if ( read_i32( scalar_data_type ) )
            return read_error;
        if ( read_i32( scalar_bytes ) )
            return read_error;
        if ( read_i32( vec_dim ) )
            return read_error;
    }

    if ( read_i32( metadata.checkpoint_subdomain_ordering ) )
        return read_error;

    if ( metadata.checkpoint_subdomain_ordering == 0 )
    {
        const auto num_global_subdomains = 10 * metadata.num_subdomains_per_diamond_lateral_direction *
                                           metadata.num_subdomains_per_diamond_lateral_direction *
                                           metadata.num_subdomains_per_diamond_radial_direction;

        metadata.checkpoint_ordering_0_global_subdomain_ids.resize( num_global_subdomains );
        for ( auto& global_subdomain_id : metadata.checkpoint_ordering_0_global_subdomain_ids )
        {
            if ( read_i64( global_subdomain_id ) )
                return read_error + " (global_subdomain_id reading error)";
        }
    }

    return metadata;
}

template < typename GridDataType >
[[nodiscard]] util::Result<> read_checkpoint(
    const std::string&                    checkpoint_directory,
    const std::string&                    data_label,
    const int                             step,
    const grid::shell::DistributedDomain& distributed_domain,
    GridDataType&                         grid_data_device )
{
    auto checkpoint_metadata_result = read_checkpoint_metadata( checkpoint_directory );

    if ( checkpoint_metadata_result.is_err() )
    {
        return "Could not read checkpoint metadata: " + checkpoint_metadata_result.error();
    }

    const auto& checkpoint_metadata = checkpoint_metadata_result.unwrap();

    if ( checkpoint_metadata.version != 0 )
    {
        return { "Checkpoint version other than 0 is not supported." };
    }

    // Check whether we have checkpoint metadata for the requested data label.

    std::optional< CheckpointMetadata::GridDataFile > requested_grid_data_file;
    for ( const auto& grid_data_file : checkpoint_metadata.grid_data_files )
    {
        if ( grid_data_file.grid_name_string == data_label )
        {
            requested_grid_data_file = grid_data_file;
            break;
        }
    }

    if ( !requested_grid_data_file.has_value() )
    {
        return { "Could not find requested data (" + data_label + ") in checkpoint" };
    }

    // Check if we can also find a binary data file in the checkpoint directory.

    const auto data_file_path = checkpoint_directory + "/" + data_label + "_" + std::to_string( step ) + ".bin";

    if ( !std::filesystem::exists( data_file_path ) )
    {
        return { "Could not find checkpoint data file for requested label and step." };
    }

    // Now we compare the grid extents with the metadata.

    if ( grid_data_device.extent( 1 ) != checkpoint_metadata.size_x ||
         grid_data_device.extent( 2 ) != checkpoint_metadata.size_y ||
         grid_data_device.extent( 3 ) != checkpoint_metadata.size_r )
    {
        return { "Grid data extents do not match metadata." };
    }

    // Let's try to read now.
    // We will attempt to read all chunks for the local subdomains (that are possibly distributed in the file)
    // into one array. In a second step we'll copy that into our grid data. That is not 100% memory efficient as we
    // have another copy (the buffer). We could write directly with the std library, but I am not sure if that is
    // scalable.

    if ( checkpoint_metadata.checkpoint_subdomain_ordering != 0 )
    {
        return { "Checkpoint ordering type is not 0. Not supported." };
    }

    // Figure out where the subdomain data is positioned in the file.

    const auto num_local_subdomains = grid_data_device.extent( 0 );

    const auto local_subdomain_num_scalars_v = checkpoint_metadata.size_x * checkpoint_metadata.size_y *
                                               checkpoint_metadata.size_r * requested_grid_data_file.value().vec_dim;

    const auto local_subdomain_data_size_in_bytes =
        local_subdomain_num_scalars_v * requested_grid_data_file.value().scalar_bytes;

    std::vector< MPI_Aint > local_subdomain_offsets_in_file_bytes( num_local_subdomains );
    std::vector< int >      local_subdomain_num_bytes( num_local_subdomains, local_subdomain_data_size_in_bytes );

    for ( int local_subdomain_id = 0; local_subdomain_id < num_local_subdomains; local_subdomain_id++ )
    {
        const auto global_subdomain_id =
            distributed_domain.subdomain_info_from_local_idx( local_subdomain_id ).global_id();

        const auto index_in_file =
            std::ranges::find( checkpoint_metadata.checkpoint_ordering_0_global_subdomain_ids, global_subdomain_id ) -
            checkpoint_metadata.checkpoint_ordering_0_global_subdomain_ids.begin();

        local_subdomain_offsets_in_file_bytes[local_subdomain_id] = index_in_file * local_subdomain_data_size_in_bytes;
    }

    // Read with MPI

    std::vector< char > buffer( num_local_subdomains * local_subdomain_data_size_in_bytes );

    MPI_File fh;
    MPI_File_open( MPI_COMM_WORLD, data_file_path.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh );

    MPI_Datatype filetype;
    MPI_Type_create_hindexed(
        num_local_subdomains,
        local_subdomain_num_bytes.data(),
        local_subdomain_offsets_in_file_bytes.data(),
        MPI_CHAR,
        &filetype );
    MPI_Type_commit( &filetype );

    // Set each rank’s view of the file to its own scattered layout
    MPI_File_set_view( fh, 0, MPI_CHAR, filetype, "native", MPI_INFO_NULL );

    MPI_File_read_all( fh, buffer.data(), buffer.size(), MPI_CHAR, MPI_STATUS_IGNORE );

    MPI_Type_free( &filetype );
    MPI_File_close( &fh );

    // Now write from buffer to grid.

    typename GridDataType::HostMirror grid_data_host =
        Kokkos::create_mirror_view( Kokkos::HostSpace{}, grid_data_device );

    const auto checkpoint_is_float =
        requested_grid_data_file.value().scalar_data_type == 2 && requested_grid_data_file.value().scalar_bytes == 4;
    const auto checkpoint_is_double =
        requested_grid_data_file.value().scalar_data_type == 2 && requested_grid_data_file.value().scalar_bytes == 8;

    if ( !( checkpoint_is_float || checkpoint_is_double ) )
    {
        return { "Unsupported data type in checkpoint." };
    }

    // Wrap vector in a stream
    std::string_view   view( reinterpret_cast< const char* >( buffer.data() ), buffer.size() );
    std::istringstream stream{ std::string( view ) }; // make a copy to own the buffer

    auto read_f32 = [&]() -> float {
        float value;
        stream.read( reinterpret_cast< char* >( &value ), sizeof( float ) );
        if ( stream.fail() )
        {
            Kokkos::abort( "Failed to read from stream." );
        }
        return value;
    };

    auto read_f64 = [&]() -> double {
        double value;
        stream.read( reinterpret_cast< char* >( &value ), sizeof( double ) );
        if ( stream.fail() )
        {
            Kokkos::abort( "Failed to read from stream." );
        }
        return value;
    };

    for ( int local_subdomain_id = 0; local_subdomain_id < grid_data_host.extent( 0 ); local_subdomain_id++ )
    {
        for ( int r = 0; r < grid_data_host.extent( 3 ); r++ )
        {
            for ( int y = 0; y < grid_data_host.extent( 2 ); y++ )
            {
                for ( int x = 0; x < grid_data_host.extent( 1 ); x++ )
                {
                    if constexpr (
                        std::is_same_v< GridDataType, grid::Grid4DDataScalar< float > > ||
                        std::is_same_v< GridDataType, grid::Grid4DDataScalar< double > > )
                    {
                        if ( requested_grid_data_file.value().vec_dim != 1 )
                        {
                            return { "Checkpoint is scalar, passed grid data view dims do not match." };
                        }

                        // scalar data
                        if ( checkpoint_is_float )
                        {
                            grid_data_host( local_subdomain_id, x, y, r ) =
                                static_cast< GridDataType::value_type >( read_f32() );
                        }
                        else if ( checkpoint_is_double )
                        {
                            grid_data_host( local_subdomain_id, x, y, r ) =
                                static_cast< GridDataType::value_type >( read_f64() );
                        }
                    }
                    else if constexpr (
                        std::is_same_v< GridDataType, grid::Grid4DDataVec< float, 3 > > ||
                        std::is_same_v< GridDataType, grid::Grid4DDataVec< double, 3 > > )
                    {
                        if ( requested_grid_data_file.value().vec_dim != 3 )
                        {
                            return { "Checkpoint is vector-valued, passed grid data view dims do not match." };
                        }

                        // vector-valued data
                        for ( int d = 0; d < grid_data_host.extent( 4 ); d++ )
                        {
                            if ( checkpoint_is_float )
                            {
                                grid_data_host( local_subdomain_id, x, y, r, d ) =
                                    static_cast< GridDataType::value_type >( read_f32() );
                            }
                            else if ( checkpoint_is_double )
                            {
                                grid_data_host( local_subdomain_id, x, y, r, d ) =
                                    static_cast< GridDataType::value_type >( read_f64() );
                            }
                        }
                    }
                }
            }
        }
    }

    Kokkos::deep_copy( grid_data_device, grid_data_host );
    Kokkos::fence();

    return { util::Ok{} };
}

} // namespace terra::visualization