
#pragma once
#include <filesystem>

#include "kokkos/kokkos_wrapper.hpp"
#include "table.hpp"

namespace terra::util {

/// @brief Prints some general information for this run to out (set, e.g., to std::cout) on the root process.
inline void print_general_info_on_root( int argc, char** argv, std::ostream& out )
{
    if ( mpi::rank() != 0 )
    {
        return;
    }

    using clock      = std::chrono::system_clock;
    const auto now   = clock::now();
    const auto now_c = clock::to_time_t( now );

    out << "=========================================\n";
    out << "          TerraNeo X - Run Info          \n";
    out << "=========================================\n";

    out << "Wall time start : " << std::put_time( std::localtime( &now_c ), "%Y-%m-%d %H:%M:%S" ) << "\n";

    // Binary path and working directory
    try
    {
        std::filesystem::path bin_path( argv[0] );
        bin_path = std::filesystem::absolute( bin_path );
        out << "Executable path : " << bin_path << "\n";
        out << "Executable dir  : " << bin_path.parent_path() << "\n";
        out << "Working dir     : " << std::filesystem::current_path() << "\n";
    }
    catch ( ... )
    {
        out << "Executable path : (unavailable)\n";
    }

    // Command-line arguments
    out << "Command line    :";
    for ( int i = 0; i < argc; ++i )
    {
        out << " " << argv[i];
    }
    out << "\n";

    // Parallel resources
    const auto threads   = Kokkos::num_threads();
    const auto devices   = Kokkos::num_devices();
    const auto mpi_procs = mpi::num_processes();
    const auto mpi_rank  = mpi::rank();

    out << "MPI processes   : " << mpi_procs << "\n";
    out << "MPI rank        : " << mpi_rank << "\n";
    out << "Kokkos threads  : " << threads << "\n";
    out << "Kokkos devices  : " << devices << "\n";

    // Kokkos defaults
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace  = ExecSpace::memory_space;
    out << "ExecSpace       : " << ExecSpace::name() << "\n";
    out << "MemSpace        : " << MemSpace::name() << "\n";

    // Optional: print host/system info if available
#if defined( __linux__ )
    char hostname[256];
    if ( gethostname( hostname, sizeof( hostname ) ) == 0 )
    {
        out << "Hostname        : " << hostname << "\n";
    }
#endif

    // Separator footer
    out << "=========================================\n" << std::endl;
}

} // namespace terra::util