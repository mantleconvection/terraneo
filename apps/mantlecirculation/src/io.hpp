
#pragma once

#include <fstream>
#include <vector>

#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "parameters.hpp"
#include "shell/radial_profiles.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"
#include "util/result.hpp"
#include "util/table.hpp"
#include "util/timer.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using util::logroot;
using util::Ok;
using util::Result;

using ScalarType = double;

namespace terra::mantlecirculation {

inline Result<> create_directories( const IOParameters& io_parameters )
{
    const auto xdmf_dir            = io_parameters.outdir + "/" + io_parameters.xdmf_dir;
    const auto radial_profiles_dir = io_parameters.outdir + "/" + io_parameters.radial_profiles_dir;
    const auto timer_trees_dir     = io_parameters.outdir + "/" + io_parameters.timer_trees_dir;

    if ( !io_parameters.overwrite && std::filesystem::exists( io_parameters.outdir ) )
    {
        return { "Will not overwrite existing directory (to not accidentally delete old simulation data). "
                 "Use -h for help and look for overwrite option or choose a different output dir name." };
    }

    util::prepare_empty_directory( io_parameters.outdir );
    util::prepare_empty_directory( xdmf_dir );
    util::prepare_empty_directory( radial_profiles_dir );
    util::prepare_empty_directory( timer_trees_dir );

    return { Ok{} };
}

inline Result<> compute_and_write_radial_profiles(
    const VectorQ1Scalar< ScalarType >& T,
    const Grid2DDataScalar< int >&      subdomain_shell_idx,
    const DistributedDomain&            domain,
    const IOParameters&                 io_parameters,
    const int                           timestep )
{
    const auto profiles = shell::radial_profiles_to_table< ScalarType >(
        shell::radial_profiles( T, subdomain_shell_idx, static_cast< int >( domain.domain_info().radii().size() ) ),
        domain.domain_info().radii() );
    std::ofstream out(
        io_parameters.outdir + "/" + io_parameters.radial_profiles_dir + "/radial_profiles_" +
        std::to_string( timestep ) + ".csv" );
    profiles.print_csv( out );

    return { Ok{} };
}

inline Result<> write_timer_tree( const IOParameters& io_parameters, const int timestep )
{
    util::TimerTree::instance().aggregate_mpi();
    if ( mpi::rank() == 0 )
    {
        const auto timer_tree_file = io_parameters.outdir + "/" + io_parameters.timer_trees_dir + "/timer_tree_" +
                                     std::to_string( timestep ) + ".json";
        logroot << "Writing timer tree to " << timer_tree_file << std::endl;
        std::ofstream out( timer_tree_file );
        out << util::TimerTree::instance().json_aggregate();
        out.close();
    }

    return { Ok{} };
}

} // namespace terra::mantlecirculation
