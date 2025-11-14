#pragma once

#include <string>
#include <variant>

#include "util/cli11_helper.hpp"
#include "util/info.hpp"
#include "util/result.hpp"

namespace terra::mantlecirculation {

struct MeshParameters
{
    int refinement_level_mesh_min   = 0;
    int refinement_level_mesh_max   = 4;
    int refinement_level_subdomains = 0;

    double radius_min = 0.5;
    double radius_max = 1.0;
};

struct IsoViscParameters
{};

struct VarViscParameters
{};

struct PhysicsParameters
{
    double diffusivity     = 1.0;
    double rayleigh_number = 1e5;
};

struct StokesSolverParameters
{
    int    krylov_restart            = 10;
    int    krylov_max_iterations     = 10;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;

    int viscous_pc_num_vcycles                 = 2;
    int viscous_pc_num_smoothing_steps_prepost = 2;
};

struct EnergySolverParameters
{
    int    krylov_restart            = 5;
    int    krylov_max_iterations     = 100;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;
};

struct TimeSteppingParameters
{
    double pseudo_cfl = 0.5;
    double t_end      = 10.0;

    int max_timesteps = 1000;

    int energy_substeps = 1;
};

struct IOParameters
{
    std::string outdir    = "output";
    bool        overwrite = false;

    std::string xdmf_dir            = "xdmf";
    std::string radial_profiles_dir = "radial_profiles";
    std::string timer_trees_dir     = "timer_trees";
};

struct Parameters
{
    MeshParameters                                                       mesh_parameters;
    std::variant< std::monostate, IsoViscParameters, VarViscParameters > viscosity_parameters;
    StokesSolverParameters                                               stokes_solver_parameters;
    EnergySolverParameters                                               energy_solver_parameters;
    PhysicsParameters                                                    physics_parameters;
    TimeSteppingParameters                                               time_stepping_parameters;
    IOParameters                                                         io_parameters;

    [[nodiscard]] bool is_isoviscous() const
    {
        return std::holds_alternative< IsoViscParameters >( viscosity_parameters );
    }

    [[nodiscard]] bool is_varviscous() const
    {
        return std::holds_alternative< VarViscParameters >( viscosity_parameters );
    }
};

inline util::Result< Parameters > parse_parameters( int argc, char** argv )
{
    CLI::App app{ "Mantle circulation simulation." };

    Parameters parameters{};

    using util::add_flag_with_default;
    using util::add_option_with_default;

    add_option_with_default( app, "--refinement-level-mesh-min", parameters.mesh_parameters.refinement_level_mesh_min );
    add_option_with_default( app, "--refinement-level-mesh-max", parameters.mesh_parameters.refinement_level_mesh_max );
    add_option_with_default( app, "--radius-min", parameters.mesh_parameters.radius_min );
    add_option_with_default( app, "--radius-max", parameters.mesh_parameters.radius_max );

    add_option_with_default( app, "--diffusivity", parameters.physics_parameters.diffusivity );
    add_option_with_default( app, "--rayleigh-number", parameters.physics_parameters.rayleigh_number );

    add_option_with_default( app, "--pseudo-cfl", parameters.time_stepping_parameters.pseudo_cfl );
    add_option_with_default( app, "--t-end", parameters.time_stepping_parameters.t_end );
    add_option_with_default( app, "--max-timesteps", parameters.time_stepping_parameters.max_timesteps );
    add_option_with_default( app, "--energy-substeps", parameters.time_stepping_parameters.energy_substeps );

    add_option_with_default( app, "--stokes-krylov-restart", parameters.stokes_solver_parameters.krylov_restart );
    add_option_with_default(
        app, "--stokes-krylov-max-iterations", parameters.stokes_solver_parameters.krylov_max_iterations );
    add_option_with_default(
        app, "--stokes-krylov-relative-tolerance", parameters.stokes_solver_parameters.krylov_relative_tolerance );
    add_option_with_default(
        app, "--stokes-krylov-absolute-tolerance", parameters.stokes_solver_parameters.krylov_absolute_tolerance );
    add_option_with_default(
        app, "--stokes-viscous-pc-num-vcycles", parameters.stokes_solver_parameters.viscous_pc_num_vcycles );
    add_option_with_default(
        app,
        "--stokes-viscous-pc-num-smoothing-steps-prepost",
        parameters.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost );

    add_option_with_default( app, "--outdir", parameters.io_parameters.outdir );
    add_flag_with_default( app, "--outdir-overwrite", parameters.io_parameters.overwrite );

    try
    {
        app.parse( argc, argv );
    }
    catch ( const CLI::ParseError& e )
    {
        app.exit( e );
        return { "CLI parse error" };
    }

    util::print_general_info( argc, argv, util::logroot );
    util::print_cli_summary( app, util::logroot );
    return parameters;
}

}; // namespace terra::mantlecirculation