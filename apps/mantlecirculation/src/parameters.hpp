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
    int refinement_level_mesh_max   = 3;
    int refinement_level_subdomains = 0;

    double radius_min = 0.5;
    double radius_max = 1.0;
};

struct ViscosityParameters
{
    bool        radial_profile_enabled       = false;
    std::string radial_profile_csv_filename  = "radial_viscosity_profile.csv";
    std::string radial_profile_radii_key     = "radii";
    std::string radial_profile_viscosity_key = "viscosity";
    double      reference_viscosity          = 1.0;
};

struct PhysicsParameters
{
    double diffusivity     = 1.0;
    double rayleigh_number = 1e5;

    ViscosityParameters viscosity_parameters{};
};

struct StokesSolverParameters
{
    int    krylov_restart            = 10;
    int    krylov_max_iterations     = 10;
    double krylov_relative_tolerance = 1e-6;
    double krylov_absolute_tolerance = 1e-12;

    int viscous_pc_num_vcycles                 = 2;
    int viscous_pc_num_smoothing_steps_prepost = 2;
    int viscous_pc_num_power_iterations        = 10;
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
    double t_end      = 1.0;

    int max_timesteps = 10;

    int energy_substeps = 1;
};

struct IOParameters
{
    std::string outdir    = "output";
    bool        overwrite = false;

    std::string xdmf_dir                = "xdmf";
    std::string radial_profiles_out_dir = "radial_profiles";
    std::string timer_trees_dir         = "timer_trees";
};

struct Parameters
{
    MeshParameters         mesh_parameters;
    StokesSolverParameters stokes_solver_parameters;
    EnergySolverParameters energy_solver_parameters;
    PhysicsParameters      physics_parameters;
    TimeSteppingParameters time_stepping_parameters;
    IOParameters           io_parameters;
};

struct CLIHelp
{};

inline util::Result< std::variant< CLIHelp, Parameters > > parse_parameters( int argc, char** argv )
{
    CLI::App app{ "Mantle circulation simulation." };

    Parameters parameters{};

    using util::add_flag_with_default;
    using util::add_option_with_default;

    ///////////////////////
    /// Domain and mesh ///
    ///////////////////////

    add_option_with_default( app, "--refinement-level-mesh-min", parameters.mesh_parameters.refinement_level_mesh_min )
        ->group( "Domain" );
    add_option_with_default( app, "--refinement-level-mesh-max", parameters.mesh_parameters.refinement_level_mesh_max )
        ->group( "Domain" );
    add_option_with_default( app, "--radius-min", parameters.mesh_parameters.radius_min )->group( "Domain" );
    add_option_with_default( app, "--radius-max", parameters.mesh_parameters.radius_max )->group( "Domain" );

    //////////////////////////////
    /// Geophysical parameters ///
    //////////////////////////////

    add_option_with_default( app, "--diffusivity", parameters.physics_parameters.diffusivity );
    add_option_with_default( app, "--rayleigh-number", parameters.physics_parameters.rayleigh_number );

    const auto radial_profile_enabled = add_flag_with_default(
                                            app,
                                            "--viscosity-radial-profile",
                                            parameters.physics_parameters.viscosity_parameters.radial_profile_enabled )
                                            ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-csv-filename",
        parameters.physics_parameters.viscosity_parameters.radial_profile_csv_filename )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-radii-key",
        parameters.physics_parameters.viscosity_parameters.radial_profile_radii_key )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app,
        "--viscosity-radial-profile-value-key",
        parameters.physics_parameters.viscosity_parameters.radial_profile_viscosity_key )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );
    add_option_with_default(
        app, "--viscosity-reference-value", parameters.physics_parameters.viscosity_parameters.reference_viscosity )
        ->needs( radial_profile_enabled )
        ->group( "Viscosity" );

    ///////////////////////////
    /// Time discretization ///
    ///////////////////////////

    add_option_with_default( app, "--pseudo-cfl", parameters.time_stepping_parameters.pseudo_cfl )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--t-end", parameters.time_stepping_parameters.t_end )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--max-timesteps", parameters.time_stepping_parameters.max_timesteps )
        ->group( "Time Discretization" );
    add_option_with_default( app, "--energy-substeps", parameters.time_stepping_parameters.energy_substeps )
        ->group( "Time Discretization" );

    /////////////////////
    /// Stokes solver ///
    /////////////////////

    add_option_with_default( app, "--stokes-krylov-restart", parameters.stokes_solver_parameters.krylov_restart )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-max-iterations", parameters.stokes_solver_parameters.krylov_max_iterations )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-relative-tolerance", parameters.stokes_solver_parameters.krylov_relative_tolerance )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-krylov-absolute-tolerance", parameters.stokes_solver_parameters.krylov_absolute_tolerance )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app, "--stokes-viscous-pc-num-vcycles", parameters.stokes_solver_parameters.viscous_pc_num_vcycles )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app,
        "--stokes-viscous-pc-num-smoothing-steps-prepost",
        parameters.stokes_solver_parameters.viscous_pc_num_smoothing_steps_prepost )
        ->group( "Stokes Solver" );
    add_option_with_default(
        app,
        "--stokes-viscous-pc-num-power-iterations",
        parameters.stokes_solver_parameters.viscous_pc_num_power_iterations )
        ->group( "Stokes Solver" );

    //////////////////////
    /// Input / output ///
    //////////////////////

    add_option_with_default( app, "--outdir", parameters.io_parameters.outdir )->group( "I/O" );
    add_flag_with_default( app, "--outdir-overwrite", parameters.io_parameters.overwrite )->group( "I/O" );

    try
    {
        app.parse( argc, argv );
    }
    catch ( const CLI::ParseError& e )
    {
        app.exit( e );
        if ( e.get_exit_code() == static_cast< int >( CLI::ExitCodes::Success ) )
        {
            return { CLIHelp{} };
        }
        return { "CLI parse error" };
    }

    util::print_general_info( argc, argv, util::logroot );
    util::print_cli_summary( app, util::logroot );
    util::logroot << std::endl;
    return { parameters };
}

}; // namespace terra::mantlecirculation