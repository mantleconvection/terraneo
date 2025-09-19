

#include <fstream>
#include <util/arg_parser.hpp>
#include <vector>

#include "communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/identity.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "grid/grid_types.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "kokkos/kokkos_wrapper.hpp"
#include "linalg/solvers/block_preconditioner_2x2.hpp"
#include "linalg/solvers/jacobi.hpp"
#include "linalg/solvers/multigrid.hpp"
#include "linalg/solvers/pbicgstab.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "shell/radial_profiles.hpp"
#include "util/init.hpp"
#include "util/table.hpp"
#include "visualization/xdmf.hpp"

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

using ScalarType = float;

struct Parameters
{
    int min_level;
    int max_level;

    ScalarType r_min;
    ScalarType r_max;

    ScalarType diffusivity;
    ScalarType rayleigh;

    ScalarType pseudo_cfl;
    ScalarType t_end;
    int        max_timesteps;

    int stokes_bicgstab_l;
    int stokes_bicgstab_max_iterations;
    int num_vcycles;
    int num_smoothing_steps_prepost;

    bool xdmf;
};

struct InitialConditionInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > data_;
    bool                           only_boundary_;

    InitialConditionInterpolator(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataScalar< ScalarType >& data,
        bool                                  only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            const dense::Vec< ScalarType, 3 > coords =
                grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

            const ScalarType radius = coords.norm();
            const ScalarType perturbation =
                // ( 0.5 - radius ) * ( 1.0 - radius ) * 0.5 * Kokkos::sin( 10.0 * coords( 0 ) + 3.0 * coords( 1 ) );
                0.0 * radius;

            data_( local_subdomain_id, x, y, r ) = Kokkos::pow( 2.0 * ( 1.0 - coords.norm() ), 5 ) + perturbation;
        }
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > data_u_;
    Grid4DDataScalar< ScalarType > data_T_;
    ScalarType                     rayleigh_number_;

    RHSVelocityInterpolator(
        const Grid3DDataVec< ScalarType, 3 >& grid,
        const Grid2DDataScalar< ScalarType >& radii,
        const Grid4DDataVec< ScalarType, 3 >& data_u,
        const Grid4DDataScalar< ScalarType >& data_T,
        ScalarType                            rayleigh_number )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , data_T_( data_T )
    , rayleigh_number_( rayleigh_number )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< ScalarType, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const auto n = coords.normalized();

        for ( int d = 0; d < 3; d++ )
        {
            data_u_( local_subdomain_id, x, y, r, d ) =
                rayleigh_number_ * n( d ) * data_T_( local_subdomain_id, x, y, r );
        }
    }
};

void run( const Parameters& prm, const std::shared_ptr< util::Table >& table )
{
    // Set up domains for all levels.

    std::vector< DistributedDomain >                  domains;
    std::vector< Grid3DDataVec< ScalarType, 3 > >     coords_shell;
    std::vector< Grid2DDataScalar< ScalarType > >     coords_radii;
    std::vector< Grid4DDataScalar< util::MaskType > > mask_data;

    for ( int level = prm.min_level; level <= prm.max_level; level++ )
    {
        const int idx = level - prm.min_level;

        domains.push_back( DistributedDomain::create_uniform_single_subdomain( level, level, prm.r_min, prm.r_max ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii< ScalarType >( domains[idx] ) );
        mask_data.push_back( linalg::setup_mask_data( domains[idx] ) );
    }

    const auto num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    // Set up Stokes vectors for the finest grid.

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    std::vector< std::string >                             stok_vec_names = { "u", "f" };

    const auto num_stokes_bicgstab_tmps = 2 * ( prm.stokes_bicgstab_l + 1 ) + 2;

    const auto num_stok_tmps = num_stokes_bicgstab_tmps;

    for ( int i = 0; i < num_stok_tmps; i++ )
    {
        stok_vec_names.push_back( "tmp_" + std::to_string( i ) );
    }

    for ( const auto& name : stok_vec_names )
    {
        stok_vecs[name] = VectorQ1IsoQ2Q1< ScalarType >(
            name,
            domains[velocity_level],
            domains[pressure_level],
            mask_data[velocity_level],
            mask_data[pressure_level] );
    }

    auto& u = stok_vecs["u"];
    auto& f = stok_vecs["f"];

    // Set up tmp vecs for multigrid.

    std::vector< VectorQ1Vec< ScalarType > > tmp_mg;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_r;
    std::vector< VectorQ1Vec< ScalarType > > tmp_mg_e;

    for ( int level = 0; level < num_levels; level++ )
    {
        tmp_mg.emplace_back( "tmp_mg_" + std::to_string( level ), domains[level], mask_data[level] );
        if ( level < num_levels - 1 )
        {
            tmp_mg_r.emplace_back( "tmp_mg_r_" + std::to_string( level ), domains[level], mask_data[level] );
            tmp_mg_e.emplace_back( "tmp_mg_e_" + std::to_string( level ), domains[level], mask_data[level] );
        }
    }

    // Set up temperature vectors.

    std::map< std::string, VectorQ1Scalar< ScalarType > > temp_vecs;
    std::vector< std::string >                            temp_vec_names = { "T", "q" };
    constexpr int                                         num_temp_tmps  = 8;

    for ( int i = 0; i < num_temp_tmps; i++ )
    {
        temp_vec_names.push_back( "tmp_" + std::to_string( i ) );
    }

    for ( const auto& name : temp_vec_names )
    {
        temp_vecs[name] = VectorQ1Scalar< ScalarType >( name, domains[velocity_level], mask_data[velocity_level] );
    }

    auto& T = temp_vecs["T"];
    auto& q = temp_vecs["q"];

    // Counting DoFs.

    const auto num_dofs_temperature =
        kernels::common::count_masked< long >( mask_data[num_levels - 1], grid::mask_owned() );
    const auto num_dofs_velocity = 3 * num_dofs_temperature;
    const auto num_dofs_pressure =
        kernels::common::count_masked< long >( mask_data[num_levels - 2], grid::mask_owned() );

    // Set up operators.

    using Stokes      = fe::wedge::operators::shell::Stokes< ScalarType >;
    using Viscous     = Stokes::Block11Type;
    using ViscousMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

    using Prolongation = fe::wedge::operators::shell::ProlongationVecConstant< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::RestrictionVecConstant< ScalarType >;

    Stokes K(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        true,
        false );

    Stokes K_neumann(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        false,
        false );

    Stokes K_neumann_diag(
        domains[velocity_level],
        domains[pressure_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        false,
        true );

    ViscousMass M( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // Multigrid operators

    std::vector< Viscous >      A_diag;
    std::vector< Viscous >      A_c;
    std::vector< Prolongation > P;
    std::vector< Restriction >  R;

    std::vector< VectorQ1Vec< ScalarType > > inverse_diagonals;

    for ( int level = 0; level < num_levels; level++ )
    {
        A_diag.emplace_back( domains[level], coords_shell[level], coords_radii[level], true, true );

        inverse_diagonals.emplace_back(
            "inverse_diagonal_" + std::to_string( level ), domains[level], mask_data[level] );

        VectorQ1Vec< ScalarType > tmp(
            "inverse_diagonal_tmp" + std::to_string( level ), domains[level], mask_data[level] );

        linalg::assign( tmp, 1.0 );
        linalg::apply( A_diag[level], tmp, inverse_diagonals.back() );
        linalg::invert_entries( inverse_diagonals.back() );

        if ( level < num_levels - 1 )
        {
            A_c.emplace_back( domains[level], coords_shell[level], coords_radii[level], true, false );
            P.emplace_back( linalg::OperatorApplyMode::Add );
            R.emplace_back( domains[level] );
        }
    }

    // Set up solvers.

    // Multigrid preconditioner.

    using Smoother = linalg::solvers::Jacobi< Viscous >;

    std::vector< Smoother > smoothers;
    for ( int level = 0; level < num_levels; level++ )
    {
        constexpr auto omega = 0.666;
        smoothers.emplace_back( inverse_diagonals[level], prm.num_smoothing_steps_prepost, tmp_mg[level], omega );
    }

    using CoarseGridSolver = linalg::solvers::PCG< Viscous >;

    std::vector< VectorQ1Vec< ScalarType > > coarse_grid_tmps;
    for ( int i = 0; i < 4; i++ )
    {
        coarse_grid_tmps.emplace_back( "tmp_coarse_grid", domains[0], mask_data[0] );
    }

    CoarseGridSolver coarse_grid_solver(
        linalg::solvers::IterativeSolverParameters{ 10, 1e-6, 1e-16 }, table, coarse_grid_tmps );

    using PrecVisc = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;
    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_grid_solver, prm.num_vcycles, 1e-6 );

    using PrecSchur = linalg::solvers::IdentitySolver< Stokes::Block22Type >;
    PrecSchur prec_22;

    using PrecStokes =
        linalg::solvers::BlockDiagonalPreconditioner2x2< Stokes, Viscous, Stokes::Block22Type, PrecVisc, PrecSchur >;
    PrecStokes prec_stokes( K.block_11(), K.block_22(), prec_11, prec_22 );

    linalg::solvers::IterativeSolverParameters solver_params{ prm.stokes_bicgstab_max_iterations, 1e-6, 1e-12 };

    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_bicgstab( num_stokes_bicgstab_tmps );
    for ( int i = 0; i < num_stokes_bicgstab_tmps; i++ )
    {
        tmp_bicgstab[i] = stok_vecs["tmp_" + std::to_string( i )];
    }

    linalg::solvers::PBiCGStab< Stokes, PrecStokes > pbicgstab(
        prm.stokes_bicgstab_l, solver_params, table, tmp_bicgstab, prec_stokes );

    /////////////////////
    /// ENERGY SOLVER ///
    /////////////////////

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;

    constexpr auto mass_scaling = 1.0;

    AD A(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        u.block_1(),
        prm.diffusivity,
        0.0,
        true,
        false,
        mass_scaling );

    AD A_neumann(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        u.block_1(),
        prm.diffusivity,
        0.0,
        false,
        false,
        mass_scaling );

    AD A_neumann_diag(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        u.block_1(),
        prm.diffusivity,
        0.0,
        false,
        true,
        mass_scaling );

    using TempMass = fe::wedge::operators::shell::Mass< ScalarType >;

    TempMass M_T( domains[velocity_level], coords_shell[velocity_level], coords_radii[velocity_level], false );

    // Set up the initial temperature.

    Kokkos::parallel_for(
        "initial temp interpolation",
        local_domain_md_range_policy_nodes( domains[velocity_level] ),
        InitialConditionInterpolator(
            coords_shell[velocity_level], coords_radii[velocity_level], T.grid_data(), false ) );

    Kokkos::fence();

    std::vector< VectorQ1Scalar< ScalarType > > tmp_bicgstab_temp( 8 );
    for ( int i = 0; i < 8; i++ )
    {
        tmp_bicgstab_temp[i] = temp_vecs["tmp_" + std::to_string( i )];
    }

    linalg::solvers::PBiCGStab< AD > energy_solver(
        2, linalg::solvers::IterativeSolverParameters{ 100, 1e-6, 1e-12 }, table, tmp_bicgstab_temp );

    table->add_row( {
        { "tag", "setup" },
        { "dofs_velocity", num_dofs_velocity },
        { "dofs_temperature", num_dofs_temperature },
        { "dofs_pressure", num_dofs_pressure },
        { "level_velocity", velocity_level },
        { "level_pressure", pressure_level },
    } );

    table->print_pretty();
    table->clear();

    visualization::XDMFOutput xdmf_output( "xdmf", coords_shell[velocity_level], coords_radii[velocity_level] );

    xdmf_output.add( T.grid_data() );

    // Time stepping

    if ( prm.xdmf )
    {
        xdmf_output.write();

        auto profiles = shell::radial_profiles_to_table< ScalarType >(
            shell::radial_profiles( T ), domains[velocity_level].domain_info().radii() );
        std::ofstream out( "radial_profiles_" + std::to_string( 0 ) + ".csv" );
        profiles.print_csv( out );
    }

    ScalarType simulated_time = 0.0;

    // We need some global h. Let's, for simplicity (does not need to be too accurate) just choose the smallest h in
    // radial direction.
    const auto h = grid::shell::min_radial_h( domains[velocity_level].domain_info().radii() );

    for ( int timestep = 1; timestep < prm.max_timesteps; timestep++ )
    {
        if ( mpi::rank() == 0 )
        {
            std::cout << "Timestep " << timestep << std::endl;
        }

        // Set up rhs data for Stokes.

        if ( mpi::rank() == 0 )
        {
            std::cout << "Setting up Stokes rhs ..." << std::endl;
        }

        Kokkos::parallel_for(
            "Stokes rhs interpolation",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            RHSVelocityInterpolator(
                coords_shell[velocity_level],
                coords_radii[velocity_level],
                stok_vecs["tmp_1"].block_1().grid_data(),
                T.grid_data(),
                prm.rayleigh ) );

        linalg::apply( M, stok_vecs["tmp_1"].block_1(), stok_vecs["f"].block_1() );

        fe::strong_algebraic_homogeneous_velocity_dirichlet_enforcement_stokes_like(
            stok_vecs["f"], mask_data[velocity_level], grid::shell::mask_domain_boundary() );

        if ( mpi::rank() == 0 )
        {
            std::cout << "Solving Stokes ..." << std::endl;
        }

        // Solve Stokes.
        solve( pbicgstab, K, u, f );

        table->query_rows_equals( "tag", "pbicgstab_solver" ).print_pretty();
        table->clear();

        if ( mpi::rank() == 0 )
        {
            std::cout << "Setting up energy solve ..." << std::endl;
        }

        // "Normalize" pressure.
        const ScalarType avg_pressure_approximation =
            kernels::common::masked_sum( u.block_2().grid_data(), u.block_2().mask_data(), grid::mask_owned() ) /
            static_cast< ScalarType >( num_dofs_pressure );
        linalg::lincomb( u.block_2(), { 1.0 }, { u.block_2() }, -avg_pressure_approximation );

        // Max velocity magnitude.
        const auto max_vel = kernels::common::max_vector_magnitude( u.block_1().grid_data() );

        // Choose "suitable" small dt for accuracy - we have and implicit time-stepping scheme so we do not really need
        // a CFL in the classical sense. Still useful for time-step size restriction.
        const auto dt_advection = h / max_vel;
        const auto dt_diffusion = ( h * h ) / prm.diffusivity;
        const auto dt           = prm.pseudo_cfl * std::min( dt_advection, dt_diffusion );

        if ( mpi::rank() == 0 )
        {
            std::cout << "Computing dt ..." << std::endl;
            std::cout << "    max_vel: " << max_vel << std::endl;
            std::cout << "    h:       " << h << std::endl;
            std::cout << "=>  dt:      " << dt << std::endl;
        }

        A.dt()              = dt;
        A_neumann.dt()      = dt;
        A_neumann_diag.dt() = dt;

        // Prepping for implicit Euler step.
        linalg::apply( M_T, T, q );

        // Set up the temperature boundary.
        assign( temp_vecs["tmp_0"], 0.0 );
        Kokkos::parallel_for(
            "boundary temp interpolation",
            local_domain_md_range_policy_nodes( domains[velocity_level] ),
            InitialConditionInterpolator(
                coords_shell[velocity_level], coords_radii[velocity_level], temp_vecs["tmp_0"].grid_data(), true ) );

        Kokkos::fence();

        fe::strong_algebraic_dirichlet_enforcement_poisson_like(
            A_neumann,
            A_neumann_diag,
            temp_vecs["tmp_0"],
            temp_vecs["tmp_1"],
            q,
            mask_data[velocity_level],
            grid::shell::mask_domain_boundary() );

        if ( mpi::rank() == 0 )
        {
            std::cout << "Solving energy ..." << std::endl;
        }

        // Solve energy.
        solve( energy_solver, A, T, q );

        table->query_rows_equals( "tag", "pbicgstab_solver" ).print_pretty();
        table->clear();

        // Output stuff, logging etc.

        table->add_row( {} );

        if ( prm.xdmf )
        {
            if ( mpi::rank() == 0 )
            {
                std::cout << "Writing XDMF output and radial profiles ..." << std::endl;
            }

            xdmf_output.write();
            auto profiles = shell::radial_profiles_to_table(
                shell::radial_profiles( T ), domains[velocity_level].domain_info().radii() );
            std::ofstream out( "radial_profiles_" + std::to_string( timestep ) + ".csv" );
            profiles.print_csv( out );
        }

        simulated_time += dt;
        if ( mpi::rank() == 0 )
        {
            std::cout << "Simulated time: " << simulated_time << " (stopping at " << prm.t_end << ", we're at "
                      << simulated_time / prm.t_end * 100.0 << "%)" << std::endl;
        }
        if ( simulated_time >= prm.t_end )
        {
            break;
        }
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    util::ArgParser args( argc, argv );

    const auto table = std::make_shared< util::Table >();

    const Parameters parameters{
        .min_level                      = args.get< int >( "min-level", 0 ),
        .max_level                      = args.get< int >( "max-level", 6 ),
        .r_min                          = 0.5,
        .r_max                          = 1.0,
        .diffusivity                    = 1.0,
        .rayleigh                       = static_cast< ScalarType >( args.get< double >( "rayleigh", 1e5 ) ),
        .pseudo_cfl                     = static_cast< ScalarType >( args.get< double >( "pseudo-cfl", 0.5 ) ),
        .t_end                          = 1000.0,
        .max_timesteps                  = 1000,
        .stokes_bicgstab_l              = args.get< int >( "stokes-bicgstab-l", 2 ),
        .stokes_bicgstab_max_iterations = args.get< int >( "stokes-bicgstab-max-iterations", 10 ),
        .num_vcycles                    = args.get< int >( "num-vcycles", 2 ),
        .num_smoothing_steps_prepost    = args.get< int >( "num-smooth", 2 ),
        .xdmf                           = true };

    run( parameters, table );

    return 0;
}