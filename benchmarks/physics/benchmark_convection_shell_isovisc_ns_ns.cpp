

#include "communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/mass.hpp"
#include "fe/wedge/operators/shell/prolongation_constant.hpp"
#include "fe/wedge/operators/shell/restriction_constant.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
#include "fe/wedge/operators/shell/vector_laplace.hpp"
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
#include "util/init.hpp"
#include "util/table.hpp"
#include "vtk/vtk.hpp"

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

struct Parameters
{
    int min_level;
    int max_level;

    double r_min;
    double r_max;

    double diffusivity;
    double rayleigh;

    double dt;
    double t_end;
    int    max_timesteps;

    int num_vcycles;
    int num_smoothing_steps_prepost;

    bool vtk;
};

struct InitialConditionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    InitialConditionInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data,
        bool                              only_boundary )
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
            const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

            data_( local_subdomain_id, x, y, r ) = 2.0 * ( 1.0 - coords.norm() );
        }
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_u_;
    Grid4DDataScalar< double > data_T_;
    double                     rayleigh_number_;

    RHSVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data_u,
        const Grid4DDataScalar< double >& data_T,
        double                            rayleigh_number )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , data_T_( data_T )
    , rayleigh_number_( rayleigh_number )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

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
    using ScalarType = double;

    // Set up domains for all levels.

    std::vector< DistributedDomain >                  domains;
    std::vector< Grid3DDataVec< double, 3 > >         coords_shell;
    std::vector< Grid2DDataScalar< double > >         coords_radii;
    std::vector< Grid4DDataScalar< util::MaskType > > mask_data;

    for ( int level = prm.min_level; level <= prm.max_level; level++ )
    {
        const int idx = level - prm.min_level;

        domains.push_back( DistributedDomain::create_uniform_single_subdomain( level, level, prm.r_min, prm.r_max ) );
        coords_shell.push_back( grid::shell::subdomain_unit_sphere_single_shell_coords( domains[idx] ) );
        coords_radii.push_back( grid::shell::subdomain_shell_radii( domains[idx] ) );
        mask_data.push_back( linalg::setup_mask_data( domains[idx] ) );
    }

    const auto num_levels     = domains.size();
    const auto velocity_level = num_levels - 1;
    const auto pressure_level = num_levels - 2;

    // Set up Stokes vectors for the finest grid.

    std::map< std::string, VectorQ1IsoQ2Q1< ScalarType > > stok_vecs;
    std::vector< std::string >                             stok_vec_names = { "u", "f" };
    constexpr int                                          num_stok_tmps  = 8;

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
        linalg::assign( stok_vecs["tmp_3"].block_1(), 1.0 );
        linalg::apply( A_diag[level], stok_vecs["tmp_3"].block_1(), inverse_diagonals.back() );
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
        linalg::solvers::IterativeSolverParameters{ 1000, 1e-8, 1e-16 }, table, coarse_grid_tmps );

    using PrecVisc = linalg::solvers::Multigrid< Viscous, Prolongation, Restriction, Smoother, CoarseGridSolver >;
    PrecVisc prec_11(
        P, R, A_c, tmp_mg_r, tmp_mg_e, tmp_mg, smoothers, smoothers, coarse_grid_solver, prm.num_vcycles, 1e-8 );

    using PrecSchur = linalg::solvers::IdentitySolver< Stokes::Block22Type >;
    PrecSchur prec_22;

    using PrecStokes = linalg::solvers::BlockDiagonalPreconditioner2x2< Stokes, PrecVisc, PrecSchur >;
    PrecStokes prec_stokes( prec_11, prec_22 );

    linalg::solvers::IterativeSolverParameters solver_params{ 100, 1e-8, 1e-12 };

    std::vector< VectorQ1IsoQ2Q1< ScalarType > > tmp_bicgstab( 8 );
    for ( int i = 0; i < 8; i++ )
    {
        tmp_bicgstab[i] = stok_vecs["tmp_" + std::to_string( i )];
    }

    linalg::solvers::PBiCGStab< Stokes, PrecStokes > pbicgstab( 2, solver_params, table, tmp_bicgstab, prec_stokes );

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
        prm.dt,
        true,
        false,
        mass_scaling );

    AD A_neumann(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        u.block_1(),
        prm.diffusivity,
        prm.dt,
        false,
        false,
        mass_scaling );

    AD A_neumann_diag(
        domains[velocity_level],
        coords_shell[velocity_level],
        coords_radii[velocity_level],
        u.block_1(),
        prm.diffusivity,
        prm.dt,
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
        2, linalg::solvers::IterativeSolverParameters{ 100, 1e-12, 1e-12 }, table, tmp_bicgstab_temp );

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

    vtk::VTKOutput vtk_fine( coords_shell[velocity_level], coords_radii[velocity_level], false );
    vtk::VTKOutput vtk_coarse( coords_shell[pressure_level], coords_radii[pressure_level], false );

    vtk_fine.add_vector_field( u.block_1().grid_data() );
    vtk_coarse.add_scalar_field( u.block_2().grid_data() );

    vtk_fine.add_vector_field( f.block_1().grid_data() );
    vtk_coarse.add_scalar_field( f.block_2().grid_data() );

    vtk_fine.add_scalar_field( T.grid_data() );

    // Time stepping

    if ( prm.vtk )
    {
        vtk_fine.write(
            "benchmark_convection_shell_isovisc_ns_ns_velocity_temperature_" + std::to_string( 0 ) + ".vtu" );
        vtk_coarse.write( "benchmark_convection_shell_isovisc_ns_ns_pressure_" + std::to_string( 0 ) + ".vtu" );
    }

    double simulated_time = 0.0;
    for ( int timestep = 1; timestep < prm.max_timesteps; timestep++ )
    {
        std::cout << "Timestep " << timestep << std::endl;

        // Set up rhs data for Stokes.

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

        // Solve Stokes.
        solve( pbicgstab, K, u, f );

        // assign( u, 0.0 );

        table->query_rows_equals( "tag", "pbicgstab_solver" ).print_pretty();
        table->clear();

        // "Normalize" pressure.
        const double avg_pressure_approximation =
            kernels::common::masked_sum( u.block_2().grid_data(), u.block_2().mask_data(), grid::mask_owned() ) /
            num_dofs_pressure;
        linalg::lincomb( u.block_2(), { 1.0 }, { u.block_2() }, -avg_pressure_approximation );

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

        // Solve energy.
        solve( energy_solver, A, T, q );

        table->query_rows_equals( "tag", "pbicgstab_solver" ).print_pretty();
        table->clear();

        // Output stuff, logging etc.

        table->add_row( {} );

        if ( prm.vtk )
        {
            vtk_fine.write(
                "benchmark_convection_shell_isovisc_ns_ns_velocity_temperature_" + std::to_string( timestep ) +
                ".vtu" );
            vtk_coarse.write(
                "benchmark_convection_shell_isovisc_ns_ns_pressure_" + std::to_string( timestep ) + ".vtu" );
        }

        simulated_time += prm.dt;
        if ( simulated_time >= prm.t_end )
        {
            break;
        }
    }
}

int main( int argc, char** argv )
{
    util::TerraScopeGuard scope_guard( &argc, &argv );

    const auto table = std::make_shared< util::Table >();

    constexpr Parameters parameters{
        .min_level                   = 0,
        .max_level                   = 4,
        .r_min                       = 0.5,
        .r_max                       = 1.0,
        .diffusivity                 = 1e-2,
        .rayleigh                    = 1e4,
        .dt                          = 1e-1,
        .t_end                       = 1000.0,
        .max_timesteps               = 1000,
        .num_vcycles                 = 1,
        .num_smoothing_steps_prepost = 1,
        .vtk                         = true };

    run( parameters, table );

    return 0;
}