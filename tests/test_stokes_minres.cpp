

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/pminres.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/vtk/vtk.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;

#define SOLUTION_TYPE 1

struct SolutionVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_u_;
    bool                       only_boundary_;

    SolutionVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data_u,
        const bool                        only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        dense::Vec< double, 3 > u;

        if ( SOLUTION_TYPE == 0 )
        {
            u( 0 ) = Kokkos::sin( cy );
            u( 1 ) = Kokkos::sin( cz );
            u( 2 ) = Kokkos::sin( cx );
        }

        else if ( SOLUTION_TYPE == 1 )
        {
            u( 0 ) = -4 * Kokkos::cos( 4 * cz );
            u( 1 ) = 8 * Kokkos::cos( 8 * cx );
            u( 2 ) = -2 * Kokkos::cos( 2 * cy );
        }

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            for ( int d = 0; d < 3; d++ )
            {
                data_u_( local_subdomain_id, x, y, r, d ) = u( d );
            }
        }
    }
};

struct SolutionPressureInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_p_;
    bool                       only_boundary_;

    SolutionPressureInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataScalar< double >& data_p,
        const bool                        only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_p_( data_p )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        double p = 0.0;

        if ( SOLUTION_TYPE == 0 )
        {
            p = 0;
        }

        else if ( SOLUTION_TYPE == 1 )
        {
            p = Kokkos::sin( 4 * cx ) * Kokkos::sin( 8 * cy ) * Kokkos::sin( 2 * cz );
        }

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_p_( local_subdomain_id, x, y, r ) = p;
        }
    }
};

struct RHSVelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_u_;

    RHSVelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data_u )
    : grid_( grid )
    , radii_( radii )
    , data_u_( data_u )

    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const double cx = coords( 0 );
        const double cy = coords( 1 );
        const double cz = coords( 2 );

        dense::Vec< double, 3 > u;

        if ( SOLUTION_TYPE == 0 )
        {
            u( 0 ) = Kokkos::sin( cy );
            u( 1 ) = Kokkos::sin( cz );
            u( 2 ) = Kokkos::sin( cx );
        }

        else if ( SOLUTION_TYPE == 1 )
        {
            u( 0 ) =
                4 * Kokkos::sin( 8 * cy ) * Kokkos::sin( 2 * cz ) * Kokkos::cos( 4 * cx ) - 64 * Kokkos::cos( 4 * cz );
            u( 1 ) =
                8 * Kokkos::sin( 4 * cx ) * Kokkos::sin( 2 * cz ) * Kokkos::cos( 8 * cy ) + 512 * Kokkos::cos( 8 * cx );
            u( 2 ) =
                2 * Kokkos::sin( 4 * cx ) * Kokkos::sin( 8 * cy ) * Kokkos::cos( 2 * cz ) - 8 * Kokkos::cos( 2 * cy );
        }

        for ( int d = 0; d < 3; d++ )
        {
            data_u_( local_subdomain_id, x, y, r, d ) = u( d );
        }
    }
};

struct SetOnBoundary
{
    Grid4DDataVec< double, 3 > src_;
    Grid4DDataVec< double, 3 > dst_;
    int                        num_shells_;

    SetOnBoundary( const Grid4DDataVec< double, 3 >& src, const Grid4DDataVec< double, 3 >& dst, const int num_shells )
    : src_( src )
    , dst_( dst )
    , num_shells_( num_shells )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_idx, const int x, const int y, const int r ) const
    {
        if ( ( r == 0 || r == num_shells_ - 1 ) )
        {
            for ( int d = 0; d < 3; ++d )
            {
                dst_( local_subdomain_idx, x, y, r, d ) = src_( local_subdomain_idx, x, y, r, d );
            }
        }
    }
};

std::pair< double, double > test( int level, util::Table& table )
{
    /**

    Boundary handling notes.

    Using inhom boundary conditions we approach the elimination as follows (for the moment).

    Let A be the "Neumann" operator, i.e., we do not treat the boundaries any differently.

    1. Interpolate Dirichlet boundary conditions into g.
    2. Compute g_A <- A       * g.
    3. Compute g_D <- diag(A) * g.
    4. Set the rhs to b = f - g_A.
    5. Set the rhs at the boundary nodes to g_D.
    6. Solve
            A_elim x = b
       where A_elim is A but with all off-diagonal entries in the same row/col as a boundary node set to zero.
       In a matrix-free context, we have to adapt the element matrix A_local accordingly by (symmetrically) zeroing
       out all the off-diagonals (row and col) that correspond to a boundary node. But we keep the diagonal intact.
       We still have diag(A) == diag(A_elim).
    7. x is the solution of the original problem. No boundary correction should be necessary.

    **/

    Kokkos::Timer timer;

    using ScalarType = double;

    const auto domain_fine   = DistributedDomain::create_uniform_single_subdomain( level, level, 0.5, 1.0 );
    const auto domain_coarse = DistributedDomain::create_uniform_single_subdomain( level - 1, level - 1, 0.5, 1.0 );

    const auto subdomain_fine_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain_fine );
    const auto subdomain_fine_radii = terra::grid::shell::subdomain_shell_radii( domain_fine );

    const auto subdomain_coarse_shell_coords =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain_coarse );
    const auto subdomain_coarse_radii = terra::grid::shell::subdomain_shell_radii( domain_coarse );

    // K w = b

    auto w      = linalg::allocate_vector_q1isoq2_q1< ScalarType >( "w", domain_fine, domain_coarse, level );
    auto g      = linalg::allocate_vector_q1isoq2_q1< ScalarType, 3 >( "g", domain_fine, domain_coarse, level );
    auto Kdiagg = linalg::allocate_vector_q1isoq2_q1< ScalarType, 3 >( "Adiagg", domain_fine, domain_coarse, level );
    auto tmp    = linalg::allocate_vector_q1isoq2_q1< ScalarType, 3 >( "tmp", domain_fine, domain_coarse, level );
    auto solution =
        linalg::allocate_vector_q1isoq2_q1< ScalarType, 3 >( "solution", domain_fine, domain_coarse, level );
    auto error = linalg::allocate_vector_q1isoq2_q1< ScalarType, 3 >( "error", domain_fine, domain_coarse, level );
    auto b     = linalg::allocate_vector_q1isoq2_q1< ScalarType, 3 >( "b", domain_fine, domain_coarse, level );
    auto r     = linalg::allocate_vector_q1isoq2_q1< ScalarType, 3 >( "r", domain_fine, domain_coarse, level );

    auto tmp_0 = linalg::allocate_vector_q1isoq2_q1< ScalarType >( "tmp_0", domain_fine, domain_coarse, level );
    auto tmp_1 = linalg::allocate_vector_q1isoq2_q1< ScalarType >( "tmp_1", domain_fine, domain_coarse, level );
    auto tmp_2 = linalg::allocate_vector_q1isoq2_q1< ScalarType >( "tmp_2", domain_fine, domain_coarse, level );
    auto tmp_3 = linalg::allocate_vector_q1isoq2_q1< ScalarType >( "tmp_3", domain_fine, domain_coarse, level );
    auto tmp_4 = linalg::allocate_vector_q1isoq2_q1< ScalarType >( "tmp_4", domain_fine, domain_coarse, level );
    auto tmp_5 = linalg::allocate_vector_q1isoq2_q1< ScalarType >( "tmp_5", domain_fine, domain_coarse, level );
    auto tmp_6 = linalg::allocate_vector_q1isoq2_q1< ScalarType >( "tmp_6", domain_fine, domain_coarse, level );

    auto mask_data_fine      = grid::shell::allocate_scalar_grid< unsigned char >( "mask_data", domain_fine );
    auto mask_data_fine_long = grid::shell::allocate_scalar_grid< long >( "mask_data_double", domain_fine );

    auto mask_data_coarse      = grid::shell::allocate_scalar_grid< unsigned char >( "mask_data", domain_coarse );
    auto mask_data_coarse_long = grid::shell::allocate_scalar_grid< long >( "mask_data_double", domain_coarse );

    linalg::setup_mask_data( domain_fine, mask_data_fine );
    linalg::setup_mask_data( domain_coarse, mask_data_coarse );

    kernels::common::cast( mask_data_fine_long, mask_data_fine );
    kernels::common::cast( mask_data_coarse_long, mask_data_coarse );

    const auto num_dofs_velocity = 3 * kernels::common::dot_product( mask_data_fine_long, mask_data_fine_long );
    const auto num_dofs_pressure = kernels::common::dot_product( mask_data_coarse_long, mask_data_coarse_long );

    w.add_mask_data( mask_data_fine, mask_data_coarse, level );
    b.add_mask_data( mask_data_fine, mask_data_coarse, level );
    Kdiagg.add_mask_data( mask_data_fine, mask_data_coarse, level );
    tmp.add_mask_data( mask_data_fine, mask_data_coarse, level );
    solution.add_mask_data( mask_data_fine, mask_data_coarse, level );
    error.add_mask_data( mask_data_fine, mask_data_coarse, level );
    b.add_mask_data( mask_data_fine, mask_data_coarse, level );
    r.add_mask_data( mask_data_fine, mask_data_coarse, level );

    tmp_0.add_mask_data( mask_data_fine, mask_data_coarse, level );
    tmp_1.add_mask_data( mask_data_fine, mask_data_coarse, level );
    tmp_2.add_mask_data( mask_data_fine, mask_data_coarse, level );
    tmp_3.add_mask_data( mask_data_fine, mask_data_coarse, level );
    tmp_4.add_mask_data( mask_data_fine, mask_data_coarse, level );
    tmp_5.add_mask_data( mask_data_fine, mask_data_coarse, level );
    tmp_6.add_mask_data( mask_data_fine, mask_data_coarse, level );

    using Stokes = fe::wedge::operators::shell::Stokes< ScalarType >;

    Stokes K( domain_fine, domain_coarse, subdomain_fine_shell_coords, subdomain_fine_radii, true, false );
    Stokes K_neumann( domain_fine, domain_coarse, subdomain_fine_shell_coords, subdomain_fine_radii, false, false );
    Stokes K_neumann_diag( domain_fine, domain_coarse, subdomain_fine_shell_coords, subdomain_fine_radii, false, true );

    using Mass = fe::wedge::operators::shell::VectorMass< ScalarType, 3 >;

    Mass M( domain_fine, subdomain_fine_shell_coords, subdomain_fine_radii, false );

    // Set up solution data.
    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain_fine ),
        SolutionVelocityInterpolator(
            subdomain_fine_shell_coords, subdomain_fine_radii, solution.block_1().grid_data( level ), false ) );

    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain_coarse ),
        SolutionPressureInterpolator(
            subdomain_coarse_shell_coords, subdomain_coarse_radii, solution.block_2().grid_data( level ), false ) );

    // Set up boundary data.
    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domain_fine ),
        SolutionVelocityInterpolator(
            subdomain_fine_shell_coords, subdomain_fine_radii, g.block_1().grid_data( level ), true ) );

    // Set up rhs data.
    Kokkos::parallel_for(
        "rhs interpolation",
        local_domain_md_range_policy_nodes( domain_fine ),
        RHSVelocityInterpolator(
            subdomain_fine_shell_coords, subdomain_fine_radii, tmp.block_1().grid_data( level ) ) );

    linalg::apply( M, tmp.block_1(), b.block_1(), level );

    linalg::apply( K_neumann_diag, g, Kdiagg, level );
    linalg::apply( K_neumann, g, tmp, level );

    linalg::lincomb( b, { 1.0, -1.0 }, { b, tmp }, level );

    Kokkos::parallel_for(
        "set on boundary",
        grid::shell::local_domain_md_range_policy_nodes( domain_fine ),
        SetOnBoundary(
            Kdiagg.block_1().grid_data( level ),
            b.block_1().grid_data( level ),
            domain_fine.domain_info().subdomain_num_nodes_radially() ) );

    linalg::solvers::IterativeSolverParameters solver_params{ 3000, 1e-8, 1e-12 };

    linalg::solvers::PMINRES< Stokes > pminres( solver_params, tmp_0, tmp_1, tmp_2, tmp_3, tmp_4, tmp_5, tmp_6 );
    pminres.set_tag( "pminres_solver_level_" + std::to_string( level ) );

    solve( pminres, K, w, b, level, table );

    const double avg_pressure_solution =
        kernels::common::masked_sum(
            solution.block_2().grid_data( level ), solution.block_2().mask_data( level ), grid::shell::mask_owned() ) /
        num_dofs_pressure;
    const double avg_pressure_approximation =
        kernels::common::masked_sum(
            w.block_2().grid_data( level ), w.block_2().mask_data( level ), grid::shell::mask_owned() ) /
        num_dofs_pressure;

    linalg::lincomb( solution.block_2(), { 1.0 }, { solution.block_2() }, -avg_pressure_solution, level );
    linalg::lincomb( w.block_2(), { 1.0 }, { w.block_2() }, -avg_pressure_approximation, level );

    linalg::apply( K, w, tmp_6, level );
    linalg::lincomb( r, { 1.0, -1.0 }, { b, tmp_6 }, level );
    const auto inf_residual_vel = linalg::inf_norm( r.block_1(), level );
    const auto inf_residual_pre = linalg::inf_norm( r.block_2(), level );

    linalg::lincomb( error, { 1.0, -1.0 }, { w, solution }, level );
    const auto l2_error_velocity =
        std::sqrt( dot( error.block_1(), error.block_1(), level ) / static_cast< double >( num_dofs_velocity ) );
    const auto l2_error_pressure =
        std::sqrt( dot( error.block_2(), error.block_2(), level ) / static_cast< double >( num_dofs_pressure ) );

    table.add_row(
        { { "level", level },
          { "dofs_vel", num_dofs_velocity },
          { "l2_error_vel", l2_error_velocity },
          { "dofs_pre", num_dofs_pressure },
          { "l2_error_pre", l2_error_pressure },
          { "inf_res_vel", inf_residual_vel },
          { "inf_res_pre", inf_residual_pre } } );

    if ( false )
    {
        vtk::VTKOutput vtk_fine(
            subdomain_fine_shell_coords,
            subdomain_fine_radii,
            "test_stokes_minres_fine_" + std::to_string( level ) + ".vtu",
            false );

        vtk::VTKOutput vtk_coarse(
            subdomain_coarse_shell_coords,
            subdomain_coarse_radii,
            "test_stokes_minres_coarse_" + std::to_string( level ) + ".vtu",
            false );

        vtk_fine.add_vector_field( w.block_1().grid_data( level ) );
        vtk_coarse.add_scalar_field( w.block_2().grid_data( level ) );

        vtk_fine.add_vector_field( solution.block_1().grid_data( level ) );
        vtk_coarse.add_scalar_field( solution.block_2().grid_data( level ) );

        vtk_fine.add_vector_field( error.block_1().grid_data( level ) );
        vtk_coarse.add_scalar_field( error.block_2().grid_data( level ) );

        vtk_fine.add_vector_field( b.block_1().grid_data( level ) );
        vtk_coarse.add_scalar_field( b.block_2().grid_data( level ) );

        vtk_fine.add_vector_field( r.block_1().grid_data( level ) );
        vtk_coarse.add_scalar_field( r.block_2().grid_data( level ) );

        vtk_fine.write();
        vtk_coarse.write();
    }

    return { l2_error_velocity, l2_error_pressure };
}

int main( int argc, char** argv )
{
    util::TerraScopeGuard scope_guard( &argc, &argv );

    util::Table table( false );

    double prev_l2_error_vel = 1.0;
    double prev_l2_error_pre = 1.0;

    for ( int level = 1; level < 5; ++level )
    {
        std::cout << "level = " << level << std::endl;
        Kokkos::Timer timer;
        timer.reset();
        const auto [l2_error_vel, l2_error_pre] = test( level, table );
        const auto time_total                   = timer.seconds();
        table.add_row( { { "level", level }, { "time_total", time_total } } );

        if ( level > 2 )
        {
            const double order_vel = prev_l2_error_vel / l2_error_vel;
            const double order_pre = prev_l2_error_pre / l2_error_pre;

            std::cout << "order_vel = " << order_vel << std::endl;
            std::cout << "order_pre = " << order_pre << std::endl;

            if ( order_vel < 3.8 )
            {
                return EXIT_FAILURE;
            }

            if ( order_vel < 3.3 )
            {
                return EXIT_FAILURE;
            }

            table.add_row( { { "level", level }, { "order_vel", order_vel }, { "order_pre", order_pre } } );
        }
        prev_l2_error_vel = l2_error_vel;
        prev_l2_error_pre = l2_error_pre;
    }

    table.query_not_none( "order_vel" ).select( { "level", "order_vel", "order_pre" } ).print_pretty();
    table.query_not_none( "dofs_vel" ).select( { "level", "dofs_vel", "l2_error_vel", "l2_error_pre" } ).print_pretty();

    return 0;
}