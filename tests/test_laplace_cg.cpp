

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/vtk/vtk.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;

struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    SolutionInterpolator(
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
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        const double                  value  = coords( 0 ) * Kokkos::sin( coords( 1 ) ) * Kokkos::sinh( coords( 2 ) );
        // const double value = coords( 0 );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

struct SetOnBoundary
{
    Grid4DDataScalar< double > src_;
    Grid4DDataScalar< double > dst_;
    int                        num_shells_;

    SetOnBoundary( const Grid4DDataScalar< double >& src, const Grid4DDataScalar< double >& dst, const int num_shells )
    : src_( src )
    , dst_( dst )
    , num_shells_( num_shells )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_idx, const int x, const int y, const int r ) const
    {
        if ( ( r == 0 || r == num_shells_ - 1 ) )
        {
            dst_( local_subdomain_idx, x, y, r ) = src_( local_subdomain_idx, x, y, r );
        }
    }
};

double test( int level, util::Table& table )
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

    const auto domain = DistributedDomain::create_uniform_single_subdomain( level, level, 0.5, 1.0 );

    auto u        = linalg::allocate_vector_q1_scalar< ScalarType >( "u", domain, level );
    auto g        = linalg::allocate_vector_q1_scalar< ScalarType >( "g", domain, level );
    auto Adiagg   = linalg::allocate_vector_q1_scalar< ScalarType >( "Adiagg", domain, level );
    auto tmp      = linalg::allocate_vector_q1_scalar< ScalarType >( "tmp", domain, level );
    auto solution = linalg::allocate_vector_q1_scalar< ScalarType >( "solution", domain, level );
    auto error    = linalg::allocate_vector_q1_scalar< ScalarType >( "error", domain, level );
    auto b        = linalg::allocate_vector_q1_scalar< ScalarType >( "b", domain, level );
    auto r        = linalg::allocate_vector_q1_scalar< ScalarType >( "r", domain, level );

    auto mask_data      = grid::shell::allocate_scalar_grid< unsigned char >( "mask_data", domain );
    auto mask_data_long = grid::shell::allocate_scalar_grid< long >( "mask_data_double", domain );

    linalg::setup_mask_data( domain, mask_data );

    kernels::common::cast( mask_data_long, mask_data );

    const auto num_dofs = kernels::common::dot_product( mask_data_long, mask_data_long );

    u.add_mask_data( mask_data, level );
    g.add_mask_data( mask_data, level );
    Adiagg.add_mask_data( mask_data, level );
    tmp.add_mask_data( mask_data, level );
    solution.add_mask_data( mask_data, level );
    error.add_mask_data( mask_data, level );
    b.add_mask_data( mask_data, level );
    r.add_mask_data( mask_data, level );

    const auto subdomain_shell_coords = terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain );
    const auto subdomain_radii        = terra::grid::shell::subdomain_shell_radii( domain );

    using Laplace = fe::wedge::operators::shell::Laplace< ScalarType >;

    Laplace A( domain, subdomain_shell_coords, subdomain_radii, true, false );
    Laplace A_neumann( domain, subdomain_shell_coords, subdomain_radii, false, false );
    Laplace A_neumann_diag( domain, subdomain_shell_coords, subdomain_radii, false, true );
    Laplace A_diag( domain, subdomain_shell_coords, subdomain_radii, true, true );

    // Set up solution data.
    Kokkos::parallel_for(
        "solution interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, solution.grid_data( level ), false ) );

    // Set up boundary data.
    Kokkos::parallel_for(
        "boundary interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( subdomain_shell_coords, subdomain_radii, g.grid_data( level ), true ) );

    linalg::apply( A_neumann_diag, g, Adiagg, level );
    linalg::apply( A_neumann, g, b, level );

    linalg::lincomb( b, { -1.0 }, { b }, level );

    Kokkos::parallel_for(
        "set on boundary",
        grid::shell::local_domain_md_range_policy_nodes( domain ),
        SetOnBoundary(
            Adiagg.grid_data( level ), b.grid_data( level ), domain.domain_info().subdomain_num_nodes_radially() ) );

    linalg::solvers::IterativeSolverParameters solver_params{ 100, 1e-12, 1e-12 };

    linalg::solvers::PCG< Laplace > pcg( solver_params, tmp, Adiagg, error, r );

    Kokkos::fence();
    timer.reset();
    linalg::solvers::solve( pcg, A, u, b, level );
    Kokkos::fence();
    const auto time_solver = timer.seconds();

    linalg::lincomb( error, { 1.0, -1.0 }, { u, solution }, level );
    const auto l2_error = std::sqrt( dot( error, error, level ) / num_dofs );

    if ( false )
    {
        vtk::VTKOutput vtk_after(
            subdomain_shell_coords, subdomain_radii, "laplace_cg_level" + std::to_string( level ) + ".vtu", false );
        vtk_after.add_scalar_field( g.grid_data( level ).label(), g.grid_data( level ) );
        vtk_after.add_scalar_field( u.grid_data( level ).label(), u.grid_data( level ) );
        vtk_after.add_scalar_field( solution.grid_data( level ).label(), solution.grid_data( level ) );
        vtk_after.add_scalar_field( error.grid_data( level ).label(), error.grid_data( level ) );

        vtk_after.write();
    }

    table.add_row(
        { { "level", level }, { "dofs", num_dofs }, { "l2_error", l2_error }, { "time_solver", time_solver } } );

    return l2_error;
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    util::Table table;

    double prev_l2_error = 1.0;

    for ( int level = 0; level < 6; ++level )
    {
        Kokkos::Timer timer;
        timer.reset();
        double     l2_error   = test( level, table );
        const auto time_total = timer.seconds();
        table.add_row( { { "level", level }, { "time_total", time_total } } );

        if ( level > 0 )
        {
            table.add_row( { { "level", level }, { "order", prev_l2_error / l2_error } } );
        }
        prev_l2_error = l2_error;
    }

    table.print_pretty( table.query_not_none( "order" ) );
    table.print_pretty( table.query_not_none( "dofs" ) );

    MPI_Finalize();
    return 0;
}