

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "linalg/solvers/pbicgstab.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/fe/wedge/operators/shell/unsteady_advection_diffusion_supg.hpp"
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

struct VelocityInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataVec< double, 3 > data_;
    bool                       only_boundary_;

    VelocityInterpolator(
        const Grid3DDataVec< double, 3 >& grid,
        const Grid2DDataScalar< double >& radii,
        const Grid4DDataVec< double, 3 >& data,
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

        data_( local_subdomain_id, x, y, r, 0 ) = -coords( 1 );
        data_( local_subdomain_id, x, y, r, 1 ) = coords( 0 );
        data_( local_subdomain_id, x, y, r, 2 ) = 0.0;
    }
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
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );

        const dense::Vec< double, 3 > center{ 0.75, 0.0, 0.0 };
        const double                  radius = 0.1;

        if ( ( coords - center ).norm() < radius )
        {
            data_( local_subdomain_id, x, y, r ) = 1.0;
        }
    }
};

void test( int level, util::Table& table )
{
    Kokkos::Timer timer;

    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform_single_subdomain( level, level, 0.5, 1.0 );

    auto T = linalg::allocate_vector_q1_scalar< ScalarType >( "T", domain, level );
    auto u = linalg::allocate_vector_q1_vec< ScalarType, 3 >( "u", domain, level );
    auto f = linalg::allocate_vector_q1_scalar< ScalarType >( "f", domain, level );

    auto mask_data = grid::shell::allocate_scalar_grid< unsigned char >( "mask_data", domain );

    linalg::setup_mask_data( domain, mask_data );

    T.add_mask_data( mask_data, level );
    u.add_mask_data( mask_data, level );

    std::vector< linalg::VectorQ1Scalar< double > > tmps;
    for ( int i = 0; i < 8; ++i )
    {
        tmps.emplace_back( linalg::allocate_vector_q1_scalar< ScalarType >( "tmpp", domain, level ) );
        tmps[i].add_mask_data( mask_data, level );
    }

    linalg::assign( tmps[0], 1.0, level );
    const auto num_dofs = linalg::dot( tmps[0], tmps[0], level );
    std::cout << "Number of dofs: " << num_dofs << std::endl;

    const auto subdomain_shell_coords = terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain );
    const auto subdomain_radii        = terra::grid::shell::subdomain_shell_radii( domain );

    using AD = fe::wedge::operators::shell::UnsteadyAdvectionDiffusionSUPG< ScalarType >;

    AD A( domain, subdomain_shell_coords, subdomain_radii, u, 1e-3, 1e-2, false, false );

    using Mass = fe::wedge::operators::shell::Mass< ScalarType >;

    Mass M( domain, subdomain_shell_coords, subdomain_radii, false );

    // Set up solution data.
    Kokkos::parallel_for(
        "velocity interpolation",
        local_domain_md_range_policy_nodes( domain ),
        VelocityInterpolator( subdomain_shell_coords, subdomain_radii, u.grid_data( level ), false ) );

    Kokkos::fence();

    // Set up the initial temperature.
    Kokkos::parallel_for(
        "initial temp interpolation",
        local_domain_md_range_policy_nodes( domain ),
        InitialConditionInterpolator( subdomain_shell_coords, subdomain_radii, T.grid_data( level ), false ) );

    Kokkos::fence();

    linalg::solvers::IterativeSolverParameters solver_params{ 1000, 1e-12, 1e-12 };

    linalg::solvers::PBiCGStab< AD > bicgstab( 2, solver_params, tmps );
    bicgstab.set_tag( "bicgstab_solver_level_" + std::to_string( level ) );

    const int timesteps = 10;

    if ( true )
    {
        vtk::VTKOutput vtk_after(
            subdomain_shell_coords,
            subdomain_radii,
            "advection_diffusion_" + std::to_string( level ) + "_ts_" + std::to_string( 0 ) + ".vtu",
            false );
        vtk_after.add_scalar_field( T.grid_data( level ) );
        vtk_after.add_vector_field( u.grid_data( level ) );
        vtk_after.write();
    }

    for ( int ts = 1; ts < timesteps; ++ts )
    {
        std::cout << "Timestep " << ts << std::endl;

        linalg::apply( M, T, f, level );
        linalg::solvers::solve( bicgstab, A, T, f, level, table );

        table.print_pretty();
        table.clear();

        if ( true )
        {
            vtk::VTKOutput vtk_after(
                subdomain_shell_coords,
                subdomain_radii,
                "advection_diffusion_" + std::to_string( level ) + "_ts_" + std::to_string( ts ) + ".vtu",
                false );
            vtk_after.add_scalar_field( T.grid_data( level ) );
            vtk_after.add_vector_field( u.grid_data( level ) );
            vtk_after.write();
        }
    }
}

int main( int argc, char** argv )
{
    util::TerraScopeGuard scope_guard( &argc, &argv );

    util::Table table;

    const int level = 4;

    test( level, table );

    return 0;
}