
#include <kernels/common/grid_operations.hpp>

#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/stokes.hpp"
#include "fe/wedge/operators/shell/vector_laplace_simple.hpp"
#include "linalg/operator.hpp"
#include "linalg/vector.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/util/table_printer.hpp"
#include "util/info.hpp"
#include "util/table.hpp"

using namespace terra;

using fe::wedge::operators::shell::Laplace;
using fe::wedge::operators::shell::Stokes;
using fe::wedge::operators::shell::VectorLaplaceSimple;
using linalg::apply;
using linalg::DstOf;
using linalg::OperatorLike;
using linalg::SrcOf;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;

enum class BenchmarkType : int
{
    LaplaceFloat,
    LaplaceDouble,
    VectorLaplaceFloat,
    VectorLaplaceDouble,
    VectorLaplaceNeumannDouble,
    StokesDouble,
};

constexpr auto all_benchmark_types = {
    BenchmarkType::LaplaceFloat,
    BenchmarkType::LaplaceDouble,
    BenchmarkType::VectorLaplaceFloat,
    BenchmarkType::VectorLaplaceDouble,
    BenchmarkType::VectorLaplaceNeumannDouble,
    BenchmarkType::StokesDouble };

const std::map< BenchmarkType, std::string > benchmark_description = {
    { BenchmarkType::LaplaceFloat, "Laplace (float)" },
    { BenchmarkType::LaplaceDouble, "Laplace (double)" },
    { BenchmarkType::VectorLaplaceFloat, "VectorLaplace (float)" },
    { BenchmarkType::VectorLaplaceDouble, "VectorLaplace (double)" },
    { BenchmarkType::VectorLaplaceNeumannDouble, "VectorLaplaceNeumann (double)" },
    { BenchmarkType::StokesDouble, "Stokes (double)" } };

struct BenchmarkData
{
    int    level;
    long   dofs;
    double duration;
};

template < OperatorLike OperatorT >
double measure_run_time( int executions, OperatorT& A, const SrcOf< OperatorT >& src, DstOf< OperatorT >& dst )
{
    Kokkos::Timer timer;

    Kokkos::fence();
    timer.reset();

    for ( int i = 0; i < executions; ++i )
    {
        apply( A, src, dst );
    }

    Kokkos::fence();

    // Ensure stuff is not optimized out?!
    // const auto mm = kernels::common::max_abs_entry( dst.grid_data() );
    // std::cout << "Printing some derived value to ensure nothing is optimized out: " << mm << std::endl;

    const double duration = timer.seconds() / executions;
    return duration;
}

BenchmarkData run( const BenchmarkType benchmark, const int level, const int executions )
{
    if ( level < 1 )
    {
        Kokkos::abort( "level must be >= 1" );
    }

    const auto domain = grid::shell::DistributedDomain::create_uniform_single_subdomain(
        level, level, 0.5, 1.0, grid::shell::subdomain_to_rank_distribute_full_diamonds );

    const auto domain_coarse = grid::shell::DistributedDomain::create_uniform_single_subdomain(
        level - 1, level - 1, 0.5, 1.0, grid::shell::subdomain_to_rank_distribute_full_diamonds );

    const auto coords_shell_double = grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain );
    const auto coords_radii_double = grid::shell::subdomain_shell_radii< double >( domain );

    const auto coords_shell_float = grid::shell::subdomain_unit_sphere_single_shell_coords< float >( domain );
    const auto coords_radii_float = grid::shell::subdomain_shell_radii< float >( domain );

    const auto coords_shell_coarse_double =
        grid::shell::subdomain_unit_sphere_single_shell_coords< double >( domain_coarse );
    const auto coords_radii_coarse_double = grid::shell::subdomain_shell_radii< double >( domain_coarse );

    const auto coords_shell_coarse_float =
        grid::shell::subdomain_unit_sphere_single_shell_coords< float >( domain_coarse );
    const auto coords_radii_coarse_float = grid::shell::subdomain_shell_radii< float >( domain_coarse );

    auto mask_data        = linalg::setup_mask_data( domain );
    auto mask_data_coarse = linalg::setup_mask_data( domain_coarse );

    const auto dofs_scalar        = kernels::common::count_masked< long >( mask_data, grid::mask_owned() );
    const auto dofs_vec           = 3 * dofs_scalar;
    const auto dofs_scalar_coarse = kernels::common::count_masked< long >( mask_data_coarse, grid::mask_owned() );
    const auto dofs_stokes        = dofs_vec + dofs_scalar_coarse;

    VectorQ1Scalar< double > src_scalar_double( "src_scalar_double", domain, mask_data );
    VectorQ1Scalar< double > dst_scalar_double( "dst_scalar_double", domain, mask_data );

    VectorQ1Scalar< float > src_scalar_float( "src_scalar_float", domain, mask_data );
    VectorQ1Scalar< float > dst_scalar_float( "dst_scalar_float", domain, mask_data );

    VectorQ1Vec< double > src_vec_double( "src_vec_double", domain, mask_data );
    VectorQ1Vec< double > dst_vec_double( "dst_vec_double", domain, mask_data );

    VectorQ1Vec< float > src_vec_float( "src_vec_float", domain, mask_data );
    VectorQ1Vec< float > dst_vec_float( "dst_vec_float", domain, mask_data );

    VectorQ1IsoQ2Q1< double > src_stokes_double(
        "src_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );
    VectorQ1IsoQ2Q1< double > dst_stokes_double(
        "dst_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );

    VectorQ1IsoQ2Q1< float > src_stokes_float(
        "src_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );
    VectorQ1IsoQ2Q1< float > dst_stokes_float(
        "dst_stokes_double", domain, domain_coarse, mask_data, mask_data_coarse );

    linalg::randomize( src_scalar_double );
    linalg::randomize( src_scalar_float );
    linalg::randomize( src_vec_double );
    linalg::randomize( src_vec_float );
    linalg::randomize( src_stokes_double );
    linalg::randomize( src_stokes_float );

    double duration = 0.0;
    long   dofs     = 0;

    if ( benchmark == BenchmarkType::LaplaceFloat )
    {
        Laplace< float > A( domain, coords_shell_float, coords_radii_float, true, false );
        duration = measure_run_time( executions, A, src_scalar_float, dst_scalar_float );
        dofs     = dofs_scalar;
    }
    else if ( benchmark == BenchmarkType::LaplaceDouble )
    {
        Laplace< double > A( domain, coords_shell_double, coords_radii_double, true, false );
        duration = measure_run_time( executions, A, src_scalar_double, dst_scalar_double );
        dofs     = dofs_scalar;
    }
    else if ( benchmark == BenchmarkType::VectorLaplaceFloat )
    {
        VectorLaplaceSimple< float > A( domain, coords_shell_float, coords_radii_float, true, false );
        duration = measure_run_time( executions, A, src_vec_float, dst_vec_float );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::VectorLaplaceDouble )
    {
        VectorLaplaceSimple< double > A( domain, coords_shell_double, coords_radii_double, true, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::VectorLaplaceNeumannDouble )
    {
        VectorLaplaceSimple< double > A( domain, coords_shell_double, coords_radii_double, false, false );
        duration = measure_run_time( executions, A, src_vec_double, dst_vec_double );
        dofs     = dofs_vec;
    }
    else if ( benchmark == BenchmarkType::StokesDouble )
    {
        Stokes< double > A( domain, domain_coarse, coords_shell_double, coords_radii_double, true, false );
        duration = measure_run_time( executions, A, src_stokes_double, dst_stokes_double );
        dofs     = dofs_stokes;
    }
    else
    {
        Kokkos::abort( "Unknown benchmark type" );
    }

    return BenchmarkData{ level, dofs, duration };
}

void run_all()
{
    constexpr int min_level  = 1;
    constexpr int max_level  = 6;
    constexpr int executions = 5;

    std::cout << "Running operator (matvec) benchmarks." << std::endl;
    std::cout << "min_level:            " << min_level << std::endl;
    std::cout << "max_level:            " << max_level << std::endl;
    std::cout << "executions per level: " << executions << std::endl;
    std::cout << std::endl;

    for ( auto benchmark : all_benchmark_types )
    {
        std::cout << benchmark_description.at( benchmark ) << std::endl;

        util::Table table;

        for ( int i = min_level; i <= max_level; ++i )
        {
            const auto data = run( benchmark, i, executions );
            table.add_row(
                { { "level", i },
                  { "dofs", data.dofs },
                  { "duration (s)", data.duration },
                  { "updated dofs/sec", data.dofs / data.duration } } );
        }

        table.print_pretty();

        std::cout << std::endl;
        std::cout << std::endl;
    }
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    terra::util::info_table().print_pretty();

    run_all();

    MPI_Finalize();
}