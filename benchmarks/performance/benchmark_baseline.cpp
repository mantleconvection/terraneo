
#include <kernels/common/grid_operations.hpp>

#include "terra/dense/mat.hpp"
#include "terra/dense/vec.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/util/table_printer.hpp"

using terra::grid::shell::allocate_scalar_grid;
using terra::grid::shell::DistributedDomain;
using terra::kernels::common::lincomb;
using terra::kernels::common::set_constant;

enum class BenchmarkType : int
{
    LINCOMB_1 = 0,
    LINCOMB_2,
    STENCIL_INNER_CONSTANT_7,
    ELEMENTWISE_CONSTANT_MATVEC,
};

constexpr auto all_benchmark_types = {
    BenchmarkType::LINCOMB_1,
    BenchmarkType::LINCOMB_2,
    BenchmarkType::STENCIL_INNER_CONSTANT_7,
    BenchmarkType::ELEMENTWISE_CONSTANT_MATVEC };

const std::map< BenchmarkType, std::string > benchmark_description = {
    { BenchmarkType::LINCOMB_1, "Simple y <- c0 * x0 (c scalar, y, x vector)" },
    { BenchmarkType::LINCOMB_2, "Simple y <- c0 * x0 + c1 * x1 (c scalar, y, x vector)" },
    { BenchmarkType::STENCIL_INNER_CONSTANT_7,
      "Constant 7-point stencil applied to the inner nodes of each subdomain (y <- Ax)." },
    { BenchmarkType::ELEMENTWISE_CONSTANT_MATVEC,
      "Constant elementwise operator to all elements of each subdomain (bilinear elements, y <- Ax)." },
};

struct BenchmarkData
{
    int    lateral_refinement_level;
    int    radial_refinement_level;
    int    dofs;
    double duration;
};

struct ConstantStencilInner
{
    terra::grid::Grid4DDataScalar< double > src;
    terra::grid::Grid4DDataScalar< double > dst;

    ConstantStencilInner( terra::grid::Grid4DDataScalar< double > _src, terra::grid::Grid4DDataScalar< double > _dst )
    : src( _src )
    , dst( _dst )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int subdomain_idx, const int x, const int y, const int r ) const
    {
        dst( subdomain_idx, x, y, r ) = 4.0 * src( subdomain_idx, x, y, r ) -
                                        ( src( subdomain_idx, x + 1, y, r ) + src( subdomain_idx, x - 1, y, r ) +
                                          src( subdomain_idx, x, y + 1, r ) + src( subdomain_idx, x, y - 1, r ) +
                                          src( subdomain_idx, x, y, r + 1 ) + +src( subdomain_idx, x, y, r - 1 ) );
    }
};

struct ConstantElementwise
{
    terra::grid::Grid4DDataScalar< double > src;
    terra::grid::Grid4DDataScalar< double > dst;

    ConstantElementwise( terra::grid::Grid4DDataScalar< double > _src, terra::grid::Grid4DDataScalar< double > _dst )
    : src( _src )
    , dst( _dst )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int subdomain_idx, const int x_cell, const int y_cell, const int r_cell ) const
    {
        terra::dense::Vec< double, 8 >    src_local;
        terra::dense::Mat< double, 8, 8 > mat;

        mat.fill( 0.1 );

        for ( int x = x_cell; x < x_cell + 1; x++ )
        {
            for ( int y = y_cell; y < y_cell + 1; y++ )
            {
                for ( int r = r_cell; r < r_cell + 1; r++ )
                {
                    src_local( 4 * ( r - r_cell ) + 2 * ( y - y_cell ) + ( x - x_cell ) ) =
                        src( subdomain_idx, x, y, r );
                }
            }
        }

        terra::dense::Vec< double, 8 > dst_local = mat * src_local;

        for ( int x = x_cell; x < x_cell + 1; x++ )
        {
            for ( int y = y_cell; y < y_cell + 1; y++ )
            {
                for ( int r = r_cell; r < r_cell + 1; r++ )
                {
                    Kokkos::atomic_add(
                        &dst( subdomain_idx, x, y, r ),
                        dst_local( 4 * ( r - r_cell ) + 2 * ( y - y_cell ) + ( x - x_cell ) ) );
                }
            }
        }
    }
};

BenchmarkData
    run( const BenchmarkType benchmark,
         const int           lateral_refinement_level,
         const int           radial_refinement_level,
         const int           executions )
{
    const auto domain = DistributedDomain::create_uniform_single_subdomain(
        lateral_refinement_level, radial_refinement_level, 0.5, 1.0 );

    const auto y  = allocate_scalar_grid< double >( "y", domain );
    const auto x0 = allocate_scalar_grid< double >( "x0", domain );
    const auto x1 = allocate_scalar_grid< double >( "x1", domain );

    set_constant( x0, 1.0 );
    set_constant( x1, 1.0 );

    if ( !y.span_is_contiguous() )
    {
        std::cout << "Span is not contiguous!" << std::endl;
    }
    const int dofs = y.span();

    Kokkos::Timer timer;

    Kokkos::fence();
    timer.reset();

    for ( int i = 0; i < executions; ++i )
    {
        if ( benchmark == BenchmarkType::LINCOMB_1 )
        {
            lincomb( y, 0.0, 42.0, x0 );
        }
        else if ( benchmark == BenchmarkType::LINCOMB_2 )
        {
            lincomb( y, 0.0, 42.0, x0, 4711.0, x1 );
        }
        else if ( benchmark == BenchmarkType::STENCIL_INNER_CONSTANT_7 )
        {
            Kokkos::parallel_for(
                "stencil",
                Kokkos::MDRangePolicy(
                    { 0, 1, 1, 1 }, { y.extent( 0 ), y.extent( 1 ) - 1, y.extent( 2 ) - 1, y.extent( 3 ) - 1 } ),
                ConstantStencilInner( x0, y ) );
        }
        else if ( benchmark == BenchmarkType::ELEMENTWISE_CONSTANT_MATVEC )
        {
            Kokkos::parallel_for(
                "elementwise",
                terra::grid::shell::local_domain_md_range_policy_cells( domain ),
                ConstantElementwise( x0, y ) );
        }
    }

    Kokkos::fence();
    const double duration = timer.seconds() / executions;

    // This does not seem to be necessary but leaving it here just in case.
    const bool print_derived_value = false;
    if ( print_derived_value )
    {
        const auto mm = terra::kernels::common::max_abs_entry( y );
        std::cout << "Printing some derived value to ensure nothing is optimized out: " << mm << std::endl;
    }

    return BenchmarkData{ lateral_refinement_level, radial_refinement_level, dofs, duration };
}

void run_all()
{
    const int min_level  = 2;
    const int max_level  = 8;
    const int executions = 100;

    for ( auto benchmark : all_benchmark_types )
    {
        std::cout << "===================================================================================" << std::endl;
        std::cout << benchmark_description.at( benchmark ) << std::endl;
        std::cout << "===================================================================================" << std::endl;

        terra::util::TablePrinter table;
        table.addRow( { "lateral level", "radial level", "dofs", "duration (s)", "updated dofs/sec" } );

        for ( int i = min_level; i <= max_level; ++i )
        {
            const auto data = run( benchmark, i, i, executions );
            table.addRow( { i, i, data.dofs, data.duration, data.dofs / data.duration } );
        }

        table.print();
        std::cout << "===================================================================================" << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
}

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::ScopeGuard scope_guard( argc, argv );

    run_all();

    MPI_Finalize();
}