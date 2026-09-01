// Mesh export and operator validation for the manufactured-solution Stokes dataset.
//
// The symbolic work lives in python/terra_data/stokes_symbolic.py, which derives
//     f = -div( 2 eta ( eps(u) - (1/3)(div u) I ) ) + grad p
// with sympy and checks it against the manufactured solution hardcoded in
// tests/test_epsilon_divdiv_stokes.cpp. This tool supplies the two things that can
// only come from TERRA-NG itself:
//
//   --dump-coords   node coordinates of the velocity and pressure grids, so the
//                   Python generator evaluates its polynomials at exactly the nodes
//                   the solver uses.
//
//   --validate      interpolates TERRA-NG's own analytic test case, applies the real
//                   discretised Stokes operator, and reports ||K u_h - M f_h|| against
//                   refinement. The mass matrix is there because the assembled load
//                   vector is M f, not f -- the same construction the Stokes test uses.
//                   The residual is the FE consistency error and must fall with h.
//
//   --check-sample  reads one generated sample, applies the real discretised operator to
//                   its (u, p) and compares against M f_u from the same file. This is the
//                   check on the *data*: --validate only ever exercises the analytic case
//                   that ships with the test, whereas a generated sample has a different
//                   polynomial, a different viscosity and a nonzero div u.
//
// Single rank: one sample is one file covering the whole domain.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "fe/wedge/operators/shell/epsilon_divdiv_stokes.hpp"
#include "fe/wedge/operators/shell/vector_mass.hpp"
#include "grid/shell/bit_masks.hpp"
#include "linalg/vector_q1isoq2_q1.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "util/cli11_helper.hpp"
#include "util/cli11_wrapper.hpp"
#include "util/filesystem.hpp"
#include "util/init.hpp"
#include "util/logging.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::Grid4DDataVec;
using grid::shell::DistributedDomain;
using linalg::VectorQ1IsoQ2Q1;
using linalg::VectorQ1Scalar;
using linalg::VectorQ1Vec;
using util::add_flag_with_default;
using util::add_option_with_default;
using util::logroot;

using ScalarType = double;
using Stokes     = fe::wedge::operators::shell::EpsDivDivStokes< ScalarType >;
using VectorMass = fe::wedge::operators::shell::VectorMass< ScalarType >;

// --------------------------------------------------------------------- analytic case
//
// Transcribed from tests/test_epsilon_divdiv_stokes.cpp so the two stay comparable.
// python/terra_data/stokes_symbolic.py carries the same expressions and re-derives
// FIELD_F from FIELD_U, FIELD_P and FIELD_ETA symbolically.

struct AnalyticVelocity
{
    Grid3DDataVec< ScalarType, 3 >   grid_;
    Grid2DDataScalar< ScalarType >   radii_;
    Grid4DDataVec< ScalarType, 3 >   out_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int i, const int j, const int k ) const
    {
        const auto c = grid::shell::coords( sd, i, j, k, grid_, radii_ );
        out_( sd, i, j, k, 0 ) = -4 * Kokkos::cos( 4 * c( 2 ) );
        out_( sd, i, j, k, 1 ) = 8 * Kokkos::cos( 8 * c( 0 ) );
        out_( sd, i, j, k, 2 ) = -2 * Kokkos::cos( 2 * c( 1 ) );
    }
};

struct AnalyticPressure
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > out_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int i, const int j, const int k ) const
    {
        const auto c = grid::shell::coords( sd, i, j, k, grid_, radii_ );
        out_( sd, i, j, k ) =
            Kokkos::sin( 4 * c( 0 ) ) * Kokkos::sin( 8 * c( 1 ) ) * Kokkos::sin( 2 * c( 2 ) );
    }
};

struct AnalyticViscosity
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataScalar< ScalarType > out_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int i, const int j, const int k ) const
    {
        const auto c = grid::shell::coords( sd, i, j, k, grid_, radii_ );
        out_( sd, i, j, k ) = 2 + Kokkos::sin( c( 2 ) );
    }
};

struct AnalyticRHS
{
    Grid3DDataVec< ScalarType, 3 > grid_;
    Grid2DDataScalar< ScalarType > radii_;
    Grid4DDataVec< ScalarType, 3 > out_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int i, const int j, const int k ) const
    {
        const auto   c  = grid::shell::coords( sd, i, j, k, grid_, radii_ );
        const double x0 = 4 * c( 2 );

        out_( sd, i, j, k, 0 ) = -64.0 * ( Kokkos::sin( c( 2 ) ) + 2 ) * Kokkos::cos( x0 ) -
                                 16.0 * Kokkos::sin( x0 ) * Kokkos::cos( c( 2 ) ) +
                                 4 * Kokkos::sin( 8 * c( 1 ) ) * Kokkos::sin( 2 * c( 2 ) ) *
                                     Kokkos::cos( 4 * c( 0 ) );
        out_( sd, i, j, k, 1 ) = 512.0 * ( Kokkos::sin( c( 2 ) ) + 2 ) * Kokkos::cos( 8 * c( 0 ) ) +
                                 8 * Kokkos::sin( 4 * c( 0 ) ) * Kokkos::sin( 2 * c( 2 ) ) *
                                     Kokkos::cos( 8 * c( 1 ) ) -
                                 4.0 * Kokkos::sin( 2 * c( 1 ) ) * Kokkos::cos( c( 2 ) );
        out_( sd, i, j, k, 2 ) = -8.0 * ( Kokkos::sin( c( 2 ) ) + 2 ) * Kokkos::cos( 2 * c( 1 ) ) +
                                 2 * Kokkos::sin( 4 * c( 0 ) ) * Kokkos::sin( 8 * c( 1 ) ) *
                                     Kokkos::cos( 2 * c( 2 ) );
    }
};

/// @brief Zeroes a velocity field on the boundary.
///
/// The comparison below is interior-only on purpose. With natural (Neumann) boundary
/// conditions the weak form contributes a surface term \int (sigma . n) . v that the
/// strong-form right-hand side has no counterpart for. That term scales like h^2 per
/// node against the volume term's h^3, so boundary rows do not merely pollute the
/// residual -- they dominate it, and it grows under refinement instead of falling.
struct ZeroAtBoundary
{
    Grid4DDataVec< ScalarType, 3 >                     field_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;

    KOKKOS_INLINE_FUNCTION
    void operator()( const int sd, const int i, const int j, const int k ) const
    {
        if ( util::has_flag( mask_( sd, i, j, k ), grid::shell::ShellBoundaryFlag::BOUNDARY ) )
            for ( int d = 0; d < 3; ++d )
                field_( sd, i, j, k, d ) = 0.0;
    }
};

/// Writes node coordinates as flat float64, layout [n_sd, nx, ny, nr, 3].
void dump_coords(
    const std::string&                    path,
    const DistributedDomain&              domain,
    const Grid3DDataVec< ScalarType, 3 >& shell,
    const Grid2DDataScalar< ScalarType >& radii )
{
    auto shell_h = Kokkos::create_mirror_view( shell );
    auto radii_h = Kokkos::create_mirror_view( radii );
    Kokkos::deep_copy( shell_h, shell );
    Kokkos::deep_copy( radii_h, radii );

    const auto n_sd = static_cast< int >( domain.subdomains().size() );
    const auto nx   = domain.domain_info().subdomain_num_nodes_per_side_laterally();
    const auto nr   = domain.domain_info().subdomain_num_nodes_radially();

    std::vector< double > buffer;
    buffer.reserve( static_cast< std::size_t >( n_sd ) * nx * nx * nr * 3 );
    for ( int s = 0; s < n_sd; ++s )
        for ( int i = 0; i < nx; ++i )
            for ( int j = 0; j < nx; ++j )
                for ( int k = 0; k < nr; ++k )
                    for ( int d = 0; d < 3; ++d )
                        buffer.push_back( shell_h( s, i, j, d ) * radii_h( s, k ) );

    std::ofstream out( path, std::ios::binary );
    out.write( reinterpret_cast< const char* >( buffer.data() ),
               static_cast< std::streamsize >( buffer.size() * sizeof( double ) ) );
    logroot << "  wrote " << path << "  [" << n_sd << ", " << nx << ", " << nx << ", " << nr
            << ", 3] float64\n";
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );

    std::string outdir    = "stokes_dataset";
    std::string check_sample;
    int         min_level = 2;
    int         max_level = 4;
    double      r_min = 0.5, r_max = 1.0;
    bool        validate = false;

    {
        CLI::App app{ "Mesh export and operator validation for the Stokes manufactured dataset." };
        add_option_with_default( app, "--outdir", outdir, "Output directory." );
        add_option_with_default( app, "--min-level", min_level, "Coarsest level (validation sweep start)." );
        add_option_with_default( app, "--max-level", max_level, "Finest level; coordinates are dumped for this." );
        add_option_with_default( app, "--r-min", r_min, "Inner radius." );
        add_option_with_default( app, "--r-max", r_max, "Outer radius." );
        add_flag_with_default(
            app, "--validate", validate, "Check the discrete operator against the analytic test case." );
        add_option_with_default(
            app, "--check-sample", check_sample, "Path of a generated sample .bin to verify against the operator." );
        CLI11_PARSE( app, argc, argv );
    }

    if ( mpi::num_processes() != 1 )
    {
        logroot << "Run on a single rank.\n";
        return 1;
    }

    std::filesystem::create_directories( outdir );

    // One (velocity, pressure) pair per level: pressure is always one level coarser.
    auto build = [&]( const int level ) {
        auto dom = DistributedDomain::create_uniform_single_subdomain_per_diamond( level, level, r_min, r_max );
        return dom;
    };

    if ( validate )
    {
        logroot << "\nConsistency of the discrete operator against the analytic test case\n";
        logroot << "  (residual = || K u_h - M f_h || / || M f_h ||, over interior velocity nodes)\n\n";
        logroot << "  level        h      relative residual     ratio\n";

        double previous = 0.0;
        for ( int level = min_level + 1; level <= max_level; ++level )
        {
            auto dom_v = build( level );
            auto dom_p = build( level - 1 );

            auto shell_v = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom_v );
            auto radii_v = grid::shell::subdomain_shell_radii< ScalarType >( dom_v );
            auto shell_p = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom_p );
            auto radii_p = grid::shell::subdomain_shell_radii< ScalarType >( dom_p );

            auto mask_v  = grid::setup_node_ownership_mask_data( dom_v );
            auto mask_p  = grid::setup_node_ownership_mask_data( dom_p );
            auto bmask_v = grid::shell::setup_boundary_mask_data( dom_v );

            VectorQ1Scalar< ScalarType > eta( "eta", dom_v, mask_v );
            Kokkos::parallel_for( "eta",
                                  grid::shell::local_domain_md_range_policy_nodes( dom_v ),
                                  AnalyticViscosity{ shell_v, radii_v, eta.grid_data() } );

            // Neumann everywhere: the analytic solution does not satisfy homogeneous
            // Dirichlet data, so eliminating boundary rows would compare different things.
            grid::shell::BoundaryConditions bcs = {
                { grid::shell::ShellBoundaryFlag::CMB, grid::shell::BoundaryConditionFlag::NEUMANN },
                { grid::shell::ShellBoundaryFlag::SURFACE, grid::shell::BoundaryConditionFlag::NEUMANN } };

            Stokes     K( dom_v, dom_p, shell_v, radii_v, bmask_v, eta.grid_data(), bcs, false );
            VectorMass M( dom_v, shell_v, radii_v, false );

            VectorQ1IsoQ2Q1< ScalarType > u( "u", dom_v, dom_p, mask_v, mask_p );
            VectorQ1IsoQ2Q1< ScalarType > Ku( "Ku", dom_v, dom_p, mask_v, mask_p );
            VectorQ1Vec< ScalarType >     f_point( "f_point", dom_v, mask_v );
            VectorQ1Vec< ScalarType >     f_assembled( "f_assembled", dom_v, mask_v );

            Kokkos::parallel_for( "u",
                                  grid::shell::local_domain_md_range_policy_nodes( dom_v ),
                                  AnalyticVelocity{ shell_v, radii_v, u.block_1().grid_data() } );
            Kokkos::parallel_for( "p",
                                  grid::shell::local_domain_md_range_policy_nodes( dom_p ),
                                  AnalyticPressure{ shell_p, radii_p, u.block_2().grid_data() } );
            Kokkos::parallel_for( "f",
                                  grid::shell::local_domain_md_range_policy_nodes( dom_v ),
                                  AnalyticRHS{ shell_v, radii_v, f_point.grid_data() } );
            Kokkos::fence();

            linalg::apply( M, f_point, f_assembled );
            linalg::apply( K, u, Ku );

            linalg::lincomb( Ku.block_1(), { 1.0, -1.0 }, { Ku.block_1(), f_assembled } );

            const auto policy = grid::shell::local_domain_md_range_policy_nodes( dom_v );
            Kokkos::parallel_for( "drop_boundary_residual", policy,
                                  ZeroAtBoundary{ Ku.block_1().grid_data(), bmask_v } );
            Kokkos::parallel_for( "drop_boundary_reference", policy,
                                  ZeroAtBoundary{ f_assembled.grid_data(), bmask_v } );
            Kokkos::fence();

            const double residual = linalg::norm_2( Ku.block_1() ) / linalg::norm_2( f_assembled );
            const double h        = ( r_max - r_min ) / std::pow( 2.0, level );

            logroot << "  " << level << "     " << h << "      " << residual << "      "
                    << ( previous > 0.0 ? previous / residual : 0.0 ) << "\n";
            previous = residual;
        }
        logroot << "\n";
    }

    // ------------------------------------------------------------------ sample check
    if ( !check_sample.empty() )
    {
        auto dom_v = build( max_level );
        auto dom_p = build( max_level - 1 );

        auto shell_v = grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom_v );
        auto radii_v = grid::shell::subdomain_shell_radii< ScalarType >( dom_v );
        auto mask_v  = grid::setup_node_ownership_mask_data( dom_v );
        auto mask_p  = grid::setup_node_ownership_mask_data( dom_p );
        auto bmask_v = grid::shell::setup_boundary_mask_data( dom_v );

        const int n_sd_v = static_cast< int >( dom_v.subdomains().size() );
        const int nx_v   = dom_v.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nr_v   = dom_v.domain_info().subdomain_num_nodes_radially();
        const int n_sd_p = static_cast< int >( dom_p.subdomains().size() );
        const int nx_p   = dom_p.domain_info().subdomain_num_nodes_per_side_laterally();
        const int nr_p   = dom_p.domain_info().subdomain_num_nodes_radially();

        const std::size_t nv = static_cast< std::size_t >( n_sd_v ) * nx_v * nx_v * nr_v;
        const std::size_t np = static_cast< std::size_t >( n_sd_p ) * nx_p * nx_p * nr_p;

        std::ifstream in( check_sample, std::ios::binary );
        if ( !in )
        {
            logroot << "cannot open " << check_sample << "\n";
            return 1;
        }
        std::vector< float > raw( 3 * nv + np + nv + 3 * nv + np );
        in.read( reinterpret_cast< char* >( raw.data() ),
                 static_cast< std::streamsize >( raw.size() * sizeof( float ) ) );
        if ( static_cast< std::size_t >( in.gcount() ) != raw.size() * sizeof( float ) )
        {
            logroot << "sample is " << in.gcount() << " bytes, expected "
                    << raw.size() * sizeof( float ) << "\n";
            return 1;
        }

        VectorQ1Scalar< ScalarType > eta( "eta", dom_v, mask_v );
        VectorQ1IsoQ2Q1< ScalarType > u( "u", dom_v, dom_p, mask_v, mask_p );
        VectorQ1IsoQ2Q1< ScalarType > Ku( "Ku", dom_v, dom_p, mask_v, mask_p );
        VectorQ1Vec< ScalarType >     f_point( "f_point", dom_v, mask_v );
        VectorQ1Vec< ScalarType >     f_assembled( "f_assembled", dom_v, mask_v );

        // Unpack the flat float32 into the device fields, undoing the interleave.
        auto load_vec = [&]( std::size_t offset, auto& target ) {
            auto host = grid::create_mirror( Kokkos::HostSpace{}, target.grid_data() );
            std::size_t flat = 0;
            for ( int s_ = 0; s_ < n_sd_v; ++s_ )
                for ( int i = 0; i < nx_v; ++i )
                    for ( int j = 0; j < nx_v; ++j )
                        for ( int k = 0; k < nr_v; ++k, ++flat )
                            for ( int d = 0; d < 3; ++d )
                                host( s_, i, j, k, d ) = raw[offset + flat * 3 + d];
            grid::deep_copy< ScalarType, 3 >( target.grid_data(), host );
        };
        auto load_scalar = [&]( std::size_t offset, auto& target, int n_sd, int nx, int nr ) {
            auto host = Kokkos::create_mirror( Kokkos::HostSpace{}, target.grid_data() );
            std::size_t flat = 0;
            for ( int s_ = 0; s_ < n_sd; ++s_ )
                for ( int i = 0; i < nx; ++i )
                    for ( int j = 0; j < nx; ++j )
                        for ( int k = 0; k < nr; ++k, ++flat )
                            host( s_, i, j, k ) = raw[offset + flat];
            Kokkos::deep_copy( target.grid_data(), host );
        };

        std::size_t off = 0;
        load_vec( off, u.block_1() );                                off += 3 * nv;
        load_scalar( off, u.block_2(), n_sd_p, nx_p, nr_p );         off += np;
        load_scalar( off, eta, n_sd_v, nx_v, nr_v );                 off += nv;
        load_vec( off, f_point );                                    off += 3 * nv;

        grid::shell::BoundaryConditions bcs = {
            { grid::shell::ShellBoundaryFlag::CMB, grid::shell::BoundaryConditionFlag::NEUMANN },
            { grid::shell::ShellBoundaryFlag::SURFACE, grid::shell::BoundaryConditionFlag::NEUMANN } };

        Stokes     K( dom_v, dom_p, shell_v, radii_v, bmask_v, eta.grid_data(), bcs, false );
        VectorMass M( dom_v, shell_v, radii_v, false );

        linalg::apply( M, f_point, f_assembled );
        linalg::apply( K, u, Ku );
        linalg::lincomb( Ku.block_1(), { 1.0, -1.0 }, { Ku.block_1(), f_assembled } );

        const auto policy = grid::shell::local_domain_md_range_policy_nodes( dom_v );
        Kokkos::parallel_for( "drop_b1", policy, ZeroAtBoundary{ Ku.block_1().grid_data(), bmask_v } );
        Kokkos::parallel_for( "drop_b2", policy, ZeroAtBoundary{ f_assembled.grid_data(), bmask_v } );
        Kokkos::fence();

        logroot << "\nSample " << check_sample << "\n";
        logroot << "  eta range      " << kernels::common::min_entry( eta.grid_data() ) << " .. "
                << kernels::common::max_entry( eta.grid_data() ) << "\n";
        logroot << "  || K u - M f ||  / || M f ||  (interior) = "
                << linalg::norm_2( Ku.block_1() ) / linalg::norm_2( f_assembled ) << "\n\n";
        return 0;
    }

    // ------------------------------------------------------------------ coordinate dump
    {
        auto dom_v = build( max_level );
        auto dom_p = build( max_level - 1 );

        logroot << "Dumping node coordinates for level " << max_level << " (pressure at "
                << ( max_level - 1 ) << ")\n";
        dump_coords( outdir + "/coords_velocity.bin",
                     dom_v,
                     grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom_v ),
                     grid::shell::subdomain_shell_radii< ScalarType >( dom_v ) );
        dump_coords( outdir + "/coords_pressure.bin",
                     dom_p,
                     grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( dom_p ),
                     grid::shell::subdomain_shell_radii< ScalarType >( dom_p ) );

        std::ofstream meta( outdir + "/mesh.json" );
        meta << "{\n";
        meta << "  \"level\": " << max_level << ",\n";
        meta << "  \"r_min\": " << r_min << ",\n  \"r_max\": " << r_max << ",\n";
        meta << "  \"velocity_shape\": [" << dom_v.subdomains().size() << ", "
             << dom_v.domain_info().subdomain_num_nodes_per_side_laterally() << ", "
             << dom_v.domain_info().subdomain_num_nodes_per_side_laterally() << ", "
             << dom_v.domain_info().subdomain_num_nodes_radially() << "],\n";
        meta << "  \"pressure_shape\": [" << dom_p.subdomains().size() << ", "
             << dom_p.domain_info().subdomain_num_nodes_per_side_laterally() << ", "
             << dom_p.domain_info().subdomain_num_nodes_per_side_laterally() << ", "
             << dom_p.domain_info().subdomain_num_nodes_radially() << "],\n";
        meta << "  \"coords_dtype\": \"float64\"\n";
        meta << "}\n";
        logroot << "  wrote " << outdir << "/mesh.json\n";
    }

    return 0;
}
