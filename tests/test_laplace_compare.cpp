

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/laplace_batched.hpp"
#include "fe/wedge/operators/shell/laplace_no_matrix.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/grid/grid_types.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/kernels/common/grid_operations.hpp"
#include "terra/kokkos/kokkos_wrapper.hpp"
#include "terra/visualization/xdmf.hpp"
#include "util/init.hpp"
#include "util/table.hpp"

using namespace terra;

using grid::Grid2DDataScalar;
using grid::Grid3DDataScalar;
using grid::Grid3DDataVec;
using grid::Grid4DDataScalar;
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;

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
        // const double                  value  = coords( 0 ) * Kokkos::sin( coords( 1 ) ) * Kokkos::sinh( coords( 2 ) );
        const auto   rr = radii_( local_subdomain_id, r );
        const double value =
            ( rr - 1.0 ) * ( rr - 0.5 ) * ( 1.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );

        data_( local_subdomain_id, x, y, r ) = value;
    }
};

void test( int level )
{
    Kokkos::Timer timer;

    using ScalarType = double;

    const auto domain = DistributedDomain::create_uniform_single_subdomain(
        level, level, 0.5, 1.0, grid::shell::subdomain_to_rank_distribute_full_diamonds );

    auto mask_data = linalg::setup_mask_data( domain );

    VectorQ1Scalar< ScalarType > src( "src", domain, mask_data );
    VectorQ1Scalar< ScalarType > dst_a( "dst_a", domain, mask_data );
    VectorQ1Scalar< ScalarType > dst_b( "dst_b", domain, mask_data );
    VectorQ1Scalar< ScalarType > dst_c( "dst_c", domain, mask_data );
    VectorQ1Scalar< ScalarType > error_b( "error_b", domain, mask_data );
    VectorQ1Scalar< ScalarType > error_c( "error_c", domain, mask_data );

    const auto num_dofs = kernels::common::count_masked< long >( mask_data, grid::mask_owned() );

    const auto coords_shell = terra::grid::shell::subdomain_unit_sphere_single_shell_coords< ScalarType >( domain );
    const auto coords_radii = terra::grid::shell::subdomain_shell_radii< ScalarType >( domain );

    using LaplaceA = fe::wedge::operators::shell::LaplaceSimple< ScalarType >;
    using LaplaceB = fe::wedge::operators::shell::LaplaceBatched< ScalarType >;
    using LaplaceC = fe::wedge::operators::shell::LaplaceNoMatrix< ScalarType >;

    LaplaceA A( domain, coords_shell, coords_radii, false, false );
    LaplaceB B( domain, coords_shell, coords_radii, false, false );
    LaplaceC C( domain, coords_shell, coords_radii, false, false );

    // Set up solution data.
    Kokkos::parallel_for(
        "src interpolation",
        local_domain_md_range_policy_nodes( domain ),
        SolutionInterpolator( coords_shell, coords_radii, src.grid_data(), false ) );

    Kokkos::fence();

    linalg::apply( A, src, dst_a );
    linalg::apply( B, src, dst_b );
    linalg::apply( C, src, dst_c );

    linalg::lincomb( error_b, { 1.0, -1.0 }, { dst_a, dst_b } );
    linalg::lincomb( error_c, { 1.0, -1.0 }, { dst_a, dst_c } );

    const auto l2_error_b = std::sqrt( dot( error_b, error_b ) / num_dofs );
    const auto l2_error_c = std::sqrt( dot( error_c, error_c ) / num_dofs );

    const auto inf_error_b = linalg::norm_inf( error_b );
    const auto inf_error_c = linalg::norm_inf( error_c );

    std::cout << "L2 error (batched):    " << l2_error_b << std::endl;
    std::cout << "L2 error (no matrix):  " << l2_error_c << std::endl;

    std::cout << "inf error (batched):   " << inf_error_b << std::endl;
    std::cout << "inf error (no matrix): " << inf_error_c << std::endl;

    if ( true )
    {
        visualization::XDMFOutput xdmf( "out", coords_shell, coords_radii );
        xdmf.add( src.grid_data() );
        xdmf.add( dst_a.grid_data() );
        xdmf.add( dst_b.grid_data() );
        xdmf.add( error_b.grid_data() );
        xdmf.add( error_c.grid_data() );
        xdmf.write();
    }
}

int main( int argc, char** argv )
{
    util::terra_initialize( &argc, &argv );
    test( 0 );
    test( 1 );
    test( 2 );
    test( 3 );
    test( 4 );
    test( 5 );
    return 0;
}