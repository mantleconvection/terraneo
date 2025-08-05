

#include "../src/terra/communication/shell/communication.hpp"
#include "fe/strong_algebraic_dirichlet_enforcement.hpp"
#include "fe/wedge/integrands.hpp"
#include "fe/wedge/operators/shell/laplace.hpp"
#include "fe/wedge/operators/shell/laplace_simple.hpp"
#include "linalg/solvers/pcg.hpp"
#include "linalg/solvers/richardson.hpp"
#include "linalg/util/debug_sparse_assembly.hpp"
#include "terra/dense/mat.hpp"
#include "terra/fe/wedge/operators/shell/mass.hpp"
#include "terra/fe/wedge/operators/shell/prolongation.hpp"
#include "terra/fe/wedge/operators/shell/restriction.hpp"
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
using grid::shell::DistributedDomain;
using grid::shell::DomainInfo;
using grid::shell::SubdomainInfo;
using linalg::VectorQ1Scalar;

struct ConstantFunctionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    ConstantFunctionInterpolator(
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
        const double value = 1.0;

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

struct SomeFunctionInterpolator
{
    Grid3DDataVec< double, 3 > grid_;
    Grid2DDataScalar< double > radii_;
    Grid4DDataScalar< double > data_;
    bool                       only_boundary_;

    SomeFunctionInterpolator(
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

        const double value = Kokkos::sinh( coords( 0 ) ) * Kokkos::cosh( coords( 1 ) ) * Kokkos::tanh( coords( 2 ) );

        if ( !only_boundary_ || ( r == 0 || r == radii_.extent( 1 ) - 1 ) )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

void test( int level, const std::shared_ptr< util::Table >& table )
{
    using ScalarType = double;

    if ( level < 1 )
    {
        throw std::runtime_error( "level must be >= 1" );
    }

    const auto domain_fine   = DistributedDomain::create_uniform_single_subdomain( level, level, 0.5, 1.0 );
    const auto domain_coarse = DistributedDomain::create_uniform_single_subdomain( level - 1, level - 1, 0.5, 1.0 );

    auto mask_data_fine   = linalg::setup_mask_data( domain_fine );
    auto mask_data_coarse = linalg::setup_mask_data( domain_coarse );

    VectorQ1Scalar< ScalarType > u_coarse( "u_coarse", domain_coarse, mask_data_coarse );
    VectorQ1Scalar< ScalarType > u_fine( "u_fine", domain_fine, mask_data_fine );

    VectorQ1Scalar< ScalarType > solution_fine( "solution_fine", domain_fine, mask_data_fine );
    VectorQ1Scalar< ScalarType > error_fine( "error_fine", domain_fine, mask_data_fine );

    const auto subdomain_shell_coords_fine =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain_fine );
    const auto subdomain_radii_fine = terra::grid::shell::subdomain_shell_radii( domain_fine );

    const auto subdomain_shell_coords_coarse =
        terra::grid::shell::subdomain_unit_sphere_single_shell_coords( domain_coarse );
    const auto subdomain_radii_coarse = terra::grid::shell::subdomain_shell_radii( domain_coarse );

    using Prolongation = fe::wedge::operators::shell::Prolongation< ScalarType >;
    using Restriction  = fe::wedge::operators::shell::Restriction< ScalarType >;

    Prolongation P;
    Restriction  R( domain_coarse );

    Eigen::SparseMatrix< double > P_assembled =
        linalg::util::debug_sparse_assembly_operator_vec_q1_scalar( domain_coarse, P, u_coarse, u_fine );

    Eigen::SparseMatrix< double > R_assembled =
        linalg::util::debug_sparse_assembly_operator_vec_q1_scalar( domain_fine, R, u_fine, u_coarse );

    std::cout << P_assembled.toDense() << std::endl;
    std::cout << std::endl;
    std::cout << R_assembled.toDense() << std::endl;
    std::cout << std::endl;

    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > error =
        P_assembled.toDense() - R_assembled.transpose().toDense();

    std::cout << error << std::endl;

    const auto error_norm = error.norm();
    std::cout << "error norm: " << error_norm << std::endl;

    if ( error_norm > 1e-15 )
    {
        throw std::runtime_error( "error is not zero" );
    }
}

int main( int argc, char** argv )
{
    util::TerraScopeGuard scope_guard( &argc, &argv );

    auto table = std::make_shared< util::Table >();

    test( 1, table );

    return 0;
}