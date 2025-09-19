#include "kernels/common/grid_operations.hpp"
#include "linalg/vector_q1.hpp"
#include "terra/grid/shell/spherical_shell.hpp"
#include "terra/visualization/vtk.hpp"
#include "util/init.hpp"

/// For dot products to work correctly, we need to properly define vertex ownership.
/// Neighboring subdomains share vertices, and to uniquely mark them, we create a mask View, that is either 1 or 0
/// at each vertex, such that each logically identical vertex is only marked 1 exactly once.
///
/// This test checks some properties to ensure this is implemented correctly.
void test( const int level )
{
    // The refinement level defines the number of unknowns (for equal refinement in all directions).

    const int number_of_nodes_along_each_diamond_edge = ( 1 << level ) + 1;

    const int number_of_nodes =
        ( 10 * ( number_of_nodes_along_each_diamond_edge - 1 ) * ( number_of_nodes_along_each_diamond_edge - 1 ) + 2 ) *
        number_of_nodes_along_each_diamond_edge;

    const auto domain = terra::grid::shell::DistributedDomain::create_uniform_single_subdomain(
        level, level, 0.5, 1.0, terra::grid::shell::subdomain_to_rank_distribute_full_diamonds );

    auto mask_data = terra::linalg::setup_mask_data( domain );

    // Summing up all the mask entries should be equal to the number of nodes.
    // First casting to larger type to enable adding those up.

    auto ones           = terra::grid::shell::allocate_scalar_grid< long >( "ones", domain );
    auto mask_data_long = terra::grid::shell::allocate_scalar_grid< long >( "mask_data_long", domain );
    terra::kernels::common::set_constant( ones, 1l );
    terra::kernels::common::assign_masked_else_keep_old( mask_data_long, ones, mask_data, terra::grid::mask_owned() );

    const auto number_of_nodes_mask = terra::kernels::common::sum_of_absolutes( mask_data_long );

    std::cout << "Level:                                      " << level << std::endl;
    std::cout << "Number of nodes (analytical):               " << number_of_nodes << std::endl;
    std::cout << "Number of nodes (mask):                     " << number_of_nodes_mask << std::endl;

    if ( number_of_nodes_mask != number_of_nodes )
    {
        throw std::logic_error( "Number of nodes does not match number of mask entries." );
    }

    const auto min_mask = terra::kernels::common::min_abs_entry( mask_data_long );
    const auto max_mask = terra::kernels::common::max_abs_entry( mask_data_long );

    if ( min_mask != 0 || max_mask != 1 )
    {
        throw std::logic_error( "Mask entries are not in [0, 1]." );
    }

    // Now we can also test additive communication of the mask. This should set all entries to one.
    terra::communication::shell::send_recv( domain, mask_data_long );

    const auto min_mask_after = terra::kernels::common::min_abs_entry( mask_data_long );
    const auto max_mask_after = terra::kernels::common::max_abs_entry( mask_data_long );

    if ( min_mask_after != 1 || max_mask_after != 1 )
    {
        throw std::logic_error( "Mask entries are not 1 (after communication)." );
    }

    // Now we check if the sum is equal to the actual number of allocated nodes ("with overlap").
    const auto sum_mask_after = terra::kernels::common::sum_of_absolutes( mask_data_long );

    const auto number_of_nodes_with_overlap = number_of_nodes_along_each_diamond_edge *
                                              number_of_nodes_along_each_diamond_edge *
                                              number_of_nodes_along_each_diamond_edge * 10;

    std::cout << "Number of nodes (with overlap, analytical): " << number_of_nodes_with_overlap << std::endl;
    std::cout << "Number of nodes (with overlap, mask):       " << sum_mask_after << std::endl;

    if ( sum_mask_after != ( number_of_nodes_along_each_diamond_edge * number_of_nodes_along_each_diamond_edge *
                             number_of_nodes_along_each_diamond_edge * 10 ) )
    {
        throw std::logic_error( "Sum of mask entries does not match (after communication)." );
    }

    std::cout << std::endl;
}

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    try
    {
        for ( int level = 0; level < 5; level++ )
        {
            test( level );
        }
    }
    catch ( const std::exception& e )
    {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}