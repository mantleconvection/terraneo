
#pragma once
#include <mpi.h>

#include "communication/shell/communication.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "vector.hpp"
#include "vector_q1.hpp"

namespace terra::linalg {

template < typename ScalarT, int VecDim = 3 >
class VectorQ1IsoQ2Q1
{
  public:
    using ScalarType = ScalarT;

    using Block1Type = VectorQ1Vec< ScalarType, VecDim >;
    using Block2Type = VectorQ1Scalar< ScalarType >;

    VectorQ1IsoQ2Q1(
        const std::string&                              label,
        const grid::shell::DistributedDomain&           distributed_domain_fine,
        const grid::shell::DistributedDomain&           distributed_domain_coarse,
        const grid::Grid4DDataScalar< util::MaskType >& mask_data_fine,
        const grid::Grid4DDataScalar< util::MaskType >& mask_data_coarse )
    : u_( label + "_u", distributed_domain_fine, mask_data_fine )
    , p_( label + "_p", distributed_domain_coarse, mask_data_coarse )
    {}

    void
        lincomb_impl( const std::vector< ScalarType >& c, const std::vector< VectorQ1IsoQ2Q1 >& x, const ScalarType c0 )
    {
        std::vector< Block1Type > us;
        std::vector< Block2Type > ps;

        for ( const auto& xx : x )
        {
            us.emplace_back( xx.block_1() );
            ps.emplace_back( xx.block_2() );
        }

        u_.lincomb_impl( c, us, c0 );
        p_.lincomb_impl( c, ps, c0 );
    }

    ScalarType dot_impl( const VectorQ1IsoQ2Q1& x ) const
    {
        return x.block_1().dot_impl( u_ ) + x.block_2().dot_impl( p_ );
    }

    void invert_entries_impl()
    {
        block_1().invert_entries_impl();
        block_2().invert_entries_impl();
    }

    void scale_with_vector_impl( const VectorQ1IsoQ2Q1& x )
    {
        block_1().scale_with_vector_impl( x.block_1() );
        block_2().scale_with_vector_impl( x.block_2() );
    }

    void randomize_impl()
    {
        block_1().randomize_impl();
        block_2().randomize_impl();
    }

    ScalarType max_abs_entry_impl() const
    {
        return std::max( block_1().max_abs_entry_impl(), block_2().max_abs_entry_impl() );
    }

    bool has_nan_impl() const { return block_1().has_nan_impl() || block_2().has_nan_impl(); }

    void swap_impl( VectorQ1IsoQ2Q1& other )
    {
        u_.swap_impl( other.u_ );
        p_.swap_impl( other.p_ );
    }

    void add_mask_data(
        const grid::Grid4DDataScalar< unsigned char >& mask_data_block_1,
        const grid::Grid4DDataScalar< unsigned char >& mask_data_block_2 )
    {
        block_1().mask_data() = mask_data_block_1;
        block_2().mask_data() = mask_data_block_2;
    }

    const Block1Type& block_1() const { return u_; }
    const Block2Type& block_2() const { return p_; }

    Block1Type& block_1() { return u_; }
    Block2Type& block_2() { return p_; }

  private:
    Block1Type u_;
    Block2Type p_;
};

static_assert( Block2VectorLike< VectorQ1IsoQ2Q1< double > > );

template < typename ValueType, int VecDim = 3 >
VectorQ1IsoQ2Q1< ValueType, VecDim > allocate_vector_q1isoq2_q1(
    const std::string                     label,
    const grid::shell::DistributedDomain& distributed_domain_fine,
    const grid::shell::DistributedDomain& distributed_domain_coarse,
    const int                             level )
{
    grid::Grid4DDataVec< ValueType, VecDim > grid_data_fine(
        label,
        distributed_domain_fine.subdomains().size(),
        distributed_domain_fine.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain_fine.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain_fine.domain_info().subdomain_num_nodes_radially() );

    grid::Grid4DDataScalar< ValueType > grid_data_coarse(
        label,
        distributed_domain_coarse.subdomains().size(),
        distributed_domain_coarse.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain_coarse.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain_coarse.domain_info().subdomain_num_nodes_radially() );

    VectorQ1IsoQ2Q1< ValueType, VecDim > vector_q1isoq2_q1;
    vector_q1isoq2_q1.block_1().add_grid_data( grid_data_fine, level );
    vector_q1isoq2_q1.block_2().add_grid_data( grid_data_coarse, level );
    return vector_q1isoq2_q1;
}

} // namespace terra::linalg