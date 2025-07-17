
#pragma once
#include <mpi.h>

#include "communication/shell/communication.hpp"
#include "grid/shell/spherical_shell.hpp"
#include "kernels/common/grid_operations.hpp"
#include "vector.hpp"

namespace terra::linalg {

inline void
    setup_mask_data( const grid::shell::DistributedDomain& domain, grid::Grid4DDataScalar< unsigned char >& mask_data )
{
    auto tmp_data_for_global_subdomain_indices =
        grid::shell::allocate_scalar_grid< int >( "tmp_data_for_global_subdomain_indices", domain );

    terra::communication::shell::SubdomainNeighborhoodSendBuffer< int > send_buffers( domain );
    terra::communication::shell::SubdomainNeighborhoodRecvBuffer< int > recv_buffers( domain );

    std::vector< std::array< int, 11 > > expected_recvs_metadata;
    std::vector< MPI_Request >           expected_recvs_requests;

    // Interpolate the unique subdomain ID.
    for ( const auto& [subdomain_info, value] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = value;

        const auto global_subdomain_id = subdomain_info.global_id();

        Kokkos::parallel_for(
            "set_global_subdomain_id",
            Kokkos::MDRangePolicy(
                { 0, 0, 0 }, { mask_data.extent( 1 ), mask_data.extent( 2 ), mask_data.extent( 3 ) } ),
            KOKKOS_LAMBDA( const int x, const int y, const int r ) {
                tmp_data_for_global_subdomain_indices( local_subdomain_id, x, y, r ) = global_subdomain_id;
            } );
    }

    // Communicate and reduce with minimum.
    terra::communication::shell::pack_and_send_local_subdomain_boundaries(
        domain, tmp_data_for_global_subdomain_indices, send_buffers, expected_recvs_requests, expected_recvs_metadata );

    terra::communication::shell::recv_unpack_and_add_local_subdomain_boundaries(
        domain,
        tmp_data_for_global_subdomain_indices,
        recv_buffers,
        expected_recvs_requests,
        expected_recvs_metadata,
        terra::communication::shell::CommuncationReduction::MIN );

    // Set all nodes to 1 if the global_subdomain_id matches - 0 otherwise.
    for ( const auto& [subdomain_info, value] : domain.subdomains() )
    {
        const auto& [local_subdomain_id, neighborhood] = value;

        const auto global_subdomain_id = subdomain_info.global_id();

        Kokkos::parallel_for(
            "set_flags",
            Kokkos::MDRangePolicy(
                { 0, 0, 0 }, { mask_data.extent( 1 ), mask_data.extent( 2 ), mask_data.extent( 3 ) } ),
            KOKKOS_LAMBDA( const int x, const int y, const int r ) {
                if ( tmp_data_for_global_subdomain_indices( local_subdomain_id, x, y, r ) == global_subdomain_id )
                {
                    mask_data( local_subdomain_id, x, y, r ) = 1;
                }
                else
                {
                    mask_data( local_subdomain_id, x, y, r ) = 0;
                }
            } );

        Kokkos::fence();
    }
}

template < typename ScalarT >
class VectorQ1Scalar
{
  public:
    VectorQ1Scalar() = default;

    using ScalarType = ScalarT;

    void lincomb_impl(
        const std::vector< ScalarType >&     c,
        const std::vector< VectorQ1Scalar >& x,
        const ScalarType                     c0,
        const int                            level )
    {
        if ( c.size() != x.size() )
        {
            throw std::runtime_error( "VectorQ1Scalar::lincomb: c and x must have the same size" );
        }

        if ( x.size() == 0 )
        {
            kernels::common::set_constant( grid_data( level ), c0 );
        }
        else if ( x.size() == 1 )
        {
            kernels::common::lincomb( grid_data( level ), c0, c[0], x[0].grid_data( level ) );
        }
        else if ( x.size() == 2 )
        {
            kernels::common::lincomb(
                grid_data( level ), c0, c[0], x[0].grid_data( level ), c[1], x[1].grid_data( level ) );
        }
        else if ( x.size() == 3 )
        {
            kernels::common::lincomb(
                grid_data( level ),
                c0,
                c[0],
                x[0].grid_data( level ),
                c[1],
                x[1].grid_data( level ),
                c[2],
                x[2].grid_data( level ) );
        }
        else
        {
            throw std::runtime_error( "VectorQ1Scalar::lincomb: not implemented" );
        }
    }

    ScalarType dot_impl( const VectorQ1Scalar& x, const int level ) const
    {
        return kernels::common::masked_dot_product( grid_data( level ), x.grid_data( level ), mask_data( level ) );
    }

    ScalarType max_abs_entry_impl( const int level ) const
    {
        return kernels::common::max_abs_entry( grid_data( level ) );
    }

    bool has_nan_impl( const int level ) const { return kernels::common::has_nan( grid_data( level ) ); }

    void swap_impl( VectorQ1Scalar& other )
    {
        grid_data_.swap( other.grid_data_ );
        mask_data_.swap( other.mask_data_ );
    }

    void add_grid_data( const grid::Grid4DDataScalar< ScalarType >& grid_data, int level )
    {
        grid_data_.insert( { level, grid_data } );
    }

    void add_mask_data( const grid::Grid4DDataScalar< unsigned char >& mask_data, int level )
    {
        mask_data_.insert( { level, mask_data } );
    }

    grid::Grid4DDataScalar< ScalarType > grid_data( int level ) const
    {
        if ( !grid_data_.contains( level ) )
        {
            throw std::runtime_error( "VectorQ1Scalar::grid_data: level not found" );
        }
        return grid_data_.at( level );
    }

    grid::Grid4DDataScalar< unsigned char > mask_data( int level ) const
    {
        if ( !mask_data_.contains( level ) )
        {
            throw std::runtime_error( "VectorQ1Scalar::mask_data: level not found" );
        }
        return mask_data_.at( level );
    }

  private:
    std::map< int, grid::Grid4DDataScalar< ScalarType > >    grid_data_;
    std::map< int, grid::Grid4DDataScalar< unsigned char > > mask_data_;
};

static_assert( VectorLike< VectorQ1Scalar< double > > );

template < typename ScalarT, int VecDim >
class VectorQ1Vec
{
  public:
    VectorQ1Vec() = default;

    using ScalarType     = ScalarT;
    const static int Dim = VecDim;

    void lincomb_impl(
        const std::vector< ScalarType >&  c,
        const std::vector< VectorQ1Vec >& x,
        const ScalarType                  c0,
        const int                         level )
    {
        if ( c.size() != x.size() )
        {
            throw std::runtime_error( "VectorQ1Scalar::lincomb: c and x must have the same size" );
        }

        if ( x.size() == 0 )
        {
            kernels::common::set_constant( grid_data( level ), c0 );
        }
        else if ( x.size() == 1 )
        {
            kernels::common::lincomb( grid_data( level ), c0, c[0], x[0].grid_data( level ) );
        }
        else if ( x.size() == 2 )
        {
            kernels::common::lincomb(
                grid_data( level ), c0, c[0], x[0].grid_data( level ), c[1], x[1].grid_data( level ) );
        }
        else if ( x.size() == 3 )
        {
            kernels::common::lincomb(
                grid_data( level ),
                c0,
                c[0],
                x[0].grid_data( level ),
                c[1],
                x[1].grid_data( level ),
                c[2],
                x[2].grid_data( level ) );
        }
        else
        {
            throw std::runtime_error( "VectorQ1Scalar::lincomb: not implemented" );
        }
    }

    ScalarType dot_impl( const VectorQ1Vec& x, const int level ) const
    {
        return kernels::common::masked_dot_product( grid_data( level ), x.grid_data( level ), mask_data( level ) );
    }

    ScalarType max_abs_entry_impl( const int level ) const
    {
        return kernels::common::max_abs_entry( grid_data( level ) );
    }

    bool has_nan_impl( const int level ) const { return kernels::common::has_nan( grid_data( level ) ); }

    void swap_impl( VectorQ1Vec& other )
    {
        grid_data_.swap( other.grid_data_ );
        mask_data_.swap( other.mask_data_ );
    }

    void add_grid_data( const grid::Grid4DDataVec< ScalarType, VecDim >& grid_data, int level )
    {
        grid_data_.insert( { level, grid_data } );
    }

    void add_mask_data( const grid::Grid4DDataScalar< unsigned char >& mask_data, int level )
    {
        mask_data_.insert( { level, mask_data } );
    }

    grid::Grid4DDataVec< ScalarType, VecDim > grid_data( int level ) const
    {
        if ( !grid_data_.contains( level ) )
        {
            throw std::runtime_error( "VectorQ1Vec::grid_data: level not found" );
        }
        return grid_data_.at( level );
    }

    grid::Grid4DDataScalar< unsigned char > mask_data( int level ) const
    {
        if ( !mask_data_.contains( level ) )
        {
            throw std::runtime_error( "VectorQ1Scalar::mask_data: level not found" );
        }
        return mask_data_.at( level );
    }

  private:
    std::map< int, grid::Grid4DDataVec< ScalarType, VecDim > > grid_data_;
    std::map< int, grid::Grid4DDataScalar< unsigned char > >   mask_data_;
};

static_assert( VectorLike< VectorQ1Vec< double, 3 > > );

template < typename ValueType >
VectorQ1Scalar< ValueType > allocate_vector_q1_scalar(
    const std::string                     label,
    const grid::shell::DistributedDomain& distributed_domain,
    const int                             level )
{
    grid::Grid4DDataScalar< ValueType > grid_data(
        label,
        distributed_domain.subdomains().size(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_radially() );

    VectorQ1Scalar< ValueType > vector_q1_scalar;
    vector_q1_scalar.add_grid_data( grid_data, level );
    return vector_q1_scalar;
}

template < typename ValueType, int VecDim >
VectorQ1Vec< ValueType, VecDim > allocate_vector_q1_vec(
    const std::string                     label,
    const grid::shell::DistributedDomain& distributed_domain,
    const int                             level )
{
    grid::Grid4DDataVec< ValueType, VecDim > grid_data(
        label,
        distributed_domain.subdomains().size(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_radially() );

    VectorQ1Vec< ValueType, VecDim > vector_q1_vec;
    vector_q1_vec.add_grid_data( grid_data, level );
    return vector_q1_vec;
}

} // namespace terra::linalg