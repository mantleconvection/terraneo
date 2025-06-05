

#include <iostream>
#include <mpi.h>

#include "terra/kokkos/kokkos_wrapper.hpp"

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    {
        int world_rank, world_size;
        MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
        MPI_Comm_size( MPI_COMM_WORLD, &world_size );

        std::cout << "Hello from rank " << world_rank << " out of " << world_size << " processes\n";

        Kokkos::View< double* > src( "src", 10 );
        Kokkos::View< double* > dst( "dst", 10 );

        // Init src
        Kokkos::parallel_for(
            "fill_src", Kokkos::RangePolicy( 0, 10 ), KOKKOS_LAMBDA( const int i ) {
                src( i ) = static_cast< double >( i );
            } );

        MPI_Send( src.data(), src.span(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );
        MPI_Recv( dst.data(), dst.span(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );

        auto dst_host = Kokkos::create_mirror_view( dst );
        Kokkos::deep_copy( dst_host, dst );

        for ( int i = 0; i < dst_host.extent( 0 ); i++ )
        {
            std::cout << dst_host( i ) << std::endl;
        }
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}