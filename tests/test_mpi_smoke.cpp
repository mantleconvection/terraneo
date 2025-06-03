

#include <iostream>
#include <mpi.h>

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );

    int world_rank, world_size;
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    std::cout << "Hello from rank " << world_rank << " out of " << world_size << " processes\n";

    MPI_Finalize();
    return 0;
}