
#pragma once
#include <mpi.h>

namespace terra::mpi {

using MPIRank = int;

inline MPIRank rank()
{
    MPIRank rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    return rank;
}

inline int num_processes()
{
    int num_processes;
    MPI_Comm_size( MPI_COMM_WORLD, &num_processes );
    return num_processes;
}

} // namespace terra::mpi