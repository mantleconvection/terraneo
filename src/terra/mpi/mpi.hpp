
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

template < typename T >
MPI_Datatype mpi_datatype()
{
    static_assert( sizeof( T ) == 0, "No MPI datatype mapping for this type." );
    return MPI_DATATYPE_NULL;
}

template <>
inline MPI_Datatype mpi_datatype< char >()
{
    return MPI_CHAR;
}

template <>
inline MPI_Datatype mpi_datatype< signed char >()
{
    return MPI_SIGNED_CHAR;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned char >()
{
    return MPI_UNSIGNED_CHAR;
}

template <>
inline MPI_Datatype mpi_datatype< int >()
{
    return MPI_INT;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned int >()
{
    return MPI_UNSIGNED;
}

template <>
inline MPI_Datatype mpi_datatype< short >()
{
    return MPI_SHORT;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned short >()
{
    return MPI_UNSIGNED_SHORT;
}

template <>
inline MPI_Datatype mpi_datatype< long >()
{
    return MPI_LONG;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned long >()
{
    return MPI_UNSIGNED_LONG;
}

template <>
inline MPI_Datatype mpi_datatype< long long >()
{
    return MPI_LONG_LONG;
}

template <>
inline MPI_Datatype mpi_datatype< unsigned long long >()
{
    return MPI_UNSIGNED_LONG_LONG;
}

template <>
inline MPI_Datatype mpi_datatype< float >()
{
    return MPI_FLOAT;
}

template <>
inline MPI_Datatype mpi_datatype< double >()
{
    return MPI_DOUBLE;
}

template <>
inline MPI_Datatype mpi_datatype< long double >()
{
    return MPI_LONG_DOUBLE;
}

} // namespace terra::mpi