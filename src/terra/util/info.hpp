
#pragma once
#include "kokkos/kokkos_wrapper.hpp"
#include "table.hpp"

namespace terra::util {

inline Table info_table()
{
    Table table;

    const auto threads = Kokkos::num_threads();
    table.add_row( { { "key", "threads" }, { "value", threads } } );

    const auto devices = Kokkos::num_devices();
    table.add_row( { { "key", "devices" }, { "value", devices } } );

    const auto mpi_procs = mpi::num_processes();
    table.add_row( { { "key", "MPI processes" }, { "value", mpi_procs } } );

    return table;
}

} // namespace terra::util