

#pragma once
#include "kokkos/kokkos_wrapper.hpp"
#include "mpi/mpi.hpp"

namespace terra::util {

/// @brief RAII approach to safely initialize MPI and Kokkos.
///
/// At the start of your main(), create this object to run MPI_Init/Finalize and to start the Kokkos scope.
///
/// Like this:
///
///      int main( int argc, char** argv)
///      {
///          // Make sure to not destroy it right away!
///          //     TerraScopeGuard( &argc, &argv );
///          // will not work! Name the thing!
///
///          TerraScopeGuard terra_scope_guard( &argc, &argv );
///
///          // Here goes your cool app/test. MPI and Kokkos are finalized automatically (even if you throw an
///          // exception).
///
///          // ...
///      } // Destructor handles stuff here.
class TerraScopeGuard
{
  public:
    TerraScopeGuard( int* argc, char*** argv )
    : mpi_scope_guard_( argc, argv )
    , kokkos_scope_guard_( *argc, *argv )
    {}

  private:
    mpi::MPIScopeGuard mpi_scope_guard_;
    Kokkos::ScopeGuard kokkos_scope_guard_;
};
} // namespace terra::util