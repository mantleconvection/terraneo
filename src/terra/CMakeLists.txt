add_library(terra)

target_link_libraries(terra PUBLIC Kokkos::kokkos MPI::MPI_CXX)
target_include_directories(terra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_sources(terra
        PRIVATE
        library_dummy.cpp
        types.hpp)

add_subdirectory(communication)
add_subdirectory(dense)
add_subdirectory(eigen)
add_subdirectory(fe)
add_subdirectory(grid)
add_subdirectory(kokkos)
add_subdirectory(linalg)
add_subdirectory(mpi)
add_subdirectory(shell)
add_subdirectory(util)
add_subdirectory(visualization)