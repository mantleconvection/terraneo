add_library(terra
    spherical_shell_mesh.cpp
    spherical_shell_mesh.hpp)

target_include_directories(terra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(terra Kokkos::kokkos)

