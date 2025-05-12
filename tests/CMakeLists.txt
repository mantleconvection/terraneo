# Add test executable
add_executable(test_mesh test_mesh.cpp)

# Link Kokkos to the test target
target_link_libraries(test_mesh terra)

# Include directories for tests
target_include_directories(test_mesh PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Add a test
add_test(NAME test_mesh COMMAND test_mesh)


# Add test executable
add_executable(test_kokkos_smoke test_kokkos_smoke.cpp)

# Link Kokkos to the test target
target_link_libraries(test_kokkos_smoke terra Kokkos::kokkos)

# Include directories for tests
target_include_directories(test_kokkos_smoke PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Add a test
add_test(NAME test_kokkos_smoke COMMAND test_kokkos_smoke)