cmake_minimum_required(VERSION 3.21)

# Project name and version
project(terra-kokkos VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable verbose makefile output
set(CMAKE_VERBOSE_MAKEFILE ON)

# Find Kokkos
# find_package(Kokkos REQUIRED)

# Enable testing
enable_testing()

# Add the src and tests subdirectories
add_subdirectory(extern)
add_subdirectory(src)
add_subdirectory(tests)