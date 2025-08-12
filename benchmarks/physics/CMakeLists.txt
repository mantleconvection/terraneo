function(add_terra_physics_benchmark name)
    add_executable(${name} ${name}.cpp)
    if (MPI_FOUND)
        target_link_libraries(${name} terra MPI::MPI_CXX)
    else ()
        target_link_libraries(${name} terra)
    endif ()
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_terra_physics_benchmark(benchmark_convection_shell_isovisc_ns_ns)

