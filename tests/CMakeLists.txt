# Global property to collect all test targets.
set_property(GLOBAL PROPERTY TERRA_TEST_TARGETS "")

# Adds a test.
function(add_terra_test name)
    add_executable(${name} ${name}.cpp)
    if (MPI_FOUND)
        target_link_libraries(${name} terra MPI::MPI_CXX)
    else ()
        target_link_libraries(${name} terra)
    endif ()
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${name} COMMAND ${name})

    # Add to global test list
    get_property(tests GLOBAL PROPERTY TERRA_TEST_TARGETS)
    list(APPEND tests ${name})
    set_property(GLOBAL PROPERTY TERRA_TEST_TARGETS "${tests}")
endfunction()

# Adding all the tests.
add_terra_test(test_flag_field_unique_vertices)
add_terra_test(test_mesh)
add_terra_test(test_laplace)
add_terra_test(test_laplace_cg)
add_terra_test(test_laplace_minres)
add_terra_test(test_laplace_richardson)
add_terra_test(test_mpi_smoke)
add_terra_test(test_stokes_minres)
add_terra_test(test_vector_laplace_cg)

# Adding target for all the tests.
get_property(ALL_TERRA_TESTS GLOBAL PROPERTY TERRA_TEST_TARGETS)
add_custom_target(tests DEPENDS ${ALL_TERRA_TESTS})