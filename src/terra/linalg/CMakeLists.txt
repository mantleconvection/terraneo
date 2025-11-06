target_sources(terra
        PRIVATE
        linalg.hpp
        operator.hpp
        inv_diag_operator.hpp
        vector.hpp
        vector_q1.hpp
        vector_q1isoq2_q1.hpp)

add_subdirectory(solvers)
add_subdirectory(trafo)
add_subdirectory(util)