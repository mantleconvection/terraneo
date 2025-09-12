target_sources(terra
        PRIVATE
        divergence.hpp
        gradient.hpp
        laplace.hpp
        laplace_batched.hpp
        laplace_no_matrix.hpp
        laplace_simple.hpp
        mass.hpp
        prolongation_linear.hpp
        restriction_linear.hpp
        stokes.hpp
        unsteady_advection_diffusion_supg.hpp
        vector_laplace.hpp
        vector_mass.hpp
        zero.hpp)

