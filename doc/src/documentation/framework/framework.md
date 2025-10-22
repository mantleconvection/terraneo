# Framework documentation {#framework-documentation}

Below is a list of the documentation pages for the framework introducing various concepts.

| Section                                                       | Description                                                                            |
|---------------------------------------------------------------|----------------------------------------------------------------------------------------|
| [Model / Partial Differential Equations](01_model_pde.md)     | Core PDE formulation and governing equations.                                          |
| [Grid Structure (part I - Subdomains)](02_grid_subdomains.md) | Logical grid layout.                                                                   |
| [Finite Element Discretization](03_finite_elements.md)        | Overview of the FEM approach and element definitions.                                  |
| [Linear Algebra](04_linear_algebra.md)                        | Matrix and vector representations, solvers, and preconditioners (including multigrid). |
| [Grid Structure (part II - Kokkos)](05_grid_kokkos.md)        | Kokkos-based data structures.                                                          |
| [Thick Spherical Shell](06_shell.md)                          | Details on the thick spherical shell mesh.                                             |
| [Parallelization](07_parallelization.md)                      | Parallel execution patterns.                                                           |
| [Communication](08_communication.md)                          | Data exchange patterns and MPI communication strategies.                               |
| [Flag Fields and Masks](09_flag_fields_and_masks.md)          | Use of masks and flag grids for selective operations and boundary tagging.             |
| [Boundary Conditions](10_boundary_conditions.md)              | Definition and application of boundary conditions.                                     |
| [Input and Output](11_input_output.md)                        | Data formats, visualization, radial profiles, checkpoints, logging, etc.               |
