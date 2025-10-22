# Boundary conditions {#boundary-conditions}

#### Strong Dirichlet boundary condition enforcement

\note
See
also: [strong_algebraic_dirichlet_enforcement_poisson_like()](@ref terra::fe::strong_algebraic_dirichlet_enforcement_poisson_like)
and similar functions in that file.

<p></p>

\note
A great explanation is given in [Wolfgang Bangerth's
lectures](https://www.math.colostate.edu/~bangerth/videos.676.21.65.html).

Consider the linear problem

\f[ Lu = f \f]

with Dirichlet boundary conditions.

We approach the elimination as follows (assuming interpolating FE spaces).

Let \f$ A \f$ be the "Neumann" operator matrix of \f$ L \f$, i.e., we do not treat the boundaries any differently and
just execute the volume integrals.

1. Interpolate Dirichlet boundary conditions into a vector \f$ g \f$.
2. Compute \f$ g_A \gets A g \f$.
3. Compute \f$ g_D \gets \mathrm{diag}(A) g \f$
4. Set the rhs to \f$ b_\text{elim} = b - g_A \f$,
   where \f$ b \f$ is the assembled rhs vector for the homogeneous problem
   (the result of evaluating the linear form into a vector or of the matrix-vector product of a vector
   \f$ f_\text{vec} \f$ where
   the rhs function \f$ f \f$ has been interpolated into, and then \f$ b = M f_\text{vec} \f$ (\f$ M \f$ being the mass
   matrix))
5. Set the rhs \f$ b_elim \f$ at the boundary nodes to \f$ g_D \f$, i.e.
   \f$ b_\text{elim} \gets g_D \f$ on the Dirichlet boundary
6. Solve
   \f$ A_\text{elim} x = b_\text{elim} \f$
   where \f$ A_\text{elim} \f$ is \f$ A \f$, but with all off-diagonal entries in the same row/col as a boundary node
   set to zero.
   This feature has to be supplied by the operator implementation.
   In a matrix-free context, we have to adapt the element matrix \f$ A_\text{local} \f$ accordingly by (symmetrically)
   zeroing
   out all the off-diagonals (row and col) that correspond to a boundary node. But we keep the diagonal intact.
   We still have \f$ \mathrm{diag}(A) = \mathrm{diag}(A_\text{elim}) \f$ .
7. \f$ x \f$ is the solution of the original problem. No boundary correction should be necessary.

All of this is covered
by [strong_algebraic_dirichlet_enforcement_poisson_like()](@ref terra::fe::strong_algebraic_dirichlet_enforcement_poisson_like)
and similar functions. What needs to be supplied are the operators \f$ A \f$, \f$ \mathrm{diag}(A) \f$, and
\f$ A_\text{elim} \f$. The main "difficulty" is to implement \f$ A_\text{elim} \f$ in a matrix-free kernel.
This can be achieved if one realizes that zeroing out the offdiagonals of the global matrix is equivalently done
by zeroing out the respective entries of the local element matrices.

#### Free-slip

🏗️

#### Plate boundaries

🏗️
