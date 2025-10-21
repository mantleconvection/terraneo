# Documentation {#main}

\if NOT_DOXYGEN_GENERATED

> It seems you are viewing this in plain text or with a standard Markdown renderer (e.g., on GitHub).
> 
> ❗️This file is best read in HTML format after generating the documentation via running `doxygen` inside of `doc/src/`.

\endif

## About

**TerraNeoX** is a mantle convection code based on [Kokkos](https://github.com/kokkos/kokkos) for performance portability.

\docseplarge

## Building

### Dependencies

Mandatory:

* MPI (e.g. OpenMPI)

Optional:

* CUDA (for GPU support)

### Compiling on the LMU systems (`cachemiss`, `memoryleak`) for usage with CUDA:

```
$ module load mpi.ompi
$ module load nvidia-hpc

$ mkdir terraneox-build

$ ll
terraneox/               # <== the cloned source code
terraneox-build/

$ cd terraneox-build

$ cmake ../terraneox/ -Kokkos_ENABLE_CUDA=ON

# Build tests
$ cd tests
$ make -j16
```

Note the capitalization: it must be `Kokkos_ENABLE_CUDA=ON`, NOT `KOKKOS_ENABLE_CUDA=ON`.

\docseplarge

## Project Structure

```
terraneox/
├── apps/                     # Applications (benchmarks, tools, ...) using the framework.
│   ├── benchmarks/           # Benchmarks ...
│   │   ├── performance/      # ... run time
│   │   └── physics/          # ... physics
│   └── tools/                # Tools (e.g., for visualization of meshes)
├── data/                     # Stuff that is not exactly library/framework source code
│   └── scripts/              # Scripts, e.g., for post-processing
│       └── plotting/         # Scripts for visualization, e.g., timing, radial profiles, ... 
├── doc/                      # Documentation
│   ├── documents/            # Documents that are not part of the generated documentation (e.g., Latex documents)
│   └── src/                  # Doxygen documentation
│       └── documentation/    # Doxygen documentation source (this page is likely here)
├── extern/                   # External libraries shipped with this repository
├── src/                      # Source code
└── tests/                    # Tests
```

\docseplarge

## Framework documentation

### Model / Partial differential equations

🏗️

\docsepsmall

### Grid structure and subdomains (part I - logical structure)

\note This section is just describing the logical organization of data. For details on memory layout / allocation
refer to the Kokkos section. Details on the construction of the thick spherical shell and communication are given
in the dedicated sections below.

All grid operations are performed on a set of hexahedral subdomains (block-structured grid).
The corresponding grid data is organized via 4- or 5 dimensional arrays.

```
    data( local_subdomain_id, node_x, node_y, node_r )                     // for scalar data
    data( local_subdomain_id, node_x, node_y, node_r, vector_entry )       // for vector-valued data 
```

The first index specifies the subdomain. Then the x, y, r coordinates of the nodes (we are using r instead of z because
we will for our application most of the time think in radial directions). For vector-valued data (think velocity vectors
stored at each node) a fifth index specifies the entry of the vector. In the code, the size of that fifth dimension is
typically a compile-time constant.

This layout forces all subdomains to be of equal size.

As described in the finite element section, we mostly use wedge elements. The nodes that span wedge elements have the
same spatial organization of nodes required for hexahedral elements.
The division of one hexahedron into two wedges is implicitly done in the compute kernels and is not represented by the
grid data structure.

(Since we are using linear Lagrangian basis functions, nodes directly correspond to coefficients. For the linear cases,
we can also use the same grid data structure for linear hexahedral finite elements. Such an extension is straightforward
and just requires respective kernels. One could even mix both - although it is not clear if that is mathematically
sound.)

As a convention, the hexahedral elements are split into two wedges diagonally from node (1, 0) to node (0, 1) as
follows:

```
Example of a 5x4 subdomain, that would be extruded in r-direction.
Each node 'o' either stores a scalar or a vector. 

        o---o---o---o---o
        |\  |\  |\  |\  |
        | \ | \ | \ | \ |
        o---o---o---o---o
        |\  |\  |\  |\  |
        | \ | \ | \ | \ |
        o---o---o---o---o
   ^    |\  |\  |\  |\  |
   |    | \ | \ | \ | \ |
  y|    o---o---o---o---o
   
        -->
        x
```

\docsepsmall

### Finite element discretization

The partial differential equations and their solutions are approximated using the finite element method.

We are using linear wedge elements for all spaces, unless specified otherwise. This is very similar to the
implementation in Terra.

\note
See helper functions and documentation in [integrands.hpp](@ref integrands.hpp) or the
[namespace terra::fe:wedge](@ref terra::fe::wedge) for details, and other, derived
quantities like gradients, Jacobians, determinants, etc.

Linear wedge (or prism) elements are formed by extruding a linear triangular element in the radial direction.
The base triangle lies in the lateral plane (parameterized by \f$\xi\f$,\f$\eta\f$), while the extrusion occurs along
the radial
coordinate \f$\zeta\f$.

\note
The provided functions for the computation of the gradients, Jacobians, etc. assume that we are working on a spherical
shell and accordingly work with respect to a forward map that maps the reference element onto a wedge with the two
triangular surfaces living on two shell-slices, and the connecting beams being radially extruded from the origin.

#### Geometry

Lateral reference coordinates:
\f[ \xi, \eta \in [0, 1] \f]

Radial reference coordinates:
\f[ \zeta \in [-1, 1] \f]

With
\f[ 0 \leq \xi + \eta \leq 1 \f]

#### Node enumeration

\code

Case I:

    radial_node_idx == radial_cell_idx + 1 (outer triangle of wedge):

    5
    |\
    | \
    3--4

\endcode

\code

Case II:

    radial_node_idx == radial_cell_idx (inner triangle of wedge):

    2
    |\
    | \
    0--1

\endcode

#### Shape functions

Lateral:

\f[
\begin{align}
N^\mathrm{lat}_0 = N^\mathrm{lat}_3 &= 1 - \xi - \eta \\
N^\mathrm{lat}_1 = N^\mathrm{lat}_4 &= \xi \\
N^\mathrm{lat}_2 = N^\mathrm{lat}_5 &= \eta
\end{align}
\f]

Radial:

\f[
\begin{align}
N^\mathrm{rad}_0 = N^\mathrm{rad}_1 = N^\mathrm{rad}_2 &= \frac{1}{2} ( 1 - \zeta ) \\
N^\mathrm{rad}_3 = N^\mathrm{rad}_4 = N^\mathrm{rad}_5 &= \frac{1}{2} ( 1 + \zeta ) \\
\end{align}
\f]

Full:

\f[
N_i = N^\mathrm{lat}_i N^\mathrm{rad}_i
\f]

#### Physical coordinates

\code
r_1, r_2 radii of bottom and top (r_1 < r_2)
p1_phy, p2_phy, p3_phy coords of triangle on unit sphere
\endcode

#### Spaces:

For the Stokes system we employ the stable (\f$P_1\f$-iso-\f$P_2\f$, \f$P_1\f$) finite element pairing, i.e., both
the velocity and pressure are discretized with linear wedge elements, with the velocity living on a grid with additional
refinement compared to the pressure grid.

\docsepsmall

### Linear algebra

🏗️

#### Coefficient vectors

🏗️

#### Operators

🏗️

#### Solvers

🏗️

#### Multigrid

🏗️

\docsepsmall

### Grid structure and subdomains (part II - Kokkos)

🏗️

\docsepsmall

### Thick spherical shell

The Earth mantle is approximated via a thick spherical shell \f$\Omega\f$ , i.e., a hollow sphere centered at the origin

\f[ \Omega = \{\mathbf{x} \in \mathbb{R}^3 : r_\mathrm{min} \leq \|\mathbf{x}\| \leq r_\mathrm{max} \} \f]

#### Mesh structure

A corresponding mesh is constructed by splitting the outer surface of \f$\Omega\f$ into 10 spherical diamonds that are
extruded towards (or equivalently away from) the origin.

The figures/videos below show the diamonds in a three-dimensional visualization (each diamond is refined 4 times in
lateral and 4 times in radial direction).

Single diamond (`diamond_id == 0`):
\htmlonly
<video width="960" controls>
<source src="diamond_animation.mp4" type="video/mp4">
</video>
\endhtmlonly
\image html figures/diamond_animation.mp4

Northern (`0 <= diamond_id <= 4`) and southern diamonds (`5 <= diamond_id <= 9`):
\htmlonly
<video width="960" controls>
<source src="north_south_animation.mp4" type="video/mp4">
</video>
\endhtmlonly
\image html figures/north_south_animation.mp4

Unfolding the surface partitioning, we can visualize the surface of the 10 spherical diamonds as a net that when curved
and pieced together recovers the spherical shell:

\image html figures/thick-spherical-shell-diamond-net.jpg

Indexing in radial direction always goes from the inner boundary to the outer boundary.
Note that the extrusion in radial direction is not visible from the net.

Each diamond can (optionally) be subdivided in lateral and radial direction. The radial refinement is straightforward.
After (uniform) lateral refinement, each subdomain can be associated with a globally unique identifying tuple

``` 
subdomain_id = (diamond_id, subdomain_x, subdomain_y, subdomain_r)
```

as illustrated in the figure below (for one refinement step in the lateral direction; note that the radial refinement
is not visible in the figure and indicated by the colon in the tuple):

\image html figures/thick-spherical-shell-subdomains.jpg

The `subdomain_id` is implemented in the class \ref terra::grid::shell::SubdomainInfo.

The information about the global structure is captured in the \ref terra::grid::shell::DomainInfo class.
That class does not compute any node coordinates. It just stores the refinement information, i.e., how many subdomains
are present for each diamond in either direction.
In the lateral direction, refinement currently has to be uniform.
In the radial direction, the concrete radii of the layers can be specified.
For more details refer to the documentation of \ref terra::grid::shell::DomainInfo.

\note You typically do not construct the \ref terra::grid::shell::DomainInfo class yourself. Instead, you use the
\ref terra::grid::shell::DistributedDomain class.

#### Local subdomains

Subdomains on the same MPI process are sorted by their global `subdomain_id` (it is sortable and globally unique)
and continuously assigned to an integer `local_subdomain_id` that ranges from 0 to the number of process-local
subdomains minus 1.
The `local_subdomain_id` is then the first index of the 4D (or 5D) data grids introduced above.

For instance, for a scalar data array `data` the expression

```
    data( 3, 55, 20, 4 )
```

accesses the node with

```
    local_subdomain_id =  3
    x_index            = 55
    y_index            = 20
    r_index            =  4
```

The mapping from the `subdomain_id` (type `SubdomainInfo`) to the `local_subdomain_id` (type `int`) is performed during
set up and stored together with other information in the corresponding \ref terra::grid::shell::DistributedDomain
instance.
More details are found in the parallelization section.

#### Node coordinates

The concrete coordinates of the nodes are computed with two functions:

* \ref terra::grid::shell::subdomain_unit_sphere_single_shell_coords - computes the "lateral cartesian coordinates"
  of all nodes, i.e., computes the cartesian coordinates of a single shell of nodes with radius 1 and returns them in
  a 4D array `coords_shell( local_subdomain_id, x_index, y_index, cartesian_coord )`.
* \ref terra::grid::shell::subdomain_shell_radii - computes the radii and stores them in a 2D array
  `coords_radii( local_subdomain_id, r_index )`
  The cartesian coordinate of a node `( local_subdomain_id, x_index, y_index, r_index )` can then be computed via

```
    Vec3 cartesian_coords;
    cartesian_coords( 0 ) = coords_shell( local_subdomain_id, x_index, y_index, 0 );
    cartesian_coords( 1 ) = coords_shell( local_subdomain_id, x_index, y_index, 1 );
    cartesian_coords( 2 ) = coords_shell( local_subdomain_id, x_index, y_index, 2 );
    return cartesian_coords * coords_radii( local_subdomain_id, r_index );
```

This is implemented in \ref terra::grid::shell::coords.
The radius is obviously just `coords_radii( local_subdomain_id, r_index )`.

\docsepsmall

### Parallelization

🏗️

\docsepsmall

### Communication

For operations that do not only work locally (such as matrix-vector products) information has to be communicated
across boundaries of neighboring subdomains.
At subdomain boundaries, mesh nodes are duplicated: the same mesh node exists on multiple subdomains.

Generally, we **assume that the values at the mesh nodes are holding the correct values whenever entering linear
algebra building blocks**. That means we have to ensure that data is communicated **after** computations such as
matrix-vector multiplications.

A map of neighboring subdomains and metadata is generated via the class \ref terra::grid::shell::SubdomainNeighborhood.
That is done internally in the \ref terra::grid::shell::DistributedDomain.

#### Vector-vector operations (excluding dot products)

Vector-vector operations (such as daxpy etc.) do not require any communication as long as the duplicated nodes are
updated for each subdomain. Technically that means we perform redundant computations at the benefit of avoiding
communication and having very simple kernels (you can just loop over the entire subdomain without conditionals).

#### Dot products (and other reductions)

The computation of dot products and other reductions must be performed carefully since we must not include duplicated
nodes twice. To ensure that, we store a flag field (mostly called `mask_data` in the code) that assigns in a setup phase
(typically once at the start of the program) an `owned` flag to exactly one of the duplicated nodes. The dot
product kernel (or any kind of reduction) then skips the nodes that are not marked as `owned`. See also the
[section on masks / flag fields](#flag-fields-and-masks).

#### Assigning random values

One has to take care when assigning random values to all nodes in parallel since after such a randomization, duplicated
nodes must not have different values.

#### Matrix-vector operations

Due to the linearity of the matrix-free finite element matrix-vector multiplication kernels, we can simply (again,
assuming the source vector is already updated) apply the kernel locally and then sum up the values on duplicated nodes
and write that sum to all duplicated nodes.

#### Communication details

After a kernel has been executed, additive communication is performed using two buffers (send and recv) for
each interface that a local subdomain has with another subdomain.
The boundary data is written to the send buffer and sent to the receiver side (via MPI).
After receiving the data from the other subdomains, that data is added from the recv buffers to the respective local
subdomains.
Subdomain faces only have at most one interface with a different subdomain, whereas there can be more than one neighbor
for edges and vertices of a subdomain.

In some cases, the data has to be rotated in some way to match the nodes at the receiver side.
The convention here is that data is packed without rotation and is properly rotated during unpacking.

By coincidence, the subdomain structure of the thick spherical shell only ever requires a small subset of
rotations. Note that vertex-vertex interfaces require no rotation since only data of a single
node is sent. Edge-edge interfaces only require checking for one rotation type (either we unpack forward, or backward).

Face-face interfaces technically have a larger space of possible rotations.
Let's look at all cases to see why there are only a few types to consider:

##### Radial direction

This is the simplest case because the iteration pattern is the same in x and y direction. So the pattern is

```
   send_data( local_subdomain_id_sender, x, y, FIXED_TOP_OR_BOTTOM )
=> buffer( x, y )
=> recv_data( local_subdomain_id_recver, x, y, FIXED_BOTTOM_OR_TOP )
```

##### Lateral direction (same diamond)

If both subdomains are in the same diamond, lateral communication is also straightforward.
We keep the radial dimension the second one in the buffer.

```
Either 

   send_data( local_subdomain_id_sender, x, FIXED_START_OR_END, r )
=> buffer( x, r )
=> recv_data( local_subdomain_id_recver, x, FIXED_END_OR_START, r )

or 

   send_data( local_subdomain_id_sender, FIXED_START_OR_END, y, r )
=> buffer( y, r )
=> recv_data( local_subdomain_id_recver, FIXED_END_OR_START, y, r )
```

##### Lateral direction (at diamond-diamond interfaces)

It turns out that we only need a handful of simple rotations due to the structure of the diamonds.
This is nice and makes the communication really easy to implement for our special case.

```

NORTH-NORTH and SOUTH-SOUTH
===========================

Communication between diamonds at the same poles.

=> No rotation necessary. Just "sort into the other coordinate". 

d_0( 0, :, r ) = d_1( :, 0, r )
d_1( 0, :, r ) = d_2( :, 0, r )
d_2( 0, :, r ) = d_3( :, 0, r )
d_3( 0, :, r ) = d_4( :, 0, r )
d_4( 0, :, r ) = d_0( :, 0, r )

d_5( 0, :, r ) = d_6( :, 0, r )
d_6( 0, :, r ) = d_7( :, 0, r )
d_7( 0, :, r ) = d_8( :, 0, r )
d_8( 0, :, r ) = d_9( :, 0, r )
d_9( 0, :, r ) = d_5( :, 0, r )

E.g.:

   send_data_d_1( local_subdomain_id_sender, 0, y, r )
=> buffer( y, r )
=> recv_data_d_2( local_subdomain_id_recver, y, 0, r )

--------------------------------------------------------------------------------

NORTH-SOUTH and SOUTH-NORTH
===========================

Communication between diamonds at different poles.

=> Rotate the x/y direction during unpacking.

d_0( :, end, r ) = d_5( end, :, r )
d_1( :, end, r ) = d_6( end, :, r )
d_2( :, end, r ) = d_7( end, :, r )
d_3( :, end, r ) = d_8( end, :, r )
d_4( :, end, r ) = d_9( end, :, r )

d_5( :, end, r ) = d_1( end, :, r )
d_6( :, end, r ) = d_2( end, :, r )
d_7( :, end, r ) = d_3( end, :, r )
d_8( :, end, r ) = d_4( end, :, r )
d_9( :, end, r ) = d_0( end, :, r )

E.g.:

   send_data_d_1( local_subdomain_id_sender, x, y_size - 1, r )
=> buffer( x, r )
=> recv_data_d_6( local_subdomain_id_recver, x_size - 1, y_size - 1 - x, r )
                                                         ---------------
                                                         ^^^^^^^^^^^^^^^
                                                          rotating here
```

While the number of rotations is small, deriving the neighborhood of a subdomain is a bit tricky for all types
of interfaces.
It depends on the boundary type, the subdomain index, and whether the subdomain boundary
is located at the boundary of a diamond.
The logic is implemented in the \ref terra::grid::shell::SubdomainNeighborhood class and executed once during the
construction (which is done in the \ref terra::grid::shell::DistributedDomain class).

\docsepsmall

### Flag fields and masks {#flag-fields-and-masks}

🏗️

\docsepsmall

### Boundary conditions

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

\docsepsmall

### IO

#### Tabular data

🏗️

#### XDMF

\note Putting the string 'VTK' here in case you are looking for it via full-text search. We are using XDMF instead.

🏗️

#### Radial profiles

🏗️

\docseplarge

## TODO

### Big features (definitely required - order not clear)

- [x] advection-diffusion discretization / solver
- [x] advection-diffusion boundary handling
- [ ] advection-diffusion source term handling (must add SUPG term in linear form)
- [x] ~~GMRES~~ BiCGStab(l)
- [x] BDF2 (not yet in a dedicated function, see test_heat_eq)
- [x] multigrid (some notes: a) we need higher operator quad degree than constant (not sure where exactly: diagonal,
  fine-level, everywhere?), b) two-grid V(10, 10) looks ok, otherwise with multigrid we do not get perfectly h-ind. conv
  rates., I
  suppose we need Galerkin coarse grid operators maybe)
- [x] MPI parallel execution (multi-GPU, multi-node CPU)
- [x] intra-diamond subdomain communication (then also test/fix boundary handling in operators/tests - subdomain
  boundaries are sometimes treated as domain boundaries even if they are not)
- [ ] variable viscosity
- [ ] plates
- [ ] free-slip
- [ ] compressible Stokes
- [x] FGMRES (BiCGStab works well mostly - but seems to randomly produce NaNs occasionally (not 100% sure if related to
  the solver, but it is very likely))
- [ ] Galerkin coarsening
- [ ] iterative refinement
- [ ] spherical harmonics helper
- [ ] radial profiles loader
- [ ] checkpoints (re-use XDMF bin files!)
- [ ] return unmanaged views from SubdomainNeighborhoodSendRecvBuffer that point to contiguous memory per rank, add
  another getter to the pointer of that array and then pass that to MPI_Send/Recv instead
- [x] radial layer data assimilation
- [x] timing(tree?)
- [x] ~~compress VTK(?)~~ XDMF output (binary, actual float data, and ~~/or~~ ~~HDF5/ADIOS2~~ with a single mesh file)
- [x] CLI interface / parameter files

### Small features / improvements (not necessarily / maybe required)

- [ ] cube-like test case (this may require some new FE features)
- [ ] performance engineering
- [x] ~~curved wedges~~ the wedges are curved (unlike I assumed when writing this)
- [ ] particles(?)
- [x] matrix export / assembly (implemented for debugging - not for actual use)
- [ ] CPU SIMD kernels
- [ ] adapt solver ctor like in FGMRES (I think that is the best design)

### Documentation / cleanup / refactoring

- [ ] Github page
- [x] Doxygen
- [ ] Doxygen page
- [x] move mask stuff that generalizes away from shell namespace
- [ ] sort out what is spherical shell specific and what is not