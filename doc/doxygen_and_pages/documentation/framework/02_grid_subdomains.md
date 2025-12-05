# Grids, subdomains, allocation, kernels, Kokkos, etc. {#grid-subdomains}

\note This section is describing how to work with grids. For details on
* **thick spherical shell meshes** → [shell section](#shell),
* **parallelization** → [parallelization section](#parallelization),
* **communication & subdomain boundaries** → [communication section](#communication).

All domains are defined by a set of hexahedral (possibly curved) subdomains, forming a block-structured grid.
Each subdomain is successively refined in all three (local) coordinate directions (x, y, r) to get hexahedral 
elements.

\note We mostly work with wedge elements and therefore split these hexahedral elements into two wedges.
      Refer to the [finite element section](#finite-elements) for details.

Most data is stored per subdomain, in either 1D, 2D, or 3D arrays (per subdomain). 
The data items are usually scalars, vectors, or tensors.

The grid dimensions can mean different things depending on the application.
Most often the coefficients directly correspond to the node index in each spatial dimension, but can also sometimes
be used to represent a cell index instead.

### Allocation

Grid data is allocated directly using the `Kokkos::View` class. For convenience, we provide type aliases for the most
commonly used grid data types in \ref grid_types.hpp. You can allocate grid data views as follows:

```cpp
    // Example for a 4D grid (3D grid per subdomain, scalar nodal data, float64)
    GridData4DScalar< double > data_scalar( "my_data_scalar", num_local_subdomains, num_nodes_x, num_nodes_y, num_nodes_r );
    
    // Example for a 5D grid (3D grid per subdomain, vector nodal data, float32)
    GridData4DVec< float, 3 > data_vec( "my_data_vec", num_local_subdomains, num_nodes_x, num_nodes_y, num_nodes_r );
    
    // Example for a 2D grid (1D grid per subdomain, scalar nodal data, float32)
    // Used to store radii of shells on each subdomain.
    GridData2DScalar< float > data_radii( "my_data_radii", num_local_subdomains, num_nodes_r );
```

Using `Kokkos::View` here is straightforward, practical, and allows for easy interoperability (you can just write your
standard Kokkos kernels; there is no wrapper around the view). Refer to the Kokkos documentation for more details.

### Accessing entries

`Kokkos::View`s implement `operator()` for accessing entries.
For example, data is accessed like so:

```cpp
    auto val_scalar = data_scalar( local_subdomain_id, node_x, node_y, node_r );     // for scalar data, 3D, nodal
    
    data_vec( local_subdomain_id, node_x, node_y, node_r, vector_entry ) = val_vec;  // for vector-valued data, 3D, nodal
    
    data_radii( local_subdomain_id, node_r ) += 0.1;                                 // for radial data, 1D, nodal
```

The first index specifies the local subdomain index (more details in the sections of [shell partitioning](#shell)
and [parallelization](#parallelization)). Then the x, y, r coordinates of the nodes (we are using r instead of z because
we will for our application most of the time think in radial directions). For vector-valued data (think velocity vectors
stored at each node) a fifth index specifies the entry of the vector. In the code, the size of that fifth dimension is
typically a compile-time constant.

\note Some comments:
* This layout forces all subdomains to be of equal size. But it is convenient as it enables straightforward
[parallelization](#parallelization) over all local subdomains.
* Storing nodal data for each subdomain results in overlapping data at subdomain boundaries. Details on how this
issue is solved are described in the [communication section](#communication).
* In the example above, we store the radii data on each subdomain. Technically, this is redundant, as the radii are
equal on different subdomains that are at the same "depth". But it is just convenient to store it this way for use in 
kernels.

As described in the [finite element section](#finite-elements), we mostly use wedge elements. 
The nodes that span wedge elements have the same spatial organization of nodes required for hexahedral elements.
The division of one hexahedron into two wedges is implicitly done in the compute kernels and is not represented by the
grid data structure.

(Since we are mostly using linear Lagrangian basis functions, nodes generally directly correspond to coefficients. 
We can also use the same grid data structure for linear hexahedral finite elements. 
Such an extension is straightforward and just requires respective kernels. 
One could even mix both (think: Laplacian with wedge elements, mass matrix from hex elements), although 
it is not clear if that is mathematically sound.)

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

### Kernels

You can simply write standard Kokkos kernels on the grid data. However, you should use existing kernels from the library
wherever possible. Some special cases:

#### Working with coefficient vectors (you want to do linear algebra)

Use the classes

* \ref terra::linalg::VectorQ1Scalar, 
* \ref terra::linalg::VectorQ1Vec,
* \ref terra::linalg::VectorQ1IsoQ2Q1, 

for scalar, vector valued, and mixed coefficient vectors for Stokes. 
Also, use the provided functions for the \ref terra::linalg::VectorLike concept in 
\ref vector.hpp and take a look at the [linear algebra section](#linear-algebra).
Those already take care of the overlapping data at subdomain boundaries.

Examples:
```cpp
      using terra::linalg::dot;
      using terra::linalg::lincomb;

      VectorQ1Scalar< double > x( "x", ... );
      VectorQ1Scalar< double > y( "y", ... );
      VectorQ1Scalar< double > z( "z", ... );

      // s = (x.y)
      double s = dot( x, y );
      
      // z_j = 42.0 * x_j - 0.99 * y_j + 123.0; 
      lincomb( z, { 42.0, -0.99 }, { x, y }, 123.0 );
      
      // infinity norm
      double n = norm_inf( z );
      
      // There are many more functions and overloads ...
```

#### Working on grids directly

Sometimes your data are not exactly vectors but you want to work on grids.
Before writing your own kernels, you should check if there is a kernel that already does what you want
in \ref grid_operations.hpp.

Examples:
```cpp
      
    using terra::kernels::common::extract_vector_component;
    using terra::kernels::common::count_masked;    

    GridData4DScalar< double > x( "x", ... );    
    GridData4DVec< double, 3 > vec( "vec", ... );
    GridData4DScalar< NodeOwnershipFlag > mask( "mask", ... );

    // copy one component (second component here) of a vector-valued grid into a scalar grid
    extract_vector_component( x, vec, 1 );
    
    // count the number of "owned" nodes (excludes duplicated nodes)
    auto n = count_masked< long >( mask, NodeOwnershipFlag::OWNED );
    
    // There are many more functions and overloads ...

```


#### Writing your own kernels

You will likely have to write your own kernels for some cases, most notably when interpolating functions into FE spaces.
For this, you can look at the existing kernels in tests, apps, and \ref grid_operations.hpp to get some inspiration and 
to the Kokkos documentation, too.

For FE coefficient vectors like \ref terra::linalg::VectorQ1Scalar you can access the underlying data using the 
corresponding getters. Those vectors essentially just wrap two types of `Kokkos::View`: one for the coefficients,
and one for masking data that indicates which coefficients are "owned" by the subdomain.
Also refer to the sections on [communication](#communication) and [flags and mask data](#flag-fields-and-masks)

Examples:
```cpp

// Using a lambda (for simple kernels)

grid::Grid4DDataScalar< ScalarType > x( ... );

Kokkos::parallel_for(
    "interpolate 1 if local subdomain index is 0 (for whatever reason)",
    Kokkos::MDRangePolicy( { 0, 0, 0, 0 }, { x.extent( 0 ), x.extent( 1 ), x.extent( 2 ), x.extent( 3 ) } ),
    KOKKOS_LAMBDA( int local_subdomain_id, int i, int j, int k ) {
        if ( local_subdomain_id == 0 )
        {
            x( local_subdomain_id, i, j, k ) = 1.0;
        } 
    } );

// -------------------------------------------------------------------------------------- //

// Using a functor (for more complex kernels and reusability)

struct SolutionInterpolator
{
    Grid3DDataVec< double, 3 >                         grid_;
    Grid2DDataScalar< double >                         radii_;
    Grid4DDataScalar< double >                         data_;
    Grid4DDataScalar< grid::shell::ShellBoundaryFlag > mask_;
    bool                                               only_boundary_;

    SolutionInterpolator(
        const Grid3DDataVec< double, 3 >&                         grid,
        const Grid2DDataScalar< double >&                         radii,
        const Grid4DDataScalar< double >&                         data,
        const Grid4DDataScalar< grid::shell::ShellBoundaryFlag >& mask,
        bool                                                      only_boundary )
    : grid_( grid )
    , radii_( radii )
    , data_( data )
    , mask_( mask )
    , only_boundary_( only_boundary )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const int local_subdomain_id, const int x, const int y, const int r ) const
    {
        const dense::Vec< double, 3 > coords = grid::shell::coords( local_subdomain_id, x, y, r, grid_, radii_ );
        
        const double value = ( 1.0 / 2.0 ) * Kokkos::sin( 2 * coords( 0 ) ) * Kokkos::sinh( coords( 1 ) );

        const bool on_boundary =
            util::has_flag( mask_( local_subdomain_id, x, y, r ), grid::shell::ShellBoundaryFlag::BOUNDARY );

        if ( !only_boundary_ || on_boundary )
        {
            data_( local_subdomain_id, x, y, r ) = value;
        }
    }
};

// ...

Kokkos::parallel_for(
    "solution interpolation (only on boundary)",
    local_domain_md_range_policy_nodes( domain ),
    SolutionInterpolator( coords_shell, coords_radii, solution.grid_data(), boundary_mask_data, true ) );
```

