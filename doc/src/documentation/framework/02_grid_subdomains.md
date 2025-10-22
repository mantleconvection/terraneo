# Grid structure and subdomains (part I - logical structure) {#grid-subdomains}

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