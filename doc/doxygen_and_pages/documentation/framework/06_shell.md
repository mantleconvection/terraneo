# Thick spherical shell {#shell}

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

#### Local subdomains

Each diamond can (optionally) be subdivided in lateral and radial direction. 
Below, two angles of an exploded diamond (diamond ID 0, one lateral and one radial subdomain refinement level) are shown:

\image html figures/diamond_subdomains_explosion_0.png

\image html figures/diamond_subdomains_explosion_1.png

After radial and lateral refinement of the diamonds, each subdomain can be associated with a globally unique identifying tuple

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

```cpp
    Vec3 cartesian_coords;
    cartesian_coords( 0 ) = coords_shell( local_subdomain_id, x_index, y_index, 0 );
    cartesian_coords( 1 ) = coords_shell( local_subdomain_id, x_index, y_index, 1 );
    cartesian_coords( 2 ) = coords_shell( local_subdomain_id, x_index, y_index, 2 );
    return cartesian_coords * coords_radii( local_subdomain_id, r_index );
```

This is implemented in \ref terra::grid::shell::coords.
The radius is obviously just `coords_radii( local_subdomain_id, r_index )`.