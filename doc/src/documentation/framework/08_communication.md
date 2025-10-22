# Communication {#communication}

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

