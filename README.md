# Terra 2.0

## Documentation

* radii extents per subdomain from inner to outer (BOUNDARY HANDLING IS FLAWED FOR MORE THAN ONE SUBDOMAIN IN RADIAL
  DIRECTION (r != r.extent(...)))

## TODO

- [ ] fix/document order of indices (radii first?)
- [ ] struct and more convenient package for radii and grid data on shell subdomain
- [ ] also provide dimensions or even range policies
- [ ] document VTK
- [ ] VTK quadratic linear wedge for curved domains?
- [ ] accept host data for VTK shell and radii right away
- [ ] refactor adding data slightly

## Q&A

### What kind of elements do we use?

Candidates:

* Q1-iso-Q2 / Q1 (hex)

  Code likely gets simpler, but we have to use some kind of curved geometry, likely a little more expensive(?)

* P1-iso-P2 / P1 (wedge)

  Similar to Terra, no need to use curved geom, slightly more complicated loops and possibly complicated integrals, but
  likely cheap

* P1-iso-P2 / P1 (tet)

  Subdivision of the cubes - sounds pretty complicated and I do not see a real benefit

### What kind of projection do we use (if any)?

### How do we compute integrals in general?

### Kokkos related

* compare 1D-copy with all kernels as a benchmark
* switch up the loop order
* atomics (atomic views), random access (read-only views), scatter view(?)
* SIMD later(?)