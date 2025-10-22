# Finite element discretization {#finite-elements}

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
