#pragma once

#include <cmath>
#include <stdexcept>

#include "kokkos_wrapper.hpp"
#include "types.hpp"

namespace terra {

using SphericalShellSubMeshViewType = Kokkos::View< real_t** [3] >;

SphericalShellSubMeshViewType setup_coords_subdomain(
    int diamond_id,
    int global_refinements,
    int num_subdomains_per_side,
    int subdomain_i,
    int subdomain_j );

/// @brief Computes the vertex coordinates on a single diamond of the shell with radius 1.
SphericalShellSubMeshViewType setup_coords_classic( int ntan, int diamond_id );

} // namespace terra
