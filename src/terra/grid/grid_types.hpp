
#pragma once

#include "../kokkos/kokkos_wrapper.hpp"
#include "../types.hpp"

namespace terra::grid {

template < typename ScalarType >
using Grid1DDataScalar = Kokkos::View< ScalarType* >;

template < typename ScalarType >
using Grid2DDataScalar = Kokkos::View< ScalarType** >;

template < typename ScalarType >
using Grid3DDataScalar = Kokkos::View< ScalarType*** >;

template < typename ScalarType >
using Grid4DDataScalar = Kokkos::View< ScalarType**** >;

template < typename ScalarType, int VecDim >
using Grid1DDataVec = Kokkos::View< ScalarType* [VecDim] >;

template < typename ScalarType, int VecDim >
using Grid2DDataVec = Kokkos::View< ScalarType** [VecDim] >;

template < typename ScalarType, int VecDim >
using Grid3DDataVec = Kokkos::View< ScalarType*** [VecDim] >;

template < typename ScalarType, int VecDim >
using Grid4DDataVec = Kokkos::View< ScalarType**** [VecDim] >;

} // namespace terra::grid
