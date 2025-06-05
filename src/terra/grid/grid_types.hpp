
#pragma once

#include "../kokkos/kokkos_wrapper.hpp"
#include "../types.hpp"

namespace terra::grid {

template < typename ScalarType >
using Grid0DDataScalar = Kokkos::View< ScalarType >;

template < typename ScalarType >
using Grid1DDataScalar = Kokkos::View< ScalarType* >;

template < typename ScalarType >
using Grid2DDataScalar = Kokkos::View< ScalarType** >;

template < typename ScalarType >
using Grid3DDataScalar = Kokkos::View< ScalarType*** >;

template < typename ScalarType >
using Grid4DDataScalar = Kokkos::View< ScalarType**** >;

template < typename ScalarType, int VecDim >
using Grid0DDataVec = Kokkos::View< ScalarType[VecDim] >;

template < typename ScalarType, int VecDim >
using Grid1DDataVec = Kokkos::View< ScalarType* [VecDim] >;

template < typename ScalarType, int VecDim >
using Grid2DDataVec = Kokkos::View< ScalarType** [VecDim] >;

template < typename ScalarType, int VecDim >
using Grid3DDataVec = Kokkos::View< ScalarType*** [VecDim] >;

template < typename ScalarType, int VecDim >
using Grid4DDataVec = Kokkos::View< ScalarType**** [VecDim] >;

enum class BoundaryVertex : int
{
    V_000 = 0, // (x=0, y=0, r=0)
    V_100,
    V_010,
    V_110,
    V_001,
    V_101,
    V_011,
    V_111,
};

enum class BoundaryEdge : int
{
    E_X00, // edge along x, y=0, r=0, (:, 0, 0) in slice notation
    E_X10,
    E_X01,
    E_X11,

    E_0Y0, // (0, :, 0) in slice notation
    E_1Y0,
    E_0Y1,
    E_1Y1,

    E_00R,
    E_10R,
    E_01R,
    E_11R,
};

enum class BoundaryFace : int
{
    F_XY0, // facet orthogonal to r, r=0
    F_XY1,

    F_X0R,
    F_X1R,

    F_0YR,
    F_1YR,
};

constexpr bool is_edge_boundary_radial( const BoundaryEdge id )
{
    return id == BoundaryEdge::E_00R || id == BoundaryEdge::E_10R || id == BoundaryEdge::E_01R ||
           id == BoundaryEdge::E_11R;
}

constexpr bool is_face_boundary_normal_to_radial_direction( const BoundaryFace id )
{
    return id == BoundaryFace::F_XY0 || id == BoundaryFace::F_XY1;
}

constexpr std::array all_local_vertex_ids = {
    BoundaryVertex::V_000,
    BoundaryVertex::V_100,
    BoundaryVertex::V_010,
    BoundaryVertex::V_110,
    BoundaryVertex::V_001,
    BoundaryVertex::V_101,
    BoundaryVertex::V_011,
    BoundaryVertex::V_111 };

constexpr std::array all_local_edge_ids = {
    BoundaryEdge::E_X00,
    BoundaryEdge::E_X10,
    BoundaryEdge::E_X01,
    BoundaryEdge::E_X11,

    BoundaryEdge::E_0Y0,
    BoundaryEdge::E_1Y0,
    BoundaryEdge::E_0Y1,
    BoundaryEdge::E_1Y1,

    BoundaryEdge::E_00R,
    BoundaryEdge::E_10R,
    BoundaryEdge::E_01R,
    BoundaryEdge::E_11R,
};

constexpr std::array all_local_face_ids = {
    BoundaryFace::F_XY0,
    BoundaryFace::F_XY1,
    BoundaryFace::F_X0R,
    BoundaryFace::F_X1R,
    BoundaryFace::F_0YR,
    BoundaryFace::F_1YR,
};

// String conversion functions
constexpr std::string to_string( BoundaryVertex v )
{
    switch ( v )
    {
    case BoundaryVertex::V_000:
        return "V_000";
    case BoundaryVertex::V_100:
        return "V_100";
    case BoundaryVertex::V_010:
        return "V_010";
    case BoundaryVertex::V_110:
        return "V_110";
    case BoundaryVertex::V_001:
        return "V_001";
    case BoundaryVertex::V_101:
        return "V_101";
    case BoundaryVertex::V_011:
        return "V_011";
    case BoundaryVertex::V_111:
        return "V_111";
    default:
        return "<unknown LocalBoundaryVertex>";
    }
}

constexpr std::string to_string( BoundaryEdge e )
{
    switch ( e )
    {
    case BoundaryEdge::E_X00:
        return "E_X00";
    case BoundaryEdge::E_X10:
        return "E_X10";
    case BoundaryEdge::E_X01:
        return "E_X01";
    case BoundaryEdge::E_X11:
        return "E_X11";
    case BoundaryEdge::E_0Y0:
        return "E_0Y0";
    case BoundaryEdge::E_1Y0:
        return "E_1Y0";
    case BoundaryEdge::E_0Y1:
        return "E_0Y1";
    case BoundaryEdge::E_1Y1:
        return "E_1Y1";
    case BoundaryEdge::E_00R:
        return "E_00R";
    case BoundaryEdge::E_10R:
        return "E_10R";
    case BoundaryEdge::E_01R:
        return "E_01R";
    case BoundaryEdge::E_11R:
        return "E_11R";
    default:
        return "<unknown LocalBoundaryEdge>";
    }
}

constexpr std::string to_string( BoundaryFace f )
{
    switch ( f )
    {
    case BoundaryFace::F_XY0:
        return "F_XY0";
    case BoundaryFace::F_XY1:
        return "F_XY1";
    case BoundaryFace::F_X0R:
        return "F_X0R";
    case BoundaryFace::F_X1R:
        return "F_X1R";
    case BoundaryFace::F_0YR:
        return "F_0YR";
    case BoundaryFace::F_1YR:
        return "F_1YR";
    default:
        return "<unknown LocalBoundaryFace>";
    }
}

inline std::ostream& operator<<( std::ostream& os, BoundaryVertex v )
{
    return os << to_string( v );
}

inline std::ostream& operator<<( std::ostream& os, BoundaryEdge e )
{
    return os << to_string( e );
}

inline std::ostream& operator<<( std::ostream& os, BoundaryFace f )
{
    return os << to_string( f );
}

} // namespace terra::grid
