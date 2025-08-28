#pragma once

#include <cmath>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <vector>

#include "../grid_types.hpp"
#include "../terra/kokkos/kokkos_wrapper.hpp"
#include "dense/vec.hpp"
#include "mpi/mpi.hpp"

namespace terra::grid::shell {

template < std::floating_point T >
std::vector< T > uniform_shell_radii( T r_min, T r_max, int num_shells )
{
    if ( num_shells < 2 )
    {
        throw std::runtime_error( "Number of shells must be at least 2." );
    }
    std::vector< T > radii;
    radii.reserve( num_shells );
    const T r_step = ( r_max - r_min ) / ( num_shells - 1 );
    for ( int i = 0; i < num_shells; ++i )
    {
        radii.push_back( r_min + i * r_step );
    }

    // Set boundary exactly.
    radii[num_shells - 1] = r_max;

    return radii;
}

template < std::floating_point T >
T min_radial_h( const std::vector< T >& shell_radii )
{
    if ( shell_radii.size() < 2 )
    {
        throw std::runtime_error( " Need at least two shells to compute h. " );
    }

    T min_dist = std::numeric_limits< T >::infinity();
    for ( size_t i = 1; i < shell_radii.size(); ++i )
    {
        T d = std::abs( shell_radii[i] - shell_radii[i - 1] );
        if ( d < min_dist )
        {
            min_dist = d;
        }
    }
    return min_dist;
}

/// Struct to hold the coordinates of the four base corners
/// and the number of intervals N = ntan - 1.
template < std::floating_point T >
struct BaseCorners
{
    using Vec3 = dense::Vec< T, 3 >;

    Vec3 p00; // Coordinates for global index (0, 0)
    Vec3 p0N; // Coordinates for global index (0, N)
    Vec3 pN0; // Coordinates for global index (N, 0)
    Vec3 pNN; // Coordinates for global index (N, N)
    int  N;   // Number of intervals = ntan - 1. Must be power of 2.

    // Constructor for convenience (optional)
    BaseCorners( Vec3 p00_ = {}, Vec3 p0N_ = {}, Vec3 pN0_ = {}, Vec3 pNN_ = {}, int N_ = 0 )
    : p00( p00_ )
    , p0N( p0N_ )
    , pN0( pN0_ )
    , pNN( pNN_ )
    , N( N_ )
    {}
};

// Memoization cache type: maps (i, j) index pair to computed coordinates
template < std::floating_point T >
using MemoizationCache = std::map< std::pair< int, int >, dense::Vec< T, 3 > >;

///@brief Computes the coordinates for a specific node (i, j) in the final refined grid.
///       Uses recursion and memoization, sourcing base points from the BaseCorners struct.
///
/// @param i Row index (0 to corners.N).
/// @param j Column index (0 to corners.N).
/// @param corners Struct containing base corner coordinates and N = ntan - 1.
/// @param cache Cache to store/retrieve already computed nodes.
/// @return Vec3 Coordinates of the node (i, j) on the unit sphere.
///
template < std::floating_point T >
dense::Vec< T, 3 > compute_node_recursive( int i, int j, const BaseCorners< T >& corners, MemoizationCache< T >& cache )
{
    using Vec3 = dense::Vec< T, 3 >;

    // --- Get N and validate indices ---
    const int N    = corners.N;
    const int ntan = N + 1;
    if ( i < 0 || i >= ntan || j < 0 || j >= ntan )
    {
        throw std::out_of_range( "Requested node index out of range." );
    }
    if ( N <= 0 || ( N > 0 && ( N & ( N - 1 ) ) != 0 ) )
    {
        throw std::invalid_argument( "BaseCorners.N must be a positive power of 2." );
    }

    // --- 1. Check Cache ---
    auto cache_key = std::make_pair( i, j );
    auto it        = cache.find( cache_key );
    if ( it != cache.end() )
    {
        return it->second; // Already computed
    }

    // --- 2. Base Case: Use BaseCorners struct ---
    if ( i == 0 && j == 0 )
    {
        cache[cache_key] = corners.p00;
        return corners.p00;
    }
    if ( i == 0 && j == N )
    {
        cache[cache_key] = corners.p0N;
        return corners.p0N;
    }
    if ( i == N && j == 0 )
    {
        cache[cache_key] = corners.pN0;
        return corners.pN0;
    }
    if ( i == N && j == N )
    {
        cache[cache_key] = corners.pNN;
        return corners.pNN;
    }

    // --- 3. Recursive Step: Find creation level l2 and apply rules ---

    // Find the smallest half-stride l2 (power of 2, starting from 1)
    // such that (i, j) was NOT present on the grid with stride l = 2*l2.
    // A point is present if both i and j are multiples of l.
    int l2 = 1;
    int l  = 2; // l = 2*l2
    while ( l <= N )
    { // Iterate through possible creation strides l
        if ( i % l != 0 || j % l != 0 )
        {
            // Found the level 'l' where (i, j) was created.
            // l2 is l/2.
            break;
        }
        // If execution reaches here, (i, j) exists on grid with stride l.
        // Check the next finer level.
        l2 *= 2; // or l2 <<= 1;
        l = 2 * l2;
    }

    if ( l > N && l2 == N )
    { // If loop finished without breaking, l=2N, l2=N
        // This condition should only be true for the base corners already handled.
        // If we reach here for non-corner points, something is wrong.
        throw std::logic_error( "Internal logic error: Failed to find creation level for non-corner point." );
    }

    Vec3 p1, p2; // Parent points

    // Identify the rule used at creation level l=2*l2, based on relative position
    if ( i % l == 0 && j % l == l2 )
    {
        // Rule 1: Horizontal midpoint ("rows" loop)
        // i is multiple of l, j is halfway (offset l2)
        p1 = compute_node_recursive( i, j - l2, corners, cache );
        p2 = compute_node_recursive( i, j + l2, corners, cache );
    }
    else if ( i % l == l2 && j % l == 0 )
    {
        // Rule 2: Vertical midpoint ("columns" loop)
        // j is multiple of l, i is halfway (offset l2)
        p1 = compute_node_recursive( i - l2, j, corners, cache );
        p2 = compute_node_recursive( i + l2, j, corners, cache );
    }
    else if ( i % l == l2 && j % l == l2 )
    {
        // Rule 3: Diagonal midpoint ("diagonals" loop)
        // Both i and j are halfway (offset l2)
        p1 = compute_node_recursive( i - l2, j + l2, corners, cache );
        p2 = compute_node_recursive( i + l2, j - l2, corners, cache );
    }
    else
    {
        // This should not happen if the logic for finding l is correct and (i,j) is not a base corner.
        // The checks i%l and j%l should cover all non-zero remainder possibilities correctly.
        // If i%l==0 and j%l==0, the while loop should have continued.
        throw std::logic_error( "Internal logic error: Point does not match any creation rule." );
    }

    // Calculate Euclidean midpoint
    Vec3 mid = p1 + p2;

    // Normalize to project onto the unit sphere
    Vec3 result = mid.normalized();

    // --- 4. Store result in cache and return ---
    cache[cache_key] = result;
    return result;
}

/// @brief Generates coordinates for a rectangular subdomain of the refined spherical grid.
///
///@param subdomain_coords_host a properly sized host-allocated view that is filled with the coordinates of the points
///@param corners Struct containing the base corner points and N = ntan - 1.
///@param i_start_incl Starting row index (inclusive) of the subdomain (global index).
///@param i_end_incl Ending row index (inclusive) of the subdomain (global index).
///@param j_start_incl Starting column index (inclusive) of the subdomain (global index).
///@param j_end_incl Ending column index (inclusive) of the subdomain (global index).
///
///@return Kokkos::View<T**[3], Kokkos::HostSpace> Host view containing coordinates
///        for the subdomain. Dimensions are ((i_end_incl - 1) - i_start, (j_end_incl - 1) - j_start).
///
template < std::floating_point T >
void compute_subdomain(
    const typename Grid3DDataVec< T, 3 >::HostMirror& subdomain_coords_host,
    int                                               subdomain_idx,
    const BaseCorners< T >&                           corners,
    int                                               i_start_incl,
    int                                               i_end_incl,
    int                                               j_start_incl,
    int                                               j_end_incl )
{
    using Vec3 = dense::Vec< T, 3 >;

    const int i_start = i_start_incl;
    const int j_start = j_start_incl;
    const int i_end   = i_end_incl + 1;
    const int j_end   = j_end_incl + 1;

    // --- Input Validation ---
    const int N    = corners.N;
    const int ntan = N + 1; // Derive ntan from N in corners struct
    if ( i_start < 0 || i_end > ntan || i_start >= i_end || j_start < 0 || j_end > ntan || j_start >= j_end )
    {
        throw std::invalid_argument( "Invalid subdomain boundaries." );
    }
    if ( N <= 0 || ( N > 0 && ( N & ( N - 1 ) ) != 0 ) )
    {
        throw std::invalid_argument( "BaseCorners.N must be a positive power of 2." );
    }

    // --- Initialization ---
    const size_t subdomain_rows = i_end - i_start;
    const size_t subdomain_cols = j_end - j_start;

    if ( subdomain_coords_host.extent( 1 ) != subdomain_rows || subdomain_coords_host.extent( 2 ) != subdomain_cols )
    {
        throw std::runtime_error( "Invalid subdomain dimensions in compute_subdomain()." );
    }

    MemoizationCache< T > cache; // Each subdomain computation gets its own cache

    // --- Compute nodes within the subdomain ---
    for ( int i = i_start; i < i_end; ++i )
    {
        for ( int j = j_start; j < j_end; ++j )
        {
            // Compute the node coordinates using the recursive function
            Vec3 coords = compute_node_recursive( i, j, corners, cache ); // Pass corners struct

            // Store in the subdomain view (adjusting indices)
            subdomain_coords_host( subdomain_idx, i - i_start, j - j_start, 0 ) = coords( 0 );
            subdomain_coords_host( subdomain_idx, i - i_start, j - j_start, 1 ) = coords( 1 );
            subdomain_coords_host( subdomain_idx, i - i_start, j - j_start, 2 ) = coords( 2 );
        }
    }
}

template < std::floating_point T >
void unit_sphere_single_shell_subdomain_coords(
    const typename Grid3DDataVec< T, 3 >::HostMirror& subdomain_coords_host,
    int                                               subdomain_idx,
    int                                               diamond_id,
    int                                               ntan,
    int                                               i_start_incl,
    int                                               i_end_incl,
    int                                               j_start_incl,
    int                                               j_end_incl )
{
    // Coordinates of the twelve icosahedral nodes of the base grid
    real_t i_node[12][3];

    // Association of the ten diamonds to the twelve icosahedral nodes
    //
    // For each diamond we store the indices of its vertices on the
    // icosahedral base grid in this map. Ordering: We start with the
    // pole and proceed in counter-clockwise fashion.
    int d_node[10][4];

    // -----------------------------------------
    //  Initialise the twelve icosahedral nodes
    // -----------------------------------------

    // the pentagonal nodes on each "ring" are given in anti-clockwise ordering
    real_t fifthpi = real_c( 0.4 * std::asin( 1.0 ) );
    real_t w       = real_c( 2.0 * std::acos( 1.0 / ( 2.0 * std::sin( fifthpi ) ) ) );
    real_t cosw    = std::cos( w );
    real_t sinw    = std::sin( w );
    real_t phi     = 0.0;

    // North Pole
    i_node[0][0] = 0.0;
    i_node[0][1] = 0.0;
    i_node[0][2] = +1.0;

    // South Pole
    i_node[11][0] = 0.0;
    i_node[11][1] = 0.0;
    i_node[11][2] = -1.0;

    // upper ring
    for ( int k = 1; k <= 5; k++ )
    {
        phi          = real_c( 2.0 ) * ( real_c( k ) - real_c( 0.5 ) ) * fifthpi;
        i_node[k][0] = sinw * std::cos( phi );
        i_node[k][1] = sinw * std::sin( phi );
        i_node[k][2] = cosw;
    }

    // lower ring
    for ( int k = 1; k <= 5; k++ )
    {
        phi              = real_c( 2.0 ) * ( real_c( k ) - 1 ) * fifthpi;
        i_node[k + 5][0] = sinw * std::cos( phi );
        i_node[k + 5][1] = sinw * std::sin( phi );
        i_node[k + 5][2] = -cosw;
    }

    // ----------------------------------------------
    // Setup internal index maps for mesh generation
    // ----------------------------------------------

    // Map icosahedral node indices to diamonds (northern hemisphere)
    d_node[0][0] = 0;
    d_node[0][1] = 5;
    d_node[0][2] = 6;
    d_node[0][3] = 1;
    d_node[1][0] = 0;
    d_node[1][1] = 1;
    d_node[1][2] = 7;
    d_node[1][3] = 2;
    d_node[2][0] = 0;
    d_node[2][1] = 2;
    d_node[2][2] = 8;
    d_node[2][3] = 3;
    d_node[3][0] = 0;
    d_node[3][1] = 3;
    d_node[3][2] = 9;
    d_node[3][3] = 4;
    d_node[4][0] = 0;
    d_node[4][1] = 4;
    d_node[4][2] = 10;
    d_node[4][3] = 5;

    // Map icosahedral node indices to diamonds (southern hemisphere)
    d_node[5][0] = 11;
    d_node[5][1] = 7;
    d_node[5][2] = 1;
    d_node[5][3] = 6;
    d_node[6][0] = 11;
    d_node[6][1] = 8;
    d_node[6][2] = 2;
    d_node[6][3] = 7;
    d_node[7][0] = 11;
    d_node[7][1] = 9;
    d_node[7][2] = 3;
    d_node[7][3] = 8;
    d_node[8][0] = 11;
    d_node[8][1] = 10;
    d_node[8][2] = 4;
    d_node[8][3] = 9;
    d_node[9][0] = 11;
    d_node[9][1] = 6;
    d_node[9][2] = 5;
    d_node[9][3] = 10;

    // ------------------------
    //  Meshing of unit sphere
    // ------------------------

    // "left" and "right" w.r.t. d_node depend on hemisphere
    int L, R;
    if ( diamond_id < 5 )
    {
        L = 1;
        R = 3;
    }
    else
    {
        R = 1;
        L = 3;
    }

    BaseCorners< T > corners;
    corners.N = ntan - 1;

    // Insert coordinates of four nodes of this icosahedral diamond for each dim.
    for ( int i = 0; i < 3; ++i )
    {
        corners.p00( i ) = i_node[d_node[diamond_id][0]][i];
        corners.pN0( i ) = i_node[d_node[diamond_id][L]][i];
        corners.pNN( i ) = i_node[d_node[diamond_id][2]][i];
        corners.p0N( i ) = i_node[d_node[diamond_id][R]][i];
    }

    return compute_subdomain(
        subdomain_coords_host, subdomain_idx, corners, i_start_incl, i_end_incl, j_start_incl, j_end_incl );
}

template < std::floating_point T >
void unit_sphere_single_shell_subdomain_coords(
    const typename Grid3DDataVec< T, 3 >::HostMirror& subdomain_coords_host,
    int                                               subdomain_idx,
    int                                               diamond_id,
    int                                               global_refinements,
    int                                               num_subdomains_per_side,
    int                                               subdomain_i,
    int                                               subdomain_j )
{
    const auto elements_per_side = 1 << global_refinements;
    const auto ntan              = elements_per_side + 1;

    const auto elements_subdomain_base = elements_per_side / num_subdomains_per_side;
    const auto elements_remainder      = elements_per_side % num_subdomains_per_side;

    const auto elements_in_subdomain_i = elements_subdomain_base + ( subdomain_i < elements_remainder ? 1 : 0 );
    const auto elements_in_subdomain_j = elements_subdomain_base + ( subdomain_j < elements_remainder ? 1 : 0 );

    const auto start_i = subdomain_i * elements_subdomain_base + std::min( subdomain_i, elements_remainder );
    const auto start_j = subdomain_j * elements_subdomain_base + std::min( subdomain_j, elements_remainder );

    const auto end_i = start_i + elements_in_subdomain_i;
    const auto end_j = start_j + elements_in_subdomain_j;

    unit_sphere_single_shell_subdomain_coords< T >(
        subdomain_coords_host, subdomain_idx, diamond_id, ntan, start_i, end_i, start_j, end_j );
}

/// @brief (Sortable) Identifier for a single subdomain of a diamond.
///
/// Carries the diamond ID, and the subdomain index (x, y, r) inside the diamond.
/// Is globally unique (also in parallel settings).
/// Does not carry information about the refinement of a subdomain.
class SubdomainInfo
{
  public:
    SubdomainInfo()
    : diamond_id_( -1 )
    , subdomain_x_( -1 )
    , subdomain_y_( -1 )
    , subdomain_r_( -1 )
    {}

    SubdomainInfo( int diamond_id, int subdomain_x, int subdomain_y, int subdomain_r )
    : diamond_id_( diamond_id )
    , subdomain_x_( subdomain_x )
    , subdomain_y_( subdomain_y )
    , subdomain_r_( subdomain_r )
    {}

    /// @brief Diamond that subdomain is part of.
    int diamond_id() const { return diamond_id_; }

    /// @brief Subdomain index in lateral x-direction (local to the diamond).
    int subdomain_x() const { return subdomain_x_; }

    /// @brief Subdomain index in lateral y-direction (local to the diamond).
    int subdomain_y() const { return subdomain_y_; }

    /// @brief Subdomain index in the radial direction (local to the diamond).
    int subdomain_r() const { return subdomain_r_; }

    bool operator<( const SubdomainInfo& other ) const
    {
        return std::tie( diamond_id_, subdomain_r_, subdomain_y_, subdomain_x_ ) <
               std::tie( other.diamond_id_, other.subdomain_r_, other.subdomain_y_, other.subdomain_x_ );
    }

    bool operator==( const SubdomainInfo& other ) const
    {
        return std::tie( diamond_id_, subdomain_r_, subdomain_y_, subdomain_x_ ) ==
               std::tie( other.diamond_id_, other.subdomain_r_, other.subdomain_y_, other.subdomain_x_ );
    }

    /// @brief Scrambles the four indices (diamond ID, x, y, r) into a single integer.
    int global_id() const
    {
        if ( diamond_id_ >= 10 )
        {
            throw std::logic_error( "Diamond ID must be less than 10." );
        }

        if ( subdomain_x_ > 511 || subdomain_y_ > 511 || subdomain_r_ > 511 )
        {
            throw std::logic_error( "Subdomain indices too large." );
        }

        return ( diamond_id_ << 27 ) | ( subdomain_r_ << 18 ) | ( subdomain_y_ << 9 ) | ( subdomain_x_ );
    }

  private:
    /// Diamond that subdomain is part of.
    int diamond_id_;

    /// Subdomain index in lateral x-direction (local to the diamond).
    int subdomain_x_;

    /// Subdomain index in lateral y-direction (local to the diamond).
    int subdomain_y_;

    /// Subdomain index in radial direction.
    int subdomain_r_;
};

inline std::ostream& operator<<( std::ostream& os, const SubdomainInfo& si )
{
    os << "Diamond ID: " << si.diamond_id();
    return os;
}

inline mpi::MPIRank subdomain_to_rank_all_root( const SubdomainInfo& subdomain_info )
{
    return 0;
}

inline mpi::MPIRank subdomain_to_rank_distribute_full_diamonds( const SubdomainInfo& subdomain_info )
{
    const auto n = mpi::num_processes();

    const int size      = 10 / n;
    const int remainder = 10 % n;
    const int d         = subdomain_info.diamond_id();

    if ( d < ( size + 1 ) * remainder )
    {
        return d / ( size + 1 );
    }

    return remainder + ( d - ( size + 1 ) * remainder ) / size;
}

/// @brief Information about the thick spherical shell mesh.
///
/// The thick spherical shell is built from ten spherical diamonds. The diamonds are essentially curved hexahedra.
/// The number of cells in lateral directions is required to be a power of 2, the number of cells in the radial
/// direction can be chosen arbitrarily (though a power of two allows for maximally deep multigrid hierarchies).
///
/// Each diamond can be subdivided into subdomains (in all three directions) for better parallel distribution (each
/// process can only operate on one or more entire subdomains).
///
/// This class holds data such as
/// - the shell radii,
/// - the number of subdomains in each direction (on each diamond),
/// - the number of nodes per subdomain in each direction (including overlapping nodes where two or more subdomains
///   meet).
///
/// Note that all subdomains always have the same shape.
///
/// Since the global number of cells in a diamond in lateral and radial direction does not need to match, and since
/// the number of cells in radial direction does not even need to be a power of two (although it is a good idea to
/// choose it that way), this class computes the maximum number of coarsening steps (which is equivalent to the number
/// of "refinement levels") dynamically. Thus, a bad choice for the number of radial layers may result in a mesh that
/// cannot be coarsened at all.
///
/// This class has no notion of parallel distribution. For that refer to the DistributedDomain class.
///
class DomainInfo
{
  public:
    DomainInfo() = default;

    /// @brief Constructs a thick spherical shell with one subdomain per diamond (10 subdomains total) and uniformly
    /// distributed radial shells.
    ///
    /// Note: a 'shell' is a spherical 2D manifold in 3D space (it is thin),
    ///       a 'layer' is defined as the volume between two 'shells' (it is thick)
    ///
    /// @param diamond_lateral_refinement_level number of lateral diamond refinements
    /// @param r_min inner radius
    /// @param r_max outer radius
    /// @param num_uniform_layers number of layers (uniformly spaced using r_min and r_max)
    DomainInfo( int diamond_lateral_refinement_level, double r_min, double r_max, int num_uniform_layers )
    : diamond_lateral_refinement_level_( diamond_lateral_refinement_level )
    , radii_( uniform_shell_radii( r_min, r_max, num_uniform_layers + 1 ) )
    , num_subdomains_in_lateral_direction_( 1 )
    , num_subdomains_in_radial_direction_( 1 )
    {
        const int num_layers = num_uniform_layers;
        if ( num_layers % num_subdomains_in_radial_direction_ != 0 )
        {
            throw std::invalid_argument(
                "Number of layers must be divisible by number of subdomains in radial direction." );
        }
    }

    /// @brief The "maximum refinement level" of the subdomains.
    ///
    /// This (non-negative) number is essentially indicating how many times a subdomain can be uniformly coarsened.
    int subdomain_max_refinement_level() const
    {
        const auto max_refinement_level_lat =
            std::countr_zero( static_cast< unsigned >( subdomain_num_nodes_per_side_laterally() - 1 ) );
        const auto max_refinement_level_rad =
            std::countr_zero( static_cast< unsigned >( subdomain_num_nodes_radially() - 1 ) );

        return std::min( max_refinement_level_lat, max_refinement_level_rad );
    }

    int diamond_lateral_refinement_level() const { return diamond_lateral_refinement_level_; }

    const std::vector< double >& radii() const { return radii_; }

    int num_subdomains_per_diamond_side() const { return num_subdomains_in_lateral_direction_; }

    int num_subdomains_in_radial_direction() const { return num_subdomains_in_radial_direction_; }

    /// @brief Equivalent to calling subdomain_num_nodes_per_side_laterally( subdomain_refinement_level() )
    int subdomain_num_nodes_per_side_laterally() const
    {
        const int num_cells_per_diamond_side   = 1 << diamond_lateral_refinement_level();
        const int num_cells_per_subdomain_side = num_cells_per_diamond_side / num_subdomains_per_diamond_side();
        const int num_nodes_per_subdomain_side = num_cells_per_subdomain_side + 1;
        return num_nodes_per_subdomain_side;
    }

    /// @brief Equivalent to calling subdomain_num_nodes_radially( subdomain_refinement_level() )
    int subdomain_num_nodes_radially() const
    {
        const int num_layers               = radii_.size() - 1;
        const int num_layers_per_subdomain = num_layers / num_subdomains_in_radial_direction_;
        return num_layers_per_subdomain + 1;
    }

    /// @brief Number of nodes in the lateral direction of a subdomain on the passed level.
    ///
    /// The level must be non-negative. The finest level is given by subdomain_max_refinement_level().
    int subdomain_num_nodes_per_side_laterally( const int level ) const
    {
        if ( level < 0 )
        {
            throw std::invalid_argument( "Level must be non-negative." );
        }

        if ( level > subdomain_max_refinement_level() )
        {
            throw std::invalid_argument( "Level must be less than or equal to max subdomain refinement level." );
        }

        const int coarsening_steps = subdomain_max_refinement_level() - level;
        return ( ( subdomain_num_nodes_per_side_laterally() - 1 ) >> coarsening_steps ) + 1;
    }

    /// @brief Number of nodes in the radial direction of a subdomain on the passed level.
    ///
    /// The level must be non-negative. The finest level is given by subdomain_max_refinement_level().
    int subdomain_num_nodes_radially( const int level ) const
    {
        if ( level < 0 )
        {
            throw std::invalid_argument( "Level must be non-negative." );
        }

        if ( level > subdomain_max_refinement_level() )
        {
            throw std::invalid_argument( "Level must be less than or equal to subdomain refinement level." );
        }

        const int coarsening_steps = subdomain_max_refinement_level() - level;
        return ( ( subdomain_num_nodes_radially() - 1 ) >> coarsening_steps ) + 1;
    }

    std::vector< SubdomainInfo >
        local_subdomains( const std::function< mpi::MPIRank( const SubdomainInfo& ) >& subdomain_to_rank ) const
    {
        const auto rank = mpi::rank();

        std::vector< SubdomainInfo > subdomains;
        for ( int diamond_id = 0; diamond_id < 10; diamond_id++ )
        {
            for ( int x = 0; x < num_subdomains_per_diamond_side(); x++ )
            {
                for ( int y = 0; y < num_subdomains_per_diamond_side(); y++ )
                {
                    for ( int r = 0; r < num_subdomains_in_radial_direction_; r++ )
                    {
                        SubdomainInfo subdomain( diamond_id, x, y, r );

                        if ( subdomain_to_rank( subdomain ) == rank )
                        {
                            subdomains.push_back( subdomain );
                        }
                    }
                }
            }
        }

        if ( subdomains.empty() )
        {
            throw std::logic_error( "No local subdomains found on rank " + std::to_string( rank ) + "." );
        }

        return subdomains;
    }

  private:
    /// Number of times each diamond is refined laterally in each direction.
    int diamond_lateral_refinement_level_;

    /// Shell radii.
    std::vector< double > radii_;

    /// Number of subdomains per diamond (for parallel partitioning) in the lateral direction (at least 1).
    int num_subdomains_in_lateral_direction_;

    /// Number of subdomains per diamond (for parallel partitioning) in the radial direction (at least 1).
    int num_subdomains_in_radial_direction_;
};

/// @brief Neighborhood information of a single subdomain.
///
/// Holds information such as the MPI ranks of the neighboring subdomains, and their orientation.
/// Required for communication (packing, unpacking, sending, receiving 'ghost-layer' data).
class SubdomainNeighborhood
{
  public:
    using NeighborSubdomainTupleVertex = std::tuple< SubdomainInfo, BoundaryVertex, mpi::MPIRank >;
    using NeighborSubdomainTupleEdge   = std::tuple< SubdomainInfo, BoundaryEdge, mpi::MPIRank >;
    using NeighborSubdomainTupleFace   = std::tuple< SubdomainInfo, BoundaryFace, mpi::MPIRank >;

    SubdomainNeighborhood() = default;

    SubdomainNeighborhood(
        const DomainInfo&                                            domain_info,
        const SubdomainInfo&                                         subdomain_info,
        const std::function< mpi::MPIRank( const SubdomainInfo& ) >& subdomain_to_rank )
    {
        setup_neighborhood( domain_info, subdomain_info, subdomain_to_rank );
    }

    const std::map< BoundaryVertex, std::vector< NeighborSubdomainTupleVertex > >& neighborhood_vertex() const
    {
        return neighborhood_vertex_;
    }

    const std::map< BoundaryEdge, std::vector< NeighborSubdomainTupleEdge > >& neighborhood_edge() const
    {
        return neighborhood_edge_;
    }

    const std::map< BoundaryFace, NeighborSubdomainTupleFace >& neighborhood_face() const { return neighborhood_face_; }

  private:
    void setup_neighborhood(
        const DomainInfo&                                            domain_info,
        const SubdomainInfo&                                         subdomain_info,
        const std::function< mpi::MPIRank( const SubdomainInfo& ) >& subdomain_to_rank )
    {
        if ( domain_info.num_subdomains_per_diamond_side() != 1 ||
             domain_info.num_subdomains_in_radial_direction() != 1 )
        {
            throw std::logic_error( "Neighborhood setup only implemented for full diamonds." );
        }

        // Setup faces.
        const int diamond_id = subdomain_info.diamond_id();

        // Node equivalences: part one - communication between diamonds at the same poles

        // d_0( 0, :, r ) = d_1( :, 0, r )
        // d_1( 0, :, r ) = d_2( :, 0, r )
        // d_2( 0, :, r ) = d_3( :, 0, r )
        // d_3( 0, :, r ) = d_4( :, 0, r )
        // d_4( 0, :, r ) = d_0( :, 0, r )

        // d_5( 0, :, r ) = d_6( :, 0, r )
        // d_6( 0, :, r ) = d_7( :, 0, r )
        // d_7( 0, :, r ) = d_8( :, 0, r )
        // d_8( 0, :, r ) = d_9( :, 0, r )
        // d_9( 0, :, r ) = d_5( :, 0, r )

        // Node equivalences: part two - communication between diamonds at different poles

        // d_0( :, end, r ) = d_5( end, :, r )
        // d_1( :, end, r ) = d_6( end, :, r )
        // d_2( :, end, r ) = d_7( end, :, r )
        // d_3( :, end, r ) = d_8( end, :, r )
        // d_4( :, end, r ) = d_9( end, :, r )

        // d_5( :, end, r ) = d_1( end, :, r )
        // d_6( :, end, r ) = d_2( end, :, r )
        // d_7( :, end, r ) = d_3( end, :, r )
        // d_8( :, end, r ) = d_4( end, :, r )
        // d_9( :, end, r ) = d_0( end, :, r )

        switch ( diamond_id )
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
            // Part I
            neighborhood_face_[BoundaryFace::F_0YR] = {
                SubdomainInfo( ( diamond_id + 1 ) % 5, 0, 0, 0 ), BoundaryFace::F_X0R, -1 };
            neighborhood_face_[BoundaryFace::F_X0R] = {
                SubdomainInfo( ( diamond_id + 4 ) % 5, 0, 0, 0 ), BoundaryFace::F_0YR, -1 };

            // Part II
            neighborhood_face_[BoundaryFace::F_X1R] = {
                SubdomainInfo( diamond_id + 5, 0, 0, 0 ), BoundaryFace::F_1YR, -1 };
            neighborhood_face_[BoundaryFace::F_1YR] = {
                SubdomainInfo( ( diamond_id + 4 ) % 5 + 5, 0, 0, 0 ), BoundaryFace::F_X1R, -1 };
            break;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
            // Part I
            neighborhood_face_[BoundaryFace::F_0YR] = {
                SubdomainInfo( ( diamond_id + 1 ) % 5 + 5, 0, 0, 0 ), BoundaryFace::F_X0R, -1 };
            neighborhood_face_[BoundaryFace::F_X0R] = {
                SubdomainInfo( ( diamond_id - 1 ) % 5 + 5, 0, 0, 0 ), BoundaryFace::F_0YR, -1 };

            // Part II
            neighborhood_face_[BoundaryFace::F_X1R] = {
                SubdomainInfo( ( diamond_id - 4 ) % 5, 0, 0, 0 ), BoundaryFace::F_1YR, -1 };
            neighborhood_face_[BoundaryFace::F_1YR] = {
                SubdomainInfo( diamond_id - 5, 0, 0, 0 ), BoundaryFace::F_X1R, -1 };
            break;
        default:
            throw std::logic_error( "Invalid diamond id." );
        }

        // Now only the edges at the poles that are not already part of the faces remain.

        switch ( diamond_id )
        {
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
            // North Pole.
            neighborhood_edge_[BoundaryEdge::E_00R] = {
                { SubdomainInfo( ( diamond_id + 2 ) % 5, 0, 0, 0 ), BoundaryEdge::E_00R, -1 },
                { SubdomainInfo( ( diamond_id + 3 ) % 5, 0, 0, 0 ), BoundaryEdge::E_00R, -1 } };
            break;
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
            // South Pole.
            neighborhood_edge_[BoundaryEdge::E_00R] = {
                { SubdomainInfo( ( diamond_id + 2 ) % 5 + 5, 0, 0, 0 ), BoundaryEdge::E_00R, -1 },
                { SubdomainInfo( ( diamond_id + 3 ) % 5 + 5, 0, 0, 0 ), BoundaryEdge::E_00R, -1 } };
            break;
        default:
            throw std::logic_error( "Invalid diamond id." );
        }

        // Assigning ranks.

        for ( auto& neighbors : neighborhood_vertex_ | std::views::values )
        {
            for ( auto& [neighbor_subdomain_info, neighbor_boundary_vertex, neighbor_rank] : neighbors )
            {
                neighbor_rank = subdomain_to_rank( neighbor_subdomain_info );
            }
        }

        for ( auto& neighbors : neighborhood_edge_ | std::views::values )
        {
            for ( auto& [neighbor_subdomain_info, neighbor_boundary_edge, neighbor_rank] : neighbors )
            {
                neighbor_rank = subdomain_to_rank( neighbor_subdomain_info );
            }
        }

        for ( auto& [neighbor_subdomain_info, neighbor_boundary_face, neighbor_rank] :
              neighborhood_face_ | std::views::values )
        {
            neighbor_rank = subdomain_to_rank( neighbor_subdomain_info );
        }
    }

    std::map< BoundaryVertex, std::vector< NeighborSubdomainTupleVertex > > neighborhood_vertex_;
    std::map< BoundaryEdge, std::vector< NeighborSubdomainTupleEdge > >     neighborhood_edge_;
    std::map< BoundaryFace, NeighborSubdomainTupleFace >                    neighborhood_face_;
};

/// @brief Holds the DomainInfo plus the neighborhood information (SubdomainNeighborhood) for all process-local
///        subdomains.
class DistributedDomain
{
  public:
    DistributedDomain() = default;

    using LocalSubdomainIdx = int;

    /// @brief Creates a Domain with a single subdomain per diamond and initializes all the subdomain neighborhoods.
    static DistributedDomain create_uniform_single_subdomain(
        const int                                                    lateral_refinement_level,
        const int                                                    radial_refinement_level,
        const real_t                                                 r_min,
        const real_t                                                 r_max,
        const std::function< mpi::MPIRank( const SubdomainInfo& ) >& subdomain_to_rank =
            subdomain_to_rank_distribute_full_diamonds )
    {
        DistributedDomain domain;
        domain.domain_info_ = DomainInfo( lateral_refinement_level, r_min, r_max, 1 << radial_refinement_level );
        int idx             = 0;
        for ( const auto& subdomain : domain.domain_info_.local_subdomains( subdomain_to_rank ) )
        {
            domain.subdomains_[subdomain] = {
                idx, SubdomainNeighborhood( domain.domain_info_, subdomain, subdomain_to_rank ) };
            idx++;
        }
        return domain;
    }

    const DomainInfo& domain_info() const { return domain_info_; }
    const std::map< SubdomainInfo, std::tuple< LocalSubdomainIdx, SubdomainNeighborhood > >& subdomains() const
    {
        return subdomains_;
    }

  private:
    DomainInfo                                                                        domain_info_;
    std::map< SubdomainInfo, std::tuple< LocalSubdomainIdx, SubdomainNeighborhood > > subdomains_;
};

template < typename ValueType >
inline Grid4DDataScalar< ValueType >
    allocate_scalar_grid( const std::string label, const DistributedDomain& distributed_domain )
{
    return Grid4DDataScalar< ValueType >(
        label,
        distributed_domain.subdomains().size(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        distributed_domain.domain_info().subdomain_num_nodes_radially() );
}

template < typename ValueType >
inline Grid4DDataScalar< ValueType >
    allocate_scalar_grid( const std::string label, const DistributedDomain& distributed_domain, const int level )
{
    return Grid4DDataScalar< ValueType >(
        label,
        distributed_domain.subdomains().size(),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally( level ),
        distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally( level ),
        distributed_domain.domain_info().subdomain_num_nodes_radially( level ) );
}

inline Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >
    local_domain_md_range_policy_nodes( const DistributedDomain& distributed_domain )
{
    return Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
        { 0, 0, 0, 0 },
        { static_cast< long long >( distributed_domain.subdomains().size() ),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally(),
          distributed_domain.domain_info().subdomain_num_nodes_radially() } );
}

inline Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >
    local_domain_md_range_policy_cells( const DistributedDomain& distributed_domain )
{
    return Kokkos::MDRangePolicy< Kokkos::Rank< 4 > >(
        { 0, 0, 0, 0 },
        { static_cast< long long >( distributed_domain.subdomains().size() ),
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
          distributed_domain.domain_info().subdomain_num_nodes_per_side_laterally() - 1,
          distributed_domain.domain_info().subdomain_num_nodes_radially() - 1 } );
}

/// @brief Returns an initialized grid with the coordinates of all subdomains' nodes projected to the unit sphere.
///
/// The layout is
///
///     grid( local_subdomain_id, x_idx, y_idx, node_coord )
///
/// where node_coord is in {0, 1, 2} and refers to the cartesian coordinate of the point.
template < std::floating_point T >
Grid3DDataVec< T, 3 > subdomain_unit_sphere_single_shell_coords( const DistributedDomain& domain )
{
    Grid3DDataVec< T, 3 > subdomain_coords(
        "subdomain_unit_sphere_coords",
        domain.subdomains().size(),
        domain.domain_info().subdomain_num_nodes_per_side_laterally(),
        domain.domain_info().subdomain_num_nodes_per_side_laterally() );

    typename Grid3DDataVec< T, 3 >::HostMirror subdomain_coords_host = Kokkos::create_mirror_view( subdomain_coords );

    for ( const auto& [subdomain_info, data] : domain.subdomains() )
    {
        const auto& [subdomain_idx, neighborhood] = data;

        unit_sphere_single_shell_subdomain_coords< T >(
            subdomain_coords_host,
            subdomain_idx,
            subdomain_info.diamond_id(),
            domain.domain_info().diamond_lateral_refinement_level(),
            domain.domain_info().num_subdomains_per_diamond_side(),
            subdomain_info.subdomain_x(),
            subdomain_info.subdomain_y() );
    }

    Kokkos::deep_copy( subdomain_coords, subdomain_coords_host );
    return subdomain_coords;
}

/// @brief Returns an initialized grid with the radii of all subdomains' nodes.
///
/// The layout is
///
///     grid( local_subdomain_id, r_idx )
///
template < std::floating_point T >
Grid2DDataScalar< T > subdomain_shell_radii( const DistributedDomain& domain )
{
    const int shells_per_subdomain = domain.domain_info().subdomain_num_nodes_radially();
    const int layers_per_subdomain = shells_per_subdomain - 1;

    Grid2DDataScalar< T > radii_device( "subdomain_shell_radii", domain.subdomains().size(), shells_per_subdomain );
    typename Grid2DDataScalar< T >::HostMirror radii_host = Kokkos::create_mirror_view( radii_device );

    for ( const auto& [subdomain_info, data] : domain.subdomains() )
    {
        const auto& [subdomain_idx, neighborhood] = data;

        const int subdomain_innermost_node_idx = subdomain_info.subdomain_r() * layers_per_subdomain;
        const int subdomain_outermost_node_idx = subdomain_innermost_node_idx + layers_per_subdomain;

        int j = 0;
        for ( int node_idx = subdomain_innermost_node_idx; node_idx <= subdomain_outermost_node_idx; node_idx++ )
        {
            radii_host( subdomain_idx, j ) = domain.domain_info().radii()[node_idx];
            j++;
        }
    }

    Kokkos::deep_copy( radii_device, radii_host );
    return radii_device;
}

template < typename CoordsShellType, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION dense::Vec< typename CoordsShellType::value_type, 3 > coords(
    const int              subdomain,
    const int              x,
    const int              y,
    const int              r,
    const CoordsShellType& coords_shell,
    const CoordsRadiiType& coords_radii )
{
    using T = CoordsShellType::value_type;
    static_assert( std::is_same_v< T, typename CoordsRadiiType::value_type > );

    static_assert(
        std::is_same_v< CoordsShellType, Grid3DDataVec< T, 3 > > ||
        std::is_same_v< CoordsShellType, typename Grid3DDataVec< T, 3 >::HostMirror > );

    static_assert(
        std::is_same_v< CoordsRadiiType, Grid2DDataScalar< T > > ||
        std::is_same_v< CoordsRadiiType, typename Grid2DDataScalar< T >::HostMirror > );

    dense::Vec< T, 3 > coords;
    coords( 0 ) = coords_shell( subdomain, x, y, 0 );
    coords( 1 ) = coords_shell( subdomain, x, y, 1 );
    coords( 2 ) = coords_shell( subdomain, x, y, 2 );
    return coords * coords_radii( subdomain, r );
}

template < typename CoordsShellType, typename CoordsRadiiType >
KOKKOS_INLINE_FUNCTION dense::Vec< typename CoordsShellType::value_type, 3 > coords(
    const dense::Vec< int, 4 > subdomain_x_y_r,
    const CoordsShellType&     coords_shell,
    const CoordsRadiiType&     coords_radii )
{
    using T = CoordsShellType::value_type;
    static_assert( std::is_same_v< T, typename CoordsRadiiType::value_type > );

    static_assert(
        std::is_same_v< CoordsShellType, Grid3DDataVec< T, 3 > > ||
        std::is_same_v< CoordsShellType, typename Grid3DDataVec< T, 3 >::HostMirror > );

    static_assert(
        std::is_same_v< CoordsRadiiType, Grid2DDataScalar< T > > ||
        std::is_same_v< CoordsRadiiType, typename Grid2DDataScalar< T >::HostMirror > );

    return coords(
        subdomain_x_y_r( 0 ),
        subdomain_x_y_r( 1 ),
        subdomain_x_y_r( 2 ),
        subdomain_x_y_r( 3 ),
        coords_shell,
        coords_radii );
}

} // namespace terra::grid::shell
