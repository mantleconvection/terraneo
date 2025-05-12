
#include "spherical_shell_mesh.hpp"

#include "point_3d.hpp"

namespace terra {

// Struct to hold the coordinates of the four base corners
// and the number of intervals N = ntan - 1.
struct BaseCorners
{
    Point3D p00; // Coordinates for global index (0, 0)
    Point3D p0N; // Coordinates for global index (0, N)
    Point3D pN0; // Coordinates for global index (N, 0)
    Point3D pNN; // Coordinates for global index (N, N)
    int     N;   // Number of intervals = ntan - 1. Must be power of 2.

    // Constructor for convenience (optional)
    BaseCorners( Point3D p00_ = {}, Point3D p0N_ = {}, Point3D pN0_ = {}, Point3D pNN_ = {}, int N_ = 0 )
    : p00( p00_ )
    , p0N( p0N_ )
    , pN0( pN0_ )
    , pNN( pNN_ )
    , N( N_ )
    {}
};

// Memoization cache type: maps (i, j) index pair to computed coordinates
using MemoizationCache = std::map< std::pair< int, int >, Point3D >;

/**
 * @brief Computes the coordinates for a specific node (i, j) in the final refined grid.
 *        Uses recursion and memoization, sourcing base points from the BaseCorners struct.
 *
 * @param i Row index (0 to corners.N).
 * @param j Column index (0 to corners.N).
 * @param corners Struct containing base corner coordinates and N = ntan - 1.
 * @param cache Cache to store/retrieve already computed nodes.
 * @return Point3D Coordinates of the node (i, j) on the unit sphere.
 */
static Point3D compute_node_recursive( int i, int j, const BaseCorners& corners, MemoizationCache& cache )
{
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

    Point3D p1, p2; // Parent points

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
    Point3D mid = p1 + p2;

    // Normalize to project onto the unit sphere
    Point3D result = mid.normalized();

    // --- 4. Store result in cache and return ---
    cache[cache_key] = result;
    return result;
}

/**
 * @brief Generates coordinates for a rectangular subdomain of the refined spherical grid.
 *
 * @param corners Struct containing the base corner points and N = ntan - 1.
 * @param i_start_incl Starting row index (inclusive) of the subdomain (global index).
 * @param i_end_incl Ending row index (inclusive) of the subdomain (global index).
 * @param j_start_incl Starting column index (inclusive) of the subdomain (global index).
 * @param j_end_incl Ending column index (inclusive) of the subdomain (global index).
 * @return Kokkos::View<double**[3], Kokkos::HostSpace> Host view containing coordinates
 *         for the subdomain. Dimensions are ((i_end_incl - 1) - i_start, (j_end_incl - 1) - j_start).
 */
SphericalShellSubMeshViewType
    compute_subdomain( const BaseCorners& corners, int i_start_incl, int i_end_incl, int j_start_incl, int j_end_incl )
{
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

    SphericalShellSubMeshViewType subdomain_coords( "subdomain_coords", subdomain_rows, subdomain_cols );

    MemoizationCache cache; // Each subdomain computation gets its own cache

    // --- Compute nodes within the subdomain ---
    for ( int i = i_start; i < i_end; ++i )
    {
        for ( int j = j_start; j < j_end; ++j )
        {
            // Compute the node coordinates using the recursive function
            Point3D coords = compute_node_recursive( i, j, corners, cache ); // Pass corners struct

            // Store in the subdomain view (adjusting indices)
            subdomain_coords( i - i_start, j - j_start, 0 ) = coords.x();
            subdomain_coords( i - i_start, j - j_start, 1 ) = coords.y();
            subdomain_coords( i - i_start, j - j_start, 2 ) = coords.z();
        }
    }

    // std::cout << "Computed " << cache.size() << " unique nodes for subdomain." << std::endl;
    return subdomain_coords;
}

static SphericalShellSubMeshViewType setup_coords_subdomain(
    int diamond_id,
    int ntan,
    int i_start_incl,
    int i_end_incl,
    int j_start_incl,
    int j_end_incl )
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

    BaseCorners corners;
    corners.N = ntan - 1;

    // Insert coordinates of four nodes of this icosahedral diamond for each dim.
    for ( int i = 0; i < 3; ++i )
    {
        corners.p00( i ) = i_node[d_node[diamond_id][0]][i];
        corners.pN0( i ) = i_node[d_node[diamond_id][L]][i];
        corners.pNN( i ) = i_node[d_node[diamond_id][2]][i];
        corners.p0N( i ) = i_node[d_node[diamond_id][R]][i];
    }

    return compute_subdomain( corners, i_start_incl, i_end_incl, j_start_incl, j_end_incl );
}

/// @brief Computes the vertex coordinates on a single diamond of the shell with radius 1.
inline SphericalShellSubMeshViewType setup_coords_classic( int ntan, int diamond_id )
{
    // -------------------
    //  Consistency Check
    // -------------------
    int power = ntan - 1;
    int lvl   = 0;
    while ( power > 0 )
    {
        ++lvl;
        power /= 2;
    }
    --lvl;
    if ( 1 << lvl != ntan - 1 )
    {
        throw std::invalid_argument(
            "ERROR: For SHELLMESH_CLASSIC (ntan-1) must be a power of 2, but ntan is " + std::to_string( ntan ) );
    }

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

    // --------------------------
    //  Allocate 4D memory array
    // --------------------------

    SphericalShellSubMeshViewType             node_coords( "node_coords", ntan, ntan );
    SphericalShellSubMeshViewType::HostMirror node_coords_host = Kokkos::create_mirror_view( node_coords );

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

    // Insert coordinates of four nodes of this icosahedral diamond for each dim.
    for ( int i = 0; i < 3; ++i )
    {
        node_coords_host( 0, 0, i )               = i_node[d_node[diamond_id][0]][i];
        node_coords_host( ntan - 1, 0, i )        = i_node[d_node[diamond_id][L]][i];
        node_coords_host( ntan - 1, ntan - 1, i ) = i_node[d_node[diamond_id][2]][i];
        node_coords_host( 0, ntan - 1, i )        = i_node[d_node[diamond_id][R]][i];
    }

    // Loop over refinement levels
    for ( int m = 1; m < ntan - 1; m *= 2 )
    {
        const int l  = ( ntan - 1 ) / m;
        const int l2 = l / 2;

        // rows of diamond
        for ( int j1 = 0; j1 <= m; ++j1 )
        {
            for ( int j2 = 0; j2 < m; ++j2 )
            {
                // compute coordinates of new node
                const int i1 = j1 * l;
                const int i2 = j2 * l + l2;

                // compute new node as midpoint of arc
                const auto x = node_coords_host( i1, i2 - l2, 0 ) + node_coords_host( i1, i2 + l2, 0 );
                const auto y = node_coords_host( i1, i2 - l2, 1 ) + node_coords_host( i1, i2 + l2, 1 );
                const auto z = node_coords_host( i1, i2 - l2, 2 ) + node_coords_host( i1, i2 + l2, 2 );

                const auto scl = std::sqrt( x * x + y * y + z * z );

                node_coords_host( i1, i2, 0 ) = x / scl;
                node_coords_host( i1, i2, 1 ) = y / scl;
                node_coords_host( i1, i2, 2 ) = z / scl;
            }
        }

        // columns of diamond
        for ( int j1 = 0; j1 <= m; ++j1 )
        {
            for ( int j2 = 0; j2 < m; ++j2 )
            {
                // compute coordinates of new node
                const int i1 = j2 * l + l2;
                const int i2 = j1 * l;

                // compute new node as midpoint of arc
                const auto x = node_coords_host( i1 - l2, i2, 0 ) + node_coords_host( i1 + l2, i2, 0 );
                const auto y = node_coords_host( i1 - l2, i2, 1 ) + node_coords_host( i1 + l2, i2, 1 );
                const auto z = node_coords_host( i1 - l2, i2, 2 ) + node_coords_host( i1 + l2, i2, 2 );

                const auto scl = std::sqrt( x * x + y * y + z * z );

                node_coords_host( i1, i2, 0 ) = x / scl;
                node_coords_host( i1, i2, 1 ) = y / scl;
                node_coords_host( i1, i2, 2 ) = z / scl;
            }
        }

        // diagonals of diamond
        for ( int j1 = 0; j1 < m; ++j1 )
        {
            for ( int j2 = 0; j2 < m; ++j2 )
            {
                // compute coordinates of new node
                const int i1 = j1 * l + l2;
                const int i2 = j2 * l + l2;

                // compute new node as midpoint of arc
                const auto x = node_coords_host( i1 - l2, i2 + l2, 0 ) + node_coords_host( i1 + l2, i2 - l2, 0 );
                const auto y = node_coords_host( i1 - l2, i2 + l2, 1 ) + node_coords_host( i1 + l2, i2 - l2, 1 );
                const auto z = node_coords_host( i1 - l2, i2 + l2, 2 ) + node_coords_host( i1 + l2, i2 - l2, 2 );

                const auto scl = std::sqrt( x * x + y * y + z * z );

                node_coords_host( i1, i2, 0 ) = x / scl;
                node_coords_host( i1, i2, 1 ) = y / scl;
                node_coords_host( i1, i2, 2 ) = z / scl;
            }
        }
    }

    Kokkos::deep_copy<>( node_coords, node_coords_host );
    return node_coords;
}

SphericalShellSubMeshViewType setup_coords_subdomain(
    int diamond_id,
    int global_refinements,
    int num_subdomains_per_side,
    int subdomain_i,
    int subdomain_j )
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

    return setup_coords_subdomain( diamond_id, ntan, start_i, end_i, start_j, end_j );
}

} // namespace terra