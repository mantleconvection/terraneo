#pragma once

#include <array>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <stdexcept>

class Point3D
{
  private:
    // Internal storage using std::array for easy indexed access
    std::array< double, 3 > coords_;

  public:
    // --- Constructors ---

    // Default constructor (initializes to 0, 0, 0)
    Point3D()
    : coords_{ 0.0, 0.0, 0.0 }
    {}

    // Constructor with initial values
    Point3D( double x_val, double y_val, double z_val )
    : coords_{ x_val, y_val, z_val }
    {}

    // Copy constructor (usually default is fine, but explicit can be clearer)
    Point3D( const Point3D& other ) = default;

    // Move constructor (usually default is fine)
    Point3D( Point3D&& other ) noexcept = default;

    // --- Assignment Operators ---

    // Copy assignment
    Point3D& operator=( const Point3D& other ) = default;

    // Move assignment
    Point3D& operator=( Point3D&& other ) noexcept = default;

    // --- Accessors ---

    /**
     * @brief Access components by index (0=x, 1=y, 2=z). Non-const version.
     * @param index The component index (0, 1, or 2).
     * @return Reference to the coordinate value.
     * @throws std::out_of_range if index is invalid.
     */
    double& operator()( int index )
    {
        if ( index < 0 || index > 2 )
        {
            throw std::out_of_range( "Point3D index out of range (must be 0, 1, or 2)" );
        }
        // Correctly return reference from the array element
        return coords_[static_cast< std::size_t >( index )];
    }

    /**
     * @brief Access components by index (0=x, 1=y, 2=z). Const version.
     * @param index The component index (0, 1, or 2).
     * @return Const reference to the coordinate value.
     * @throws std::out_of_range if index is invalid.
     */
    const double& operator()( int index ) const
    {
        if ( index < 0 || index > 2 )
        {
            throw std::out_of_range( "Point3D index out of range (must be 0, 1, or 2)" );
        }
        // Correctly return const reference from the array element
        return coords_[static_cast< std::size_t >( index )];
    }

    /** @brief Access the x-coordinate. Non-const version. */
    double& x() { return coords_[0]; }
    /** @brief Access the x-coordinate. Const version. */
    const double& x() const { return coords_[0]; }

    /** @brief Access the y-coordinate. Non-const version. */
    double& y() { return coords_[1]; }
    /** @brief Access the y-coordinate. Const version. */
    const double& y() const { return coords_[1]; }

    /** @brief Access the z-coordinate. Non-const version. */
    double& z() { return coords_[2]; }
    /** @brief Access the z-coordinate. Const version. */
    const double& z() const { return coords_[2]; }

    // --- Optional Bonus Methods ---

    /** @brief Calculate the magnitude (length) of the vector from the origin. */
    double magnitude() const
    {
        return std::sqrt( coords_[0] * coords_[0] + coords_[1] * coords_[1] + coords_[2] * coords_[2] );
    }

    /** @brief Normalize the point (treat as vector) to unit length in-place. */
    void normalize()
    {
        double mag = magnitude();
        if ( mag > 1e-15 )
        { // Avoid division by zero
            coords_[0] /= mag;
            coords_[1] /= mag;
            coords_[2] /= mag;
        }
    }

    /** @brief Return a normalized copy of the point (treat as vector). */
    Point3D normalized() const
    {
        Point3D res = *this; // Make a copy
        res.normalize();
        return res;
    }

    // Allow printing the point easily (optional)
    friend std::ostream& operator<<( std::ostream& os, const Point3D& p )
    {
        os << "(" << std::fixed << std::setprecision( 6 ) // Example formatting
           << p.x() << ", " << p.y() << ", " << p.z() << ")";
        return os;
    }
};

// --- Optional: Define basic arithmetic operators outside the class ---

inline Point3D operator+( const Point3D& a, const Point3D& b )
{
    return Point3D( a.x() + b.x(), a.y() + b.y(), a.z() + b.z() );
}

inline Point3D operator-( const Point3D& a, const Point3D& b )
{
    return Point3D( a.x() - b.x(), a.y() - b.y(), a.z() - b.z() );
}

// Scalar multiplication
inline Point3D operator*( const Point3D& p, double scalar )
{
    return Point3D( p.x() * scalar, p.y() * scalar, p.z() * scalar );
}

inline Point3D operator*( double scalar, const Point3D& p )
{
    return p * scalar; // Reuse previous operator
}

// Scalar division
inline Point3D operator/( const Point3D& p, double scalar )
{
    if ( std::abs( scalar ) < 1e-15 )
    {
        throw std::runtime_error( "Division by zero in Point3D scalar division." );
    }
    return Point3D( p.x() / scalar, p.y() / scalar, p.z() / scalar );
}
