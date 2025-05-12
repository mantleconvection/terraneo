#pragma once

namespace terra {
using real_t = double;

/// @brief Cast to type real_t using "real_c(x)".
template < typename T >
real_t real_c( T t )
{
    return static_cast< real_t >( t );
}
} // namespace terra
