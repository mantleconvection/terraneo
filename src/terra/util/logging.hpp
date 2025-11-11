

#pragma once

#include <iostream>
#include <utility>

#include "mpi/mpi.hpp"
#include "timestamp.hpp"

namespace terra::util {

namespace detail {

/// @brief std::ostream subclass to enable logging with prefix and only on root.
///
/// Use global instances of this class like `logroot` as a replacement for std::cout.
class PrefixCout final : std::streambuf, public std::ostream
{
    std::function< std::string() > prefix_;
    bool                           only_root_;
    bool                           at_line_start_ = true;

  public:
    explicit PrefixCout(
        std::function< std::string() > prefix    = []() { return ""; },
        bool                           only_root = true )
    : std::ostream( this )
    , prefix_( std::move( prefix ) )
    , only_root_( only_root )
    {}

    int overflow( int ch ) override
    {
        if ( only_root_ && mpi::rank() != 0 )
        {
            return 0;
        }

        if ( ch == EOF )
        {
            return EOF;
        }

        if ( at_line_start_ )
        {
            std::cout << prefix_();
            at_line_start_ = false;
        }

        std::cout.put( static_cast< char >( ch ) );

        if ( ch == '\n' )
        {
            std::cout.flush();
            at_line_start_ = true;
        }

        return ch;
    }
};

inline std::string log_prefix()
{
    std::stringstream ss;
    ss << "[LOG | rank " << std::setw( static_cast< int >( std::to_string( mpi::num_processes() - 1 ).size() ) )
       << mpi::rank() << " | " << current_timestamp() + "] ";
    return ss.str();
}

} // namespace detail

/// @brief std::ostream subclass that just logs on root and adds a timestamp for each line.
///
/// You should be able to use this as a plugin replacement for std::cout.
inline detail::PrefixCout logroot( []() { return detail::log_prefix(); } );

/// @brief std::ostream subclass that just logs on any process and adds a timestamp for each line.
///
/// You should be able to use this as a plugin replacement for std::cout.
inline detail::PrefixCout logall( []() { return detail::log_prefix(); }, false );

} // namespace terra::util