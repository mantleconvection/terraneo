
#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "cli11_wrapper.hpp"

namespace terra::util {

/// @brief Just a small wrapper to set the default of the option to the value of the variable where it shall be stored.
///
/// Equivalent to
/// @code
/// app.add_option( name, field, std::move( desc ) )->default_val( field );
/// @endcode
/// but with this you do not have to pass 'field' twice :)
CLI::Option* add_option_with_default( CLI::App& app, const std::string& name, auto& field, std::string desc = "" )
{
    return app.add_option( name, field, std::move( desc ) )->default_val( field );
}

/// @brief Just a small wrapper to set the default of the flag to the value of the variable where it shall be stored.
///
/// Equivalent to
/// @code
/// app.add_flag( name, field, std::move( desc ) )->default_val( field );
/// @endcode
/// but with this you do not have to pass 'field' twice :)
inline CLI::Option* add_flag_with_default( CLI::App& app, const std::string& name, bool& field, std::string desc = "" )
{
    return app.add_flag( name, field, std::move( desc ) )->default_val( field );
}

/// @brief Prints an overview of the available flags, the passed arguments, defaults, etc.
inline void print_cli_summary( const CLI::App& app, std::ostream& os = std::cout )
{
    struct Row
    {
        std::string name;
        std::string count;
        std::string value;
        std::string def;
    };

    std::vector< Row > rows;
    rows.reserve( app.get_options().size() );

    for ( const auto* opt : app.get_options() )
    {
        if ( opt->get_name().empty() )
            continue;

        Row r;
        r.name  = opt->get_name();
        r.count = std::to_string( opt->count() );

        // Value(s)
        if ( opt->count() > 0 )
        {
            std::ostringstream vals;
            auto               results = opt->results();
            for ( size_t i = 0; i < results.size(); ++i )
            {
                if ( i > 0 )
                    vals << ' ';
                vals << results[i];
            }
            r.value = vals.str();
        }
        else
        {
            r.value = "(not set)";
        }

        // Default (if any)
        r.def = opt->get_default_str();

        rows.push_back( std::move( r ) );
    }

    // Compute column widths dynamically
    size_t w_name  = std::max< size_t >( 4, std::max_element( rows.begin(), rows.end(), []( auto& a, auto& b ) {
                                               return a.name.size() < b.name.size();
                                           } )->name.size() );
    size_t w_count = std::max< size_t >( 5, std::max_element( rows.begin(), rows.end(), []( auto& a, auto& b ) {
                                                return a.count.size() < b.count.size();
                                            } )->count.size() );
    size_t w_value = std::max< size_t >( 5, std::max_element( rows.begin(), rows.end(), []( auto& a, auto& b ) {
                                                return a.value.size() < b.value.size();
                                            } )->value.size() );
    size_t w_def   = std::max< size_t >( 7, std::max_element( rows.begin(), rows.end(), []( auto& a, auto& b ) {
                                              return a.def.size() < b.def.size();
                                          } )->def.size() );

    // Print header
    os << std::left << std::setw( w_name + 2 ) << "Name" << std::setw( w_count + 2 ) << "Count"
       << std::setw( w_value + 2 ) << "Value"
       << "Default\n";

    os << std::string( w_name + w_count + w_value + w_def + 8, '-' ) << "\n";

    // Print rows
    for ( const auto& r : rows )
    {
        os << std::left << std::setw( w_name + 2 ) << r.name << std::setw( w_count + 2 ) << r.count
           << std::setw( w_value + 2 ) << r.value << r.def << "\n";
    }
}

} // namespace terra::util