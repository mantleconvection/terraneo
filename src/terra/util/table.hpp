#pragma once
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace terra::util {

class Table
{
  public:
    using Value = std::variant< std::monostate, int, long, double, bool, std::string >;
    using Row   = std::unordered_map< std::string, Value >;

    Table( bool print_on_add = false )
    : print_on_add_( print_on_add )
    , next_id_( 1 )
    {
        columns_.insert( "id" );
        columns_.insert( "timestamp" );
    }

    void add_row( const Row& row_data )
    {
        Row row;

        // Auto id and timestamp
        row["id"]        = next_id_++;
        row["timestamp"] = current_timestamp();

        for ( const auto& [key, value] : row_data )
        {
            row[key] = value;
            columns_.insert( key );
        }

        rows_.emplace_back( std::move( row ) );

        if ( print_on_add_ )
        {
            // print_last_row();
            print_pretty( { rows_.back() }, std::cout );
        }
    }

    void print_csv( std::ostream& os = std::cout ) const
    {
        print_header( os, "," );
        for ( const auto& row : rows_ )
        {
            bool first = true;
            for ( const auto& col : columns_ )
            {
                if ( !first )
                    os << ",";
                os << value_to_string( get_or_none( row, col ) );
                first = false;
            }
            os << "\n";
        }
    }

    void print_pretty( std::ostream& os = std::cout ) const { print_pretty( rows_, os ); }

    void print_pretty( const std::vector< Row >& rows, std::ostream& os = std::cout ) const
    {
        std::unordered_map< std::string, size_t > widths;
        for ( const auto& col : columns_ )
        {
            widths[col] = col.size();
        }

        for ( const auto& row : rows )
        {
            for ( const auto& col : columns_ )
            {
                widths[col] = std::max( widths[col], value_to_string( get_or_none( row, col ) ).size() );
            }
        }

        auto sep = [&] {
            for ( const auto& col : columns_ )
            {
                os << "+" << std::string( widths[col] + 2, '-' );
            }
            os << "+\n";
        };

        sep();
        os << "|";
        for ( const auto& col : columns_ )
        {
            os << " " << std::setw( widths[col] ) << std::left << col << " |";
        }
        os << "\n";
        sep();

        for ( const auto& row : rows )
        {
            os << "|";
            for ( const auto& col : columns_ )
            {
                os << " " << std::setw( widths[col] ) << std::left << value_to_string( get_or_none( row, col ) )
                   << " |";
            }
            os << "\n";
        }
        sep();
    }

    std::vector< Row > query_not_none( const std::string& column ) const
    {
        std::vector< Row > result;
        for ( const auto& row : rows_ )
        {
            auto it = row.find( column );
            if ( it != row.end() && !std::holds_alternative< std::monostate >( it->second ) )
            {
                result.push_back( row );
            }
        }
        return result;
    }

    void set_print_on_add( bool enabled ) { print_on_add_ = enabled; }

    void clear()
    {
        rows_.clear();
        columns_.clear();
        columns_.insert( "id" );
        columns_.insert( "timestamp" );
        next_id_ = 1;
    }

    size_t row_count() const { return rows_.size(); }
    size_t column_count() const { return columns_.size(); }

  private:
    std::vector< Row >      rows_;
    std::set< std::string > columns_;
    bool                    print_on_add_;
    int                     next_id_;

    std::string current_timestamp() const
    {
        using namespace std::chrono;
        auto        now = system_clock::now();
        std::time_t t   = system_clock::to_time_t( now );
        std::tm     buf;
#ifdef _WIN32
        localtime_s( &buf, &t );
#else
        localtime_r( &t, &buf );
#endif
        char str[32];
        std::strftime( str, sizeof( str ), "%Y-%m-%d %H:%M:%S", &buf );
        return std::string( str );
    }

    Value get_or_none( const Row& row, const std::string& col ) const
    {
        auto it = row.find( col );
        return ( it != row.end() ) ? it->second : std::monostate{};
    }

    std::string value_to_string( const Value& v ) const
    {
        return std::visit(
            []( const auto& val ) -> std::string {
                using T = std::decay_t< decltype( val ) >;
                if constexpr ( std::is_same_v< T, std::monostate > )
                {
                    return "None";
                }
                else if constexpr ( std::is_same_v< T, std::string > )
                {
                    return val;
                }
                else if constexpr ( std::is_same_v< T, bool > )
                {
                    return val ? "true" : "false";
                }
                else if constexpr ( std::is_same_v< T, double > )
                {
                    std::ostringstream ss;
                    ss << std::fixed << std::setprecision( 3 ) << val;
                    return ss.str();
                }
                else
                {
                    return std::to_string( val );
                }
            },
            v );
    }

    void print_header( std::ostream& os, const std::string& sep ) const
    {
        bool first = true;
        for ( const auto& col : columns_ )
        {
            if ( !first )
                os << sep;
            os << col;
            first = false;
        }
        os << "\n";
    }

    void print_last_row() const
    {
        if ( !rows_.empty() )
        {
            const auto& row   = rows_.back();
            bool        first = true;
            for ( const auto& col : columns_ )
            {
                if ( !first )
                    std::cout << " ";
                std::cout << value_to_string( get_or_none( row, col ) );
                first = false;
            }
            std::cout << "\n";
        }
    }
};
} // namespace terra::util
