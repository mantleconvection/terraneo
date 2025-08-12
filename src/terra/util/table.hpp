#pragma once
#include <chrono>
#include <ctime>
#include <functional>
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
    using Value = std::variant<
        std::monostate,
        char,
        short,
        int,
        long,
        long long,
        unsigned char,
        unsigned short,
        unsigned int,
        unsigned long,
        unsigned long long,
        float,
        double,
        bool,
        std::string >;
    using Row = std::unordered_map< std::string, Value >;

    Table( bool print_on_add = false )
    : print_on_add_( print_on_add )
    {
        columns_.insert( "id" );
        columns_.insert( "timestamp" );
    }

    void add_row( const Row& row_data )
    {
        Row row;
        row["id"]        = global_id_counter++;
        row["timestamp"] = current_timestamp();

        for ( const auto& [key, value] : row_data )
        {
            row[key] = value;
            columns_.insert( key );
        }

        if ( print_on_add_ )
        {
            for ( const auto& [key, val] : row )
            {
                std::cout << key << ": " << value_to_string( val ) << ", ";
            }
            std::cout << std::endl;
        }

        rows_.emplace_back( std::move( row ) );
    }

    Table select( const std::vector< std::string >& selected_columns ) const
    {
        Table result( false );
        result.columns_ = { "id", "timestamp" }; // always keep these
        for ( const auto& col : selected_columns )
        {
            result.columns_.insert( col );
        }

        for ( const auto& row : rows_ )
        {
            Row new_row;
            for ( const auto& col : result.columns_ )
            {
                new_row[col] = get_or_none( row, col );
            }
            result.rows_.push_back( std::move( new_row ) );
        }

        return result;
    }

    Table query_not_none( const std::string& column ) const
    {
        Table result( false );
        result.columns_ = columns_;
        for ( const auto& row : rows_ )
        {
            auto it = row.find( column );
            if ( it != row.end() && !std::holds_alternative< std::monostate >( it->second ) )
            {
                result.rows_.push_back( row );
            }
        }
        return result;
    }

    Table query_equals( const std::string& column, const Value& value ) const
    {
        Table result( false );
        result.columns_ = columns_;
        for ( const auto& row : rows_ )
        {
            auto it = row.find( column );
            if ( it != row.end() && it->second == value )
            {
                result.rows_.push_back( row );
            }
        }
        return result;
    }

    Table query_where( const std::string& column, std::function< bool( const Value& ) > predicate ) const
    {
        Table result( false );
        result.columns_ = columns_;
        for ( const auto& row : rows_ )
        {
            auto it = row.find( column );
            if ( it != row.end() && predicate( it->second ) )
            {
                result.rows_.push_back( row );
            }
        }
        return result;
    }

    const Table& print_pretty( std::ostream& os = std::cout ) const
    {
        std::unordered_map< std::string, size_t > widths;
        for ( const auto& col : columns_ )
        {
            widths[col] = col.size();
        }

        for ( const auto& row : rows_ )
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
            os << " " << std::setw( widths[col] ) << std::right << col << " |";
        }
        os << "\n";
        sep();

        for ( const auto& row : rows_ )
        {
            os << "|";
            for ( const auto& col : columns_ )
            {
                os << " " << std::setw( widths[col] ) << std::right << value_to_string( get_or_none( row, col ) )
                   << " |";
            }
            os << "\n";
        }
        sep();
        return *this;
    }

    const Table& print_csv( std::ostream& os = std::cout ) const
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
        return *this;
    }

    void set_print_on_add( bool enabled ) { print_on_add_ = enabled; }

    void clear()
    {
        rows_.clear();
        columns_.clear();
        columns_.insert( "id" );
        columns_.insert( "timestamp" );
    }

    size_t row_count() const { return rows_.size(); }
    size_t column_count() const { return columns_.size(); }

    // Utility accessors
    static std::string value_to_string( const Value& v )
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
                    ss << std::scientific << std::setprecision( 3 ) << val;
                    return ss.str();
                }
                else if constexpr ( std::is_same_v< T, std::string > )
                {
                    return val;
                }
                else
                {
                    return std::to_string( val );
                }
            },
            v );
    }

    static Value get_or_none( const Row& row, const std::string& col )
    {
        auto it = row.find( col );
        return ( it != row.end() ) ? it->second : std::monostate{};
    }

  private:
    std::vector< Row >      rows_;
    std::set< std::string > columns_;
    bool                    print_on_add_;
    inline static int       global_id_counter = 1;

    static std::string current_timestamp()
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
};

;
} // namespace terra::util
