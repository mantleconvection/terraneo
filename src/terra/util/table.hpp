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

/// @brief Table class for storing and manipulating tabular data.
///
/// Provides functionality to add rows, select columns, query data, and print in various formats (pretty, JSON lines, CSV).
///
/// Not optimized for performance but designed for ease of use and flexibility.
/// If you need high performance, or millions of rows, consider using a database or specialized library.
/// But still useful for small to medium datasets, logging, prototyping, and data analysis tasks.
///
/// Rows are stored as maps from column names to values.
/// Columns are dynamically added as needed.
///
/// Supports various value types, including strings, numbers, booleans, and None (null).
///
/// @note Each row automatically gets an "id" and "timestamp" column.
///
/// @note For logging, the convention is that most functions use the key "tag" to add some keyword to the table. To
///       later sort data, therefore add a "tag" to mark where the data comes from.
///
/// Example usage:
/// @code
///
/// terra::util::Table table;
///
/// // Add rows with various data types.
/// table.add_row({{"name", "Charlie"}, {"age", 28}, {"active", true}});
/// table.add_row({{"name", "Alice"}, {"age", 30}, {"active", false}});
/// table.add_row({{"name", "Bob"}, {"age", 25}, {"active", true}});
///
/// // Add rows with different columns to the same table.
/// table.add_row({{"city", "Berlin"}, {"country", "Germany"}, {"population", 3769000}});
///
/// // Select specific columns
/// auto selected_table = table.select_columns({ "name", "age" });
///
/// // Query rows where age is greater than 26
/// auto queried_table = table.query_rows_where( "age", []( const auto& value ) {
///     return std::holds_alternative<int>( value ) && std::get<int>( value ) > 26;
/// });
///
/// // Print the table in different formats
/// table.print_pretty();
/// selected_table.print_jsonl();
/// queried_table.print_csv();
///
/// // Print to file
///
/// // CSV
/// std::ofstream file("output.csv");
/// table.print_csv(file);
/// file.close();
///
/// // JSON lines
/// std::ofstream json_file("output.jsonl");
/// table.print_jsonl(json_file);
/// json_file.close();
///
/// // Clear the table
/// table.clear();
/// @endcode
///
class Table
{
  public:
    /// @brief Max length of string values (required for safe reading of possibly non-null-terminated char arrays).
    static constexpr int MAX_STRING_LENGTH = 10000;

  private:
    using ValueBase = std::variant<
        std::monostate,
        std::string,
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
        bool >;

  public:
    /// @brief Type for table cell values.
    struct Value : ValueBase
    {
        // Using a variant directly is annoying because we need to handle string literals via const char *.
        // (That conversion btw seems to be compiler-dependent :) which makes it even more annoying.)
        // It is possible, but requires another special case throughout the accessors.
        // So we simply always convert to std::string.

        using ValueBase::ValueBase; // inherit constructors

        Value( const char* arg )
        : ValueBase( char_ptr_to_string_safe( arg ) )
        {}
    };

    /// @brief Type for a table row (mapping column name to value).
    using Row = std::unordered_map< std::string, Value >;

    /// @brief Construct an empty table.
    Table() = default;

    /// @brief Get all rows in the table.
    /// @return Vector of rows.
    [[nodiscard]] const std::vector< Row >& rows() const { return rows_; }

    /// @brief Get all column names in the table.
    /// @return Set of column names.
    [[nodiscard]] const std::set< std::string >& columns() const { return columns_; }

    /// @brief Add a row to the table.
    /// Adds "id" and "timestamp" columns automatically.
    /// @param row_data Row data as a map from column name to value.
    void add_row( const Row& row_data )
    {
        Row row;
        row["id"]        = global_id_counter++;
        row["timestamp"] = current_timestamp();
        columns_.insert( "id" );
        columns_.insert( "timestamp" );

        for ( const auto& [key, value] : row_data )
        {
            row[key] = value;
            columns_.insert( key );
        }

        rows_.emplace_back( row );
    }

    /// @brief Select a subset of columns from the table.
    /// @param selected_columns Columns to select.
    /// @return New Table with only selected columns.
    [[nodiscard]] Table select_columns( const std::vector< std::string >& selected_columns ) const
    {
        Table result;

        for ( const auto& col : selected_columns )
        {
            result.columns_.insert( col );
        }

        for ( const auto& row : rows_ )
        {
            Row new_row;
            for ( const auto& col : result.columns_ )
            {
                new_row[col] = get_value_from_row_or_none( row, col );
            }
            result.rows_.push_back( std::move( new_row ) );
        }

        return result;
    }

    /// @brief Query rows where a column is not None.
    /// @param column Column name.
    /// @return New Table with matching rows.
    [[nodiscard]] Table query_rows_not_none( const std::string& column ) const
    {
        Table result;
        result.columns_ = columns_;
        for ( const auto& row : rows_ )
        {
            if ( auto it = row.find( column );
                 it != row.end() && !std::holds_alternative< std::monostate >( it->second ) )
            {
                result.rows_.push_back( row );
            }
        }
        return result;
    }

    /// @brief Query rows where a column equals a value.
    /// @param column Column name.
    /// @param value Value to compare.
    /// @return New Table with matching rows.
    [[nodiscard]] Table query_rows_equals( const std::string& column, const Value& value ) const
    {
        Table result;
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

    /// @brief Query rows where a column satisfies a predicate.
    /// @param column Column name.
    /// @param predicate Predicate function.
    /// @return New Table with matching rows.
    Table query_rows_where( const std::string& column, const std::function< bool( const Value& ) >& predicate ) const
    {
        Table result;
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

    /// @brief Print the table in a pretty formatted style.
    /// @param os Output stream (default std::cout).
    /// @return Reference to this table.
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
                widths[col] = std::max( widths[col], value_to_string( get_value_from_row_or_none( row, col ) ).size() );
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
            os << " " << std::setw( static_cast< int >( widths[col] ) ) << std::right << col << " |";
        }
        os << "\n";
        sep();

        for ( const auto& row : rows_ )
        {
            os << "|";
            for ( const auto& col : columns_ )
            {
                os << " " << std::setw( static_cast< int >( widths[col] ) ) << std::right
                   << value_to_string( get_value_from_row_or_none( row, col ) ) << " |";
            }
            os << "\n";
        }
        sep();
        return *this;
    }

    /// @brief Print the table as JSON lines.
    /// Each row is a JSON object, one per line.
    ///
    /// Example output:
    /// {"id":1,"timestamp":"2024-06-10 12:34:56","name":"Alice","age":30}
    /// {"id":2,"timestamp":"2024-06-10 12:34:57","name":"Bob","age":25}
    ///
    /// To parse with pandas:
    /// import pandas as pd
    /// df = pd.read_json("yourfile.jsonl", lines=True)
    ///
    /// @param os Output stream (default std::cout).
    /// @return Reference to this table.
    const Table& print_jsonl( std::ostream& os = std::cout ) const
    {
        for ( const auto& row : rows_ )
        {
            os << "{";
            bool first = true;
            for ( const auto& [key, val] : row )
            {
                if ( !first )
                {
                    os << ",";
                }
                os << "\"" << key << "\":";
                if ( std::holds_alternative< std::string >( val ) )
                {
                    os << "\"" << value_to_string( val ) << "\"";
                }
                else if ( std::holds_alternative< std::monostate >( val ) )
                {
                    os << "null";
                }
                else if ( std::holds_alternative< bool >( val ) )
                {
                    os << ( std::get< bool >( val ) ? "true" : "false" );
                }
                else
                {
                    os << value_to_string( val );
                }
                first = false;
            }

            os << "}\n";
        }
        return *this;
    }

    /// @brief Print the table as CSV.
    /// @param os Output stream (default std::cout).
    /// @return Reference to this table.
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
                os << value_to_string( get_value_from_row_or_none( row, col ) );
                first = false;
            }
            os << "\n";
        }
        return *this;
    }

    /// @brief Clear all rows and columns from the table.
    ///
    /// This resets the table to an empty state - but does not reset the running id counter.
    void clear()
    {
        rows_.clear();
        columns_.clear();
    }

    /// @brief Convert a Value to a string for printing.
    /// @param v Value to convert.
    /// @return String representation.
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
                else if constexpr ( std::is_same_v< T, float > || std::is_same_v< T, double > )
                {
                    std::ostringstream ss;
                    ss << std::scientific << std::setprecision( 3 ) << val;
                    return ss.str();
                }
                else
                {
                    return std::to_string( val );
                }
            },
            v );
    }

    /// @brief Get a value from a row or return None if missing.
    /// @param row Row to search.
    /// @param col Column name.
    /// @return Value or None.
    static Value get_value_from_row_or_none( const Row& row, const std::string& col )
    {
        auto it = row.find( col );
        return ( it != row.end() ) ? it->second : std::monostate{};
    }

  private:
    std::vector< Row >      rows_;    ///< Rows of the table.
    std::set< std::string > columns_; ///< Set of column names.

    inline static int global_id_counter = 0; ///< Global row id counter.

    /// @brief Get the current timestamp as a string.
    /// @return Timestamp string.
    static std::string current_timestamp()
    {
        using namespace std::chrono;
        const auto  now = system_clock::now();
        std::time_t t   = system_clock::to_time_t( now );
        std::tm     buf{};
#ifdef _WIN32
        localtime_s( &buf, &t );
#else
        localtime_r( &t, &buf );
#endif
        char str[32];
        std::strftime( str, sizeof( str ), "%Y-%m-%d %H:%M:%S", &buf );
        return { str };
    }

    /// @brief Print the table header.
    /// @param os Output stream.
    /// @param sep Separator string.
    void print_header( std::ostream& os, const std::string& sep ) const
    {
        bool first = true;
        for ( const auto& col : columns_ )
        {
            if ( !first )
            {
                os << sep;
            }

            os << col;
            first = false;
        }
        os << "\n";
    }

    /// @brief Safely converts const char * to string, even if not null terminated (constant max string length of MAX_STRING_LENGTH).
    static std::string char_ptr_to_string_safe( const char* val )
    {
        if ( !val )
        {
            return std::string{};
        }

        std::size_t len = strnlen( val, MAX_STRING_LENGTH ); // find '\0' but stop at MAX_LEN
        return std::string( val, len );
    }
};

;
} // namespace terra::util
