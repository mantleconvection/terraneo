#include <sstream>
#include <string>

#include "util/init.hpp"
#include "util/table.hpp"

using terra::util::Table;

int main( int argc, char** argv )
{
    terra::util::TerraScopeGuard scope_guard( &argc, &argv );

    Table table;

    int failures = 0;

    // Add rows with various data types
    table.add_row( { { "name", "Charlie" }, { "age", 28 }, { "active", true } } );
    table.add_row( { { "name", "Alice" }, { "age", 30 }, { "active", false } } );
    table.add_row( { { "name", "Bob" }, { "age", 25 }, { "active", true } } );
    table.add_row( { { "city", "Berlin" }, { "country", "Germany" }, { "population", 3769000 } } );

    // Check row count
    if ( table.rows().size() != 4 )
    {
        std::cerr << "FAIL: Expected 4 rows, got " << table.rows().size() << std::endl;
        failures++;
    }

    // Check columns
    if ( table.columns().count( "name" ) == 0 || table.columns().count( "age" ) == 0 ||
         table.columns().count( "active" ) == 0 || table.columns().count( "city" ) == 0 ||
         table.columns().count( "country" ) == 0 || table.columns().count( "population" ) == 0 )
    {
        std::cerr << "FAIL: Missing expected columns in table." << std::endl;
        failures++;
    }

    // Select specific columns
    auto selected_table = table.select_columns( { "name", "age" } );
    if ( selected_table.columns().count( "name" ) == 0 || selected_table.columns().count( "age" ) == 0 )
    {
        std::cerr << "FAIL: select_columns did not include expected columns." << std::endl;
        failures++;
    }
    if ( selected_table.columns().count( "city" ) != 0 )
    {
        std::cerr << "FAIL: select_columns included unexpected column 'city'." << std::endl;
        failures++;
    }

    // Query rows where age is not None
    auto not_none_table = table.query_rows_not_none( "age" );
    if ( not_none_table.rows().size() != 3 )
    {
        std::cerr << "FAIL: query_rows_not_none('age') expected 3 rows, got " << not_none_table.rows().size()
                  << std::endl;
        failures++;
    }

    // Query rows where age equals 30
    auto equals_table = table.query_rows_equals( "age", 30 );
    if ( equals_table.rows().size() != 1 )
    {
        std::cerr << "FAIL: query_rows_equals('age', 30) expected 1 row, got " << equals_table.rows().size()
                  << std::endl;
        failures++;
    }

    // Query rows where age > 26
    auto queried_table = table.query_rows_where( "age", []( const Table::Value& value ) {
        return std::holds_alternative< int >( value ) && std::get< int >( value ) > 26;
    } );
    if ( queried_table.rows().size() != 2 )
    {
        std::cerr << "FAIL: query_rows_where('age', >26) expected 2 rows, got " << queried_table.rows().size()
                  << std::endl;
        failures++;
    }

    // Print formats (just call, don't check output)
    std::cout << "\nPretty print:\n";
    table.print_pretty();
    std::cout << "\nJSON lines:\n";
    table.print_jsonl();
    std::cout << "\nCSV:\n";
    table.print_csv();

    // Check JSONL output
    std::stringstream jsonl_ss;
    table.print_jsonl( jsonl_ss );
    std::string jsonl = jsonl_ss.str();
    if ( jsonl.find( "\"name\":\"Charlie\"" ) == std::string::npos ||
         jsonl.find( "\"name\":\"Alice\"" ) == std::string::npos ||
         jsonl.find( "\"name\":\"Bob\"" ) == std::string::npos ||
         jsonl.find( "\"city\":\"Berlin\"" ) == std::string::npos )
    {
        std::cerr << "FAIL: JSONL output missing expected values:\n" << jsonl << std::endl;
        failures++;
    }
    if ( jsonl.find( "\"population\":3769000" ) == std::string::npos )
    {
        std::cerr << "FAIL: JSONL output missing expected population value:\n" << jsonl << std::endl;
        failures++;
    }

    // Check CSV output
    std::stringstream csv_ss;
    table.print_csv( csv_ss );
    std::string csv = csv_ss.str();
    if ( csv.find( "name" ) == std::string::npos || csv.find( "age" ) == std::string::npos ||
         csv.find( "active" ) == std::string::npos || csv.find( "city" ) == std::string::npos ||
         csv.find( "country" ) == std::string::npos || csv.find( "population" ) == std::string::npos )
    {
        std::cerr << "FAIL: CSV header missing expected columns:\n" << csv << std::endl;
        failures++;
    }
    if ( csv.find( "Charlie" ) == std::string::npos || csv.find( "Alice" ) == std::string::npos ||
         csv.find( "Bob" ) == std::string::npos || csv.find( "Berlin" ) == std::string::npos )
    {
        std::cerr << "FAIL: CSV output missing expected values:\n" << csv << std::endl;
        failures++;
    }
    if ( csv.find( "3769000" ) == std::string::npos )
    {
        std::cerr << "FAIL: CSV output missing expected population value:\n" << csv << std::endl;
        failures++;
    }

    // Clear the table
    table.clear();
    if ( !table.rows().empty() || !table.columns().empty() )
    {
        std::cerr << "FAIL: clear() did not empty the table." << std::endl;
        failures++;
    }

    if ( failures > 0 )
    {
        std::cerr << "Table feature tests failed with " << failures << " error(s)." << std::endl;
        return 1;
    }
    std::cout << "All Table feature tests passed!" << std::endl;

    return 0;
}