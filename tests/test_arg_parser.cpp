#include "terra/util/cli11_helper.hpp"
#include "terra/util/cli11_wrapper.hpp"
#include "terra/util/init.hpp"

struct MyParameters
{
    int    level;
    double simulation_parameter;
};

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    CLI::App app{ "CLI11 inspection example" };

    // A switch
    bool verbose = false;
    app.add_flag( "-v,--verbose", verbose, "Enable verbose mode" );

    // A positional argument (required)
    std::string input_file;
    app.add_option( "input-file", input_file, "Input file" )->required();

    // A flag that can be passed multiple times and occurence is counted.
    int count = 0;
    app.add_flag( "-c,--count", count, "Count flag (increments)" );

    // An option with default value
    std::string output_file = "default.txt";
    app.add_option( "-o,--output-file", output_file, "Output file" )->default_val( output_file );

    // A list of strings
    std::vector< std::string > tags;
    app.add_option( "-l,--string-list", tags, "List of tags" );

    // Note that we can reference struct entries to nicely specify defaults and pass the parsed arguments to other
    // functions. We can still make the required.
    MyParameters params{ .level = 4, .simulation_parameter = 0.1 };
    // Using this wrapper function below instead of app.add_option(...) sets the struct parameter as default.
    terra::util::add_option_with_default( app, "--level", params.level );
    terra::util::add_option_with_default( app, "--simulation-parameter", params.simulation_parameter )->required();

    // Parse arguments (it's a simple macro that handles exceptions nicely).
    CLI11_PARSE( app, argc, argv );

    // Print overview.
    terra::util::print_cli_summary( app, std::cout );

    std::cout << std::endl;
    std::cout << "MyParameters:" << std::endl;
    std::cout << "Level:                " << params.level << std::endl;
    std::cout << "Simulation parameter: " << params.simulation_parameter << std::endl;
}
