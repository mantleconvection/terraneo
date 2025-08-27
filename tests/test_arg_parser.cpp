

#include "util/arg_parser.hpp"
#include "util/init.hpp"

int main( int argc, char** argv )
{
    terra::util::terra_initialize( &argc, &argv );

    const terra::util::ArgParser args( argc, argv );

    args.print();

    return 0;
}