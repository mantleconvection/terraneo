

#include "terra/kokkos_wrapper.hpp"
#include "terra/types.hpp"

using terra::real_t;

int main()
{
    Kokkos::View< real_t* > test( "test", 10 );
}