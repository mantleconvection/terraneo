#pragma once

#if defined( __CUDACC__ )
#pragma nv_diag_push
#pragma nv_diag_suppress 20011
#pragma nv_diag_suppress 20012
#pragma nv_diag_suppress 20013
#pragma nv_diag_suppress 20014
#pragma nv_diag_suppress 20015
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include "../../../extern/eigen-3.4.0/Eigen/Eigen"

#pragma clang diagnostic pop

#if defined( __CUDACC__ )
#pragma nv_diag_pop
#endif