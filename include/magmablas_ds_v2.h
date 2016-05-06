/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from include/magmablas_zc_v2.h mixed zc -> ds, Mon May  2 23:31:25 2016
*/

#ifndef MAGMABLAS_DS_V2_H
#define MAGMABLAS_DS_V2_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
#define magmablas_dsaxpycp    magmablas_dsaxpycp_q
#define magmablas_dslaswp     magmablas_dslaswp_q
#define magmablas_dlag2s      magmablas_dlag2s_q
#define magmablas_slag2d      magmablas_slag2d_q
#define magmablas_dlat2s      magmablas_dlat2s_q
#define magmablas_slat2d      magmablas_slat2d_q

#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_DS_V2_H */
