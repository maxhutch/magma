/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions mixed zc -> ds
*/

#ifndef MAGMABLAS_ZC_V2_H
#define MAGMABLAS_ZC_V2_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
#define magmablas_zcaxpycp    magmablas_zcaxpycp_q
#define magmablas_zclaswp     magmablas_zclaswp_q
#define magmablas_zlag2c      magmablas_zlag2c_q
#define magmablas_clag2z      magmablas_clag2z_q
#define magmablas_zlat2c      magmablas_zlat2c_q
#define magmablas_clat2z      magmablas_clat2z_q

#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_ZC_V2_H */
