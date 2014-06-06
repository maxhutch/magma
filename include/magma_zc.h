/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions mixed zc -> ds
*/

#ifndef MAGMA_ZC_H
#define MAGMA_ZC_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mixed precision */
magma_int_t magma_zcgesv_gpu(   char trans, magma_int_t N, magma_int_t NRHS,
                                magmaDoubleComplex *dA, magma_int_t ldda,
                                magma_int_t *IPIV, magma_int_t *dIPIV,
                                magmaDoubleComplex *dB, magma_int_t lddb,
                                magmaDoubleComplex *dX, magma_int_t lddx,
                                magmaDoubleComplex *dworkd, magmaFloatComplex *dworks,
                                magma_int_t *iter, magma_int_t *info );

magma_int_t magma_zcgetrs_gpu(  char trans, magma_int_t n, magma_int_t nrhs,
                                magmaFloatComplex  *dA, magma_int_t ldda,
                                magma_int_t *ipiv,
                                magmaDoubleComplex *dB, magma_int_t lddb,
                                magmaDoubleComplex *dX, magma_int_t lddx,
                                magmaFloatComplex  *dSX,
                                magma_int_t *info );

magma_int_t magma_zcposv_gpu(   char uplo, magma_int_t n, magma_int_t nrhs,
                                magmaDoubleComplex *dA, magma_int_t ldda,
                                magmaDoubleComplex *dB, magma_int_t lddb,
                                magmaDoubleComplex *dX, magma_int_t lddx,
                                magmaDoubleComplex *dworkd, magmaFloatComplex *dworks,
                                magma_int_t *iter, magma_int_t *info );

magma_int_t magma_zcgeqrsv_gpu( magma_int_t M, magma_int_t N, magma_int_t NRHS,
                                magmaDoubleComplex *dA,  magma_int_t ldda,
                                magmaDoubleComplex *dB,  magma_int_t lddb,
                                magmaDoubleComplex *dX,  magma_int_t lddx,
                                magma_int_t *iter,    magma_int_t *info );

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_ZC_H */
