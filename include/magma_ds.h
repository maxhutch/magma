/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magma_zc.h mixed zc -> ds, Fri May 30 10:40:32 2014
*/

#ifndef MAGMA_DS_H
#define MAGMA_DS_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mixed precision */
magma_int_t magma_dsgesv_gpu(   magma_trans_t trans, magma_int_t N, magma_int_t NRHS,
                                double *dA, magma_int_t ldda,
                                magma_int_t *IPIV, magma_int_t *dIPIV,
                                double *dB, magma_int_t lddb,
                                double *dX, magma_int_t lddx,
                                double *dworkd, float *dworks,
                                magma_int_t *iter, magma_int_t *info );

magma_int_t magma_dsgetrs_gpu(  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                                float  *dA, magma_int_t ldda,
                                magma_int_t *ipiv,
                                double *dB, magma_int_t lddb,
                                double *dX, magma_int_t lddx,
                                float  *dSX,
                                magma_int_t *info );

magma_int_t magma_dsposv_gpu(   magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                                double *dA, magma_int_t ldda,
                                double *dB, magma_int_t lddb,
                                double *dX, magma_int_t lddx,
                                double *dworkd, float *dworks,
                                magma_int_t *iter, magma_int_t *info );

magma_int_t magma_dsgeqrsv_gpu( magma_int_t M, magma_int_t N, magma_int_t NRHS,
                                double *dA,  magma_int_t ldda,
                                double *dB,  magma_int_t lddb,
                                double *dX,  magma_int_t lddx,
                                magma_int_t *iter,    magma_int_t *info );

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_DS_H */
