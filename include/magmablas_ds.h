/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated ds Tue Dec 17 13:18:17 2013
*/

#ifndef MAGMABLAS_DS_H
#define MAGMABLAS_DS_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
void magmablas_dsaxpycp(
    magma_int_t m,
    float *R, double *X,
    const double *B, double *W );

void magmablas_daxpycp(
    magma_int_t m,
    double *R, double *X,
    const double *B );

void magmablas_dslaswp(
    magma_int_t n,
    double *A, magma_int_t lda,
    float *SA, magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx );

void magmablas_dlag2s(
    magma_int_t m, magma_int_t n,
    const double *A,  magma_int_t lda,
          float  *SA, magma_int_t ldsa,
    magma_int_t *info );

void magmablas_slag2d(
    magma_int_t m, magma_int_t n,
    const float  *SA, magma_int_t ldsa,
          double *A,  magma_int_t lda,
    magma_int_t *info );

void magmablas_dlat2s(
    char uplo, magma_int_t n,
    const double *A,  magma_int_t lda,
          float  *SA, magma_int_t ldsa,
    magma_int_t *info );

#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_DS_H */
