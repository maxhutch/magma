/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions mixed zc -> ds
*/

#ifndef MAGMABLAS_ZC_H
#define MAGMABLAS_ZC_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
void magmablas_zcaxpycp(
    magma_int_t m,
    magmaFloatComplex *R, magmaDoubleComplex *X,
    const magmaDoubleComplex *B, magmaDoubleComplex *W );

void magmablas_zaxpycp(
    magma_int_t m,
    magmaDoubleComplex *R, magmaDoubleComplex *X,
    const magmaDoubleComplex *B );

void magmablas_zclaswp(
    magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaFloatComplex *SA, magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx );

void magmablas_zlag2c(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *A,  magma_int_t lda,
          magmaFloatComplex  *SA, magma_int_t ldsa,
    magma_int_t *info );

void magmablas_clag2z(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex  *SA, magma_int_t ldsa,
          magmaDoubleComplex *A,  magma_int_t lda,
    magma_int_t *info );

void magmablas_zlat2c(
    magma_uplo_t uplo, magma_int_t n,
    const magmaDoubleComplex *A,  magma_int_t lda,
          magmaFloatComplex  *SA, magma_int_t ldsa,
    magma_int_t *info );

void magmablas_clat2z(
    magma_uplo_t uplo, magma_int_t n,
    const magmaFloatComplex  *SA, magma_int_t ldsa,
          magmaDoubleComplex *A,  magma_int_t lda,
    magma_int_t *info );

#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_ZC_H */
