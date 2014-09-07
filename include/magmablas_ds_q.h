/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from magmablas_zc_q.h mixed zc -> ds, Tue Sep  2 12:38:14 2014
*/

#ifndef MAGMABLAS_DS_Q_H
#define MAGMABLAS_DS_Q_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
void magmablas_dsaxpycp_q(
    magma_int_t m,
    float *R, double *X,
    const double *B, double *W,
    magma_queue_t queue );

void magmablas_daxpycp_q(
    magma_int_t m,
    double *R, double *X,
    const double *B,
    magma_queue_t queue  );

void magmablas_dslaswp_q(
    magma_int_t n,
    double *A, magma_int_t lda,
    float *SA, magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx,
    magma_queue_t queue );

void magmablas_dlag2s_q(
    magma_int_t m, magma_int_t n,
    const double *A,  magma_int_t lda,
          float  *SA, magma_int_t ldsa,
    magma_int_t *info,
    magma_queue_t queue );

void magmablas_slag2d_q(
    magma_int_t m, magma_int_t n,
    const float  *SA, magma_int_t ldsa,
          double *A,  magma_int_t lda,
    magma_int_t *info,
    magma_queue_t queue );

void magmablas_dlat2s_q(
    magma_uplo_t uplo, magma_int_t n,
    const double *A,  magma_int_t lda,
          float  *SA, magma_int_t ldsa,
    magma_int_t *info,
    magma_queue_t queue );

void magmablas_slat2d_q(
    magma_uplo_t uplo, magma_int_t n,
    const float  *SA, magma_int_t ldsa,
          double *A,  magma_int_t lda,
    magma_int_t *info,
    magma_queue_t queue );

#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_DS_H */
