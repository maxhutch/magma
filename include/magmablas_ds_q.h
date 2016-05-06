/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from include/magmablas_zc_q.h mixed zc -> ds, Mon May  2 23:31:25 2016
*/

#ifndef MAGMABLAS_DS_Q_H
#define MAGMABLAS_DS_Q_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
void
magmablas_dsaxpycp_q(
    magma_int_t m,
    magmaFloat_ptr  r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b,
    magmaDouble_ptr w,
    magma_queue_t queue );

void
magmablas_dslaswp_q(
    magma_int_t n,
    magmaDouble_ptr A, magma_int_t lda,
    magmaFloat_ptr SA, magma_int_t ldsa,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx,
    magma_queue_t queue );

void
magmablas_dlag2s_q(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr A,  magma_int_t lda,
    magmaFloat_ptr       SA, magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slag2d_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr       A,  magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_dlat2s_q(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_const_ptr A,  magma_int_t lda,
    magmaFloat_ptr       SA, magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slat2d_q(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr       A,  magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info );

#ifdef __cplusplus
}
#endif

#endif /* MAGMABLAS_DS_H */
