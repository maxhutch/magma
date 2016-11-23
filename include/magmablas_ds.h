/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magmablas_zc.h, mixed zc -> ds, Sun Nov 20 20:20:46 2016
*/

#ifndef MAGMABLAS_DS_H
#define MAGMABLAS_DS_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
void
magmablas_dsaxpycp(
    magma_int_t m,
    magmaFloat_ptr        r,
    magmaDouble_ptr       x,
    magmaDouble_const_ptr b,
    magmaDouble_ptr       w,
    magma_queue_t queue );

void
magmablas_dslaswp(
    magma_int_t n,
    magmaDouble_ptr A, magma_int_t lda,
    magmaFloat_ptr SA, magma_int_t ldsa,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx,
    magma_queue_t queue );

void
magmablas_dlag2s(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr A,  magma_int_t lda,
    magmaFloat_ptr       SA, magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slag2d(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr       A,  magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_dlat2s(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_const_ptr A,  magma_int_t lda,
    magmaFloat_ptr       SA, magma_int_t ldsa,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slat2d(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr       A,  magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info );

#ifdef __cplusplus
}
#endif

#endif // MAGMABLAS_DS_H
