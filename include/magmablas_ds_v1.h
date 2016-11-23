/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magmablas_zc_v1.h, mixed zc -> ds, Sun Nov 20 20:20:46 2016
*/

#ifndef MAGMABLAS_DS_V1_H
#define MAGMABLAS_DS_V1_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* Mixed precision */
void
magmablas_dsaxpycp_v1(
    magma_int_t m,
    magmaFloat_ptr  r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b,
    magmaDouble_ptr w );

void
magmablas_dslaswp_v1(
    magma_int_t n,
    magmaDouble_ptr  A, magma_int_t lda,
    magmaFloat_ptr  SA,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx );

void
magmablas_dlag2s_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr  A, magma_int_t lda,
    magmaFloat_ptr        SA, magma_int_t ldsa,
    magma_int_t *info );

void
magmablas_slag2d_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr  SA, magma_int_t ldsa,
    magmaDouble_ptr        A, magma_int_t lda,
    magma_int_t *info );

void
magmablas_dlat2s_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_const_ptr  A, magma_int_t lda,
    magmaFloat_ptr        SA, magma_int_t ldsa,
    magma_int_t *info );

void
magmablas_slat2d_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr  SA, magma_int_t ldsa,
    magmaDouble_ptr        A, magma_int_t lda,
    magma_int_t *info );

#ifdef __cplusplus
}
#endif

#endif // MAGMABLAS_DS_V1_H
