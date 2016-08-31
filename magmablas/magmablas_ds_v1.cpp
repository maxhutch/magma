/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016

       @generated from magmablas/magmablas_zc_v1.cpp, mixed zc -> ds, Tue Aug 30 09:38:36 2016

       @author Mark Gates
       
       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "common_magma.h"

// These MAGMA v1 routines are all deprecated.
// See corresponding v2 functions for documentation.

/******************************************************************************/
extern "C" void
magmablas_dsaxpycp(
    magma_int_t m,
    magmaFloat_ptr r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b,
    magmaDouble_ptr w)
{
    magmablas_dsaxpycp_q( m, r, x, b, w, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slag2d(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr       A, magma_int_t lda,
    magma_int_t *info)
{
    magmablas_slag2d_q( m, n, SA, ldsa, A, lda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_slat2d(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr      A,  magma_int_t lda,
    magma_int_t *info )
{
    magmablas_slat2d_q( uplo, n, SA, ldsa, A, lda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dlag2s(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr A, magma_int_t lda,
    magmaFloat_ptr SA,       magma_int_t ldsa,
    magma_int_t *info )
{
    magmablas_dlag2s_q( m, n, A, lda, SA, ldsa, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dlat2s(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_const_ptr  A, magma_int_t lda,
    magmaFloat_ptr        SA, magma_int_t ldsa,
    magma_int_t *info )
{
    magmablas_dlat2s_q( uplo, n, A, lda, SA, ldsa, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dslaswp(
    magma_int_t n,
    magmaDouble_ptr A, magma_int_t lda,
    magmaFloat_ptr SA,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx )
{
    magmablas_dslaswp_q( n, A, lda, SA, lda, m, ipiv, incx, magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
