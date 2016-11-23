/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from magmablas/magmablas_zc_v1.cpp, mixed zc -> ds, Sun Nov 20 20:20:28 2016

       @author Mark Gates
       
       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "magma_internal.h"
#include "magmablas_v1.h"  // includes v1 prototypes; does NOT map routine names

// These MAGMA v1 routines are all deprecated.
// See corresponding v2 functions for documentation.

/******************************************************************************/
extern "C" void
magmablas_dsaxpycp_v1(
    magma_int_t m,
    magmaFloat_ptr r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b,
    magmaDouble_ptr w)
{
    magmablas_dsaxpycp( m, r, x, b, w, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slag2d_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr       A, magma_int_t lda,
    magma_int_t *info)
{
    magmablas_slag2d( m, n, SA, ldsa, A, lda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_slat2d_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr      A,  magma_int_t lda,
    magma_int_t *info )
{
    magmablas_slat2d( uplo, n, SA, ldsa, A, lda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dlag2s_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr A, magma_int_t lda,
    magmaFloat_ptr SA,       magma_int_t ldsa,
    magma_int_t *info )
{
    magmablas_dlag2s( m, n, A, lda, SA, ldsa, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dlat2s_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_const_ptr  A, magma_int_t lda,
    magmaFloat_ptr        SA, magma_int_t ldsa,
    magma_int_t *info )
{
    magmablas_dlat2s( uplo, n, A, lda, SA, ldsa, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dslaswp_v1(
    magma_int_t n,
    magmaDouble_ptr A, magma_int_t lda,
    magmaFloat_ptr SA,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx )
{
    magmablas_dslaswp( n, A, lda, SA, lda, m, ipiv, incx, magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
