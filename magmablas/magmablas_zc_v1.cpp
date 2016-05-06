/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions mixed zc -> ds

       @author Mark Gates
       
       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "common_magma.h"


/**
    @see magmablas_zcaxpycp_q
    @ingroup magma_zblas1
    ********************************************************************/
extern "C" void
magmablas_zcaxpycp(
    magma_int_t m,
    magmaFloatComplex_ptr r,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_const_ptr b,
    magmaDoubleComplex_ptr w)
{
    magmablas_zcaxpycp_q( m, r, x, b, w, magmablasGetQueue() );
}


/**
    @see magmablas_clag2z_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clag2z(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr SA, magma_int_t ldsa,
    magmaDoubleComplex_ptr       A, magma_int_t lda,
    magma_int_t *info)
{
    magmablas_clag2z_q( m, n, SA, ldsa, A, lda, magmablasGetQueue(), info );
}


/**
    @see magmablas_clat2z_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_clat2z(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_const_ptr SA, magma_int_t ldsa,
    magmaDoubleComplex_ptr      A,  magma_int_t lda,
    magma_int_t *info )
{
    magmablas_clat2z_q( uplo, n, SA, ldsa, A, lda, magmablasGetQueue(), info );
}


/**
    @see magmablas_zlag2c_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlag2c(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr A, magma_int_t lda,
    magmaFloatComplex_ptr SA,       magma_int_t ldsa,
    magma_int_t *info )
{
    magmablas_zlag2c_q( m, n, A, lda, SA, ldsa, magmablasGetQueue(), info );
}


/**
    @see magmablas_zlat2c_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlat2c(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_const_ptr  A, magma_int_t lda,
    magmaFloatComplex_ptr        SA, magma_int_t ldsa,
    magma_int_t *info )
{
    magmablas_zlat2c_q( uplo, n, A, lda, SA, ldsa, magmablasGetQueue(), info );
}


/**
    Note magmablas_zclaswp_q also adds ldsa. This assumes ldsa = lda.
    @see magmablas_zclaswp_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zclaswp(
    magma_int_t n,
    magmaDoubleComplex_ptr A, magma_int_t lda,
    magmaFloatComplex_ptr SA,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx )
{
    magmablas_zclaswp_q( n, A, lda, SA, lda, m, ipiv, incx, magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
