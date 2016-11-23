/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Mark Gates
       @generated from interface_cuda/blas_z_v1.cpp, normal z -> c, Sun Nov 20 20:20:18 2016
*/
#ifndef MAGMA_NO_V1

#include "magma_internal.h"
#include "magmablas_v1.h"  // includes v1 prototypes; does NOT map routine names
#include "error.h"

#define COMPLEX

#ifdef HAVE_CUBLAS

// These MAGMA v1 routines are all deprecated.
// See blas_c_v2.cpp for documentation.


// =============================================================================
// Level 1 BLAS

/******************************************************************************/
extern "C" magma_int_t
magma_icamax_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx )
{
    return magma_icamax( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" magma_int_t
magma_icamin_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx )
{
    return magma_icamin( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" float
magma_scasum_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx )
{
    return magma_scasum( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_caxpy_v1(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    magma_caxpy( n, alpha, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ccopy_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    magma_ccopy( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C"
magmaFloatComplex magma_cdotc_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy )
{
    return magma_cdotc( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C"
magmaFloatComplex magma_cdotu_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy )
{
    return magma_cdotu( n, dx, incx, dy, incy, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" float
magma_scnrm2_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx )
{
    return magma_scnrm2( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_crot_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float c, magmaFloatComplex s )
{
    magma_crot( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_csrot_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float c, float s )
{
    magma_csrot( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_crotm_v1(
    magma_int_t n,
    float *dx, magma_int_t incx,
    float *dy, magma_int_t incy,
    const float *param )
{
    magma_crotm( n, dx, incx, dy, incy, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_crotmg_v1(
    float *d1, float       *d2,
    float *x1, const float *y1,
    float *param )
{
    magma_crotmg( d1, d2, x1, y1, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
extern "C" void
magma_cscal_v1(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx )
{
    magma_cscal( n, alpha, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_csscal_v1(
    magma_int_t n,
    float alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx )
{
    magma_csscal( n, alpha, dx, incx, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_cswap_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy )
{
    magma_cswap( n, dx, incx, dy, incy, magmablasGetQueue() );
}


// =============================================================================
// Level 2 BLAS

/******************************************************************************/
extern "C" void
magma_cgemv_v1(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    magma_cgemv(
        transA,
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_cgerc_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda )
{
    magma_cgerc(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_cgeru_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda )
{
    magma_cgeru(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_chemv_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    magma_chemv(
        uplo,
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_cher_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dA, magma_int_t ldda )
{
    magma_cher(
        uplo,
        n,
        alpha, dx, incx,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_cher2_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda )
{
    magma_cher2(
        uplo,
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ctrmv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx )
{
    magma_ctrmv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ctrsv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx )
{
    magma_ctrsv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        magmablasGetQueue() );
}


// =============================================================================
// Level 3 BLAS

/******************************************************************************/
extern "C" void
magma_cgemm_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_cgemm(
        transA, transB,
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_csymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_csymm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_csyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_csyrk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_csyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_csyr2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_chemm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_chemm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_cherk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_cherk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_cher2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_cher2k(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_ctrmm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magma_ctrmm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ctrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magma_ctrsm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}

#endif // HAVE_CUBLAS

#undef COMPLEX

#endif // MAGMA_NO_V1
