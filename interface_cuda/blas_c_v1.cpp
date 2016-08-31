/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016
 
       @author Mark Gates
       @generated from interface_cuda/blas_z_v1.cpp, normal z -> c, Tue Aug 30 09:38:00 2016
*/
#ifndef MAGMA_NO_V1

#include "common_magma.h"
#include "error.h"

#define COMPLEX

#ifdef HAVE_CUBLAS

// These MAGMA v1 routines are all deprecated.
// See blas_c_v2.cpp for documentation.


// =============================================================================
// Level 1 BLAS

/******************************************************************************/
extern "C" magma_int_t
magma_icamax(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx )
{
    return magma_icamax_q( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" magma_int_t
magma_icamin(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx )
{
    return magma_icamin_q( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" float
magma_scasum(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx )
{
    return magma_scasum_q( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_caxpy(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    magma_caxpy_q( n, alpha, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ccopy(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    magma_ccopy_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C"
magmaFloatComplex magma_cdotc(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy )
{
    return magma_cdotc_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C"
magmaFloatComplex magma_cdotu(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy )
{
    return magma_cdotu_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" float
magma_scnrm2(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx )
{
    return magma_scnrm2_q( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_crot(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float c, magmaFloatComplex s )
{
    magma_crot_q( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_csrot(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float c, float s )
{
    magma_csrot_q( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_crotm(
    magma_int_t n,
    float *dx, magma_int_t incx,
    float *dy, magma_int_t incy,
    const float *param )
{
    magma_crotm_q( n, dx, incx, dy, incy, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_crotmg(
    float *d1, float       *d2,
    float *x1, const float *y1,
    float *param )
{
    magma_crotmg_q( d1, d2, x1, y1, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
extern "C" void
magma_cscal(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx )
{
    magma_cscal_q( n, alpha, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_csscal(
    magma_int_t n,
    float alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx )
{
    magma_csscal_q( n, alpha, dx, incx, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy )
{
    magma_cswap_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


// =============================================================================
// Level 2 BLAS

/******************************************************************************/
extern "C" void
magma_cgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    magma_cgemv_q(
        transA,
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_cgerc(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda )
{
    magma_cgerc_q(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_cgeru(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda )
{
    magma_cgeru_q(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_chemv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    magma_chemv_q(
        uplo,
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_cher(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dA, magma_int_t ldda )
{
    magma_cher_q(
        uplo,
        n,
        alpha, dx, incx,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_cher2(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda )
{
    magma_cher2_q(
        uplo,
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ctrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx )
{
    magma_ctrmv_q(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ctrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx )
{
    magma_ctrsv_q(
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
magma_cgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_cgemm_q(
        transA, transB,
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_csymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_csymm_q(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_csyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_csyrk_q(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_csyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_csyr2k_q(
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
magma_chemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_chemm_q(
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
magma_cherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_cherk_q(
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
magma_cher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magma_cher2k_q(
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
magma_ctrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magma_ctrmm_q(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_ctrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magma_ctrsm_q(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}

#endif // HAVE_CUBLAS

#undef COMPLEX

#endif // MAGMA_NO_V1
