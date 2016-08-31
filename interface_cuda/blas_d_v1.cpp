/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016
 
       @author Mark Gates
       @generated from interface_cuda/blas_z_v1.cpp, normal z -> d, Tue Aug 30 09:38:00 2016
*/
#ifndef MAGMA_NO_V1

#include "common_magma.h"
#include "error.h"

#define REAL

#ifdef HAVE_CUBLAS

// These MAGMA v1 routines are all deprecated.
// See blas_d_v2.cpp for documentation.


// =============================================================================
// Level 1 BLAS

/******************************************************************************/
extern "C" magma_int_t
magma_idamax(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx )
{
    return magma_idamax_q( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" magma_int_t
magma_idamin(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx )
{
    return magma_idamin_q( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" double
magma_dasum(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx )
{
    return magma_dasum_q( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_daxpy(
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    magma_daxpy_q( n, alpha, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dcopy(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    magma_dcopy_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C"
double magma_ddot(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy )
{
    return magma_ddot_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C"
double magma_ddot(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy )
{
    return magma_ddot_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" double
magma_dnrm2(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx )
{
    return magma_dnrm2_q( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_drot(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    double c, double s )
{
    magma_drot_q( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_drot(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    double c, double s )
{
    magma_drot_q( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_drotm(
    magma_int_t n,
    double *dx, magma_int_t incx,
    double *dy, magma_int_t incy,
    const double *param )
{
    magma_drotm_q( n, dx, incx, dy, incy, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_drotmg(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param )
{
    magma_drotmg_q( d1, d2, x1, y1, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
extern "C" void
magma_dscal(
    magma_int_t n,
    double alpha,
    magmaDouble_ptr dx, magma_int_t incx )
{
    magma_dscal_q( n, alpha, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_dscal(
    magma_int_t n,
    double alpha,
    magmaDouble_ptr dx, magma_int_t incx )
{
    magma_dscal_q( n, alpha, dx, incx, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_dswap(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy )
{
    magma_dswap_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


// =============================================================================
// Level 2 BLAS

/******************************************************************************/
extern "C" void
magma_dgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    magma_dgemv_q(
        transA,
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dger(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda )
{
    magma_dger_q(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_dger(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda )
{
    magma_dger_q(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_dsymv(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    magma_dsymv_q(
        uplo,
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsyr(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dA, magma_int_t ldda )
{
    magma_dsyr_q(
        uplo,
        n,
        alpha, dx, incx,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda )
{
    magma_dsyr2_q(
        uplo,
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dtrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dx, magma_int_t incx )
{
    magma_dtrmv_q(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dtrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dx, magma_int_t incx )
{
    magma_dtrsv_q(
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
magma_dgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dgemm_q(
        transA, transB,
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsymm_q(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsyrk_q(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsyr2k_q(
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
magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsymm_q(
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
magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsyrk_q(
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
magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsyr2k_q(
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
magma_dtrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magma_dtrmm_q(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magma_dtrsm_q(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}

#endif // HAVE_CUBLAS

#undef REAL

#endif // MAGMA_NO_V1
