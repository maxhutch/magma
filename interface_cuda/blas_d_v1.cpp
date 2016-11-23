/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Mark Gates
       @generated from interface_cuda/blas_z_v1.cpp, normal z -> d, Sun Nov 20 20:20:18 2016
*/
#ifndef MAGMA_NO_V1

#include "magma_internal.h"
#include "magmablas_v1.h"  // includes v1 prototypes; does NOT map routine names
#include "error.h"

#define REAL

#ifdef HAVE_CUBLAS

// These MAGMA v1 routines are all deprecated.
// See blas_d_v2.cpp for documentation.


// =============================================================================
// Level 1 BLAS

/******************************************************************************/
extern "C" magma_int_t
magma_idamax_v1(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx )
{
    return magma_idamax( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" magma_int_t
magma_idamin_v1(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx )
{
    return magma_idamin( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" double
magma_dasum_v1(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx )
{
    return magma_dasum( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_daxpy_v1(
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    magma_daxpy( n, alpha, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dcopy_v1(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    magma_dcopy( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C"
double magma_ddot_v1(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy )
{
    return magma_ddot( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C"
double magma_ddot_v1(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy )
{
    return magma_ddot( n, dx, incx, dy, incy, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" double
magma_dnrm2_v1(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx )
{
    return magma_dnrm2( n, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_drot_v1(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    double c, double s )
{
    magma_drot( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_drot_v1(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    double c, double s )
{
    magma_drot( n, dx, incx, dy, incy, c, s, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_drotm_v1(
    magma_int_t n,
    double *dx, magma_int_t incx,
    double *dy, magma_int_t incy,
    const double *param )
{
    magma_drotm( n, dx, incx, dy, incy, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
#ifdef REAL
extern "C" void
magma_drotmg_v1(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param )
{
    magma_drotmg( d1, d2, x1, y1, param, magmablasGetQueue() );
}
#endif // REAL


/******************************************************************************/
extern "C" void
magma_dscal_v1(
    magma_int_t n,
    double alpha,
    magmaDouble_ptr dx, magma_int_t incx )
{
    magma_dscal( n, alpha, dx, incx, magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_dscal_v1(
    magma_int_t n,
    double alpha,
    magmaDouble_ptr dx, magma_int_t incx )
{
    magma_dscal( n, alpha, dx, incx, magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_dswap_v1(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy )
{
    magma_dswap( n, dx, incx, dy, incy, magmablasGetQueue() );
}


// =============================================================================
// Level 2 BLAS

/******************************************************************************/
extern "C" void
magma_dgemv_v1(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    magma_dgemv(
        transA,
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dger_v1(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda )
{
    magma_dger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
#ifdef COMPLEX
extern "C" void
magma_dger_v1(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda )
{
    magma_dger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}
#endif // COMPLEX


/******************************************************************************/
extern "C" void
magma_dsymv_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    magma_dsymv(
        uplo,
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsyr_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dA, magma_int_t ldda )
{
    magma_dsyr(
        uplo,
        n,
        alpha, dx, incx,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsyr2_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda )
{
    magma_dsyr2(
        uplo,
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dtrmv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dx, magma_int_t incx )
{
    magma_dtrmv(
        uplo, trans, diag,
        n,
        dA, ldda,
        dx, incx,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dtrsv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dx, magma_int_t incx )
{
    magma_dtrsv(
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
magma_dgemm_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dgemm(
        transA, transB,
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsymm(
        side, uplo,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsyrk(
        uplo, trans,
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dsyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsyr2k(
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
magma_dsymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsymm(
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
magma_dsyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsyrk(
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
magma_dsyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magma_dsyr2k(
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
magma_dtrmm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magma_dtrmm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dtrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magma_dtrsm(
        side, uplo, trans, diag,
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        magmablasGetQueue() );
}

#endif // HAVE_CUBLAS

#undef REAL

#endif // MAGMA_NO_V1
