/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
 
       @author Mark Gates
       @generated d Tue Dec 17 13:18:37 2013
*/

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "error.h"

#define REAL

#ifdef HAVE_CUBLAS

// For now, magma constants are the same as cublas v1 constants (character).
// This will change in the future.
#define cublas_side_const(  x )  (x)
#define cublas_uplo_const(  x )  (x)
#define cublas_trans_const( x )  (x)
#define cublas_diag_const(  x )  (x)

// ========================================
// Level 1 BLAS

// --------------------
extern "C"
magma_int_t magma_idamax(
    magma_int_t n,
    const double *dx, magma_int_t incx )
{
    return cublasIdamax( n, dx, incx );
}

// --------------------
extern "C"
magma_int_t magma_idamin(
    magma_int_t n,
    const double *dx, magma_int_t incx )
{
    return cublasIdamin( n, dx, incx );
}

// --------------------
extern "C"
double magma_dasum(
    magma_int_t n,
    const double *dx, magma_int_t incx )
{
    return cublasDasum( n, dx, incx );
}

// --------------------
extern "C"
void magma_daxpy(
    magma_int_t n,
    double alpha,
    const double *dx, magma_int_t incx,
    double       *dy, magma_int_t incy )
{
    cublasDaxpy( n, alpha, dx, incx, dy, incy );
}

// --------------------
extern "C"
void magma_dcopy(
    magma_int_t n,
    const double *dx, magma_int_t incx,
    double       *dy, magma_int_t incy )
{
    cublasDcopy( n, dx, incx, dy, incy );
}

// --------------------
extern "C"
double magma_ddot(
    magma_int_t n,
    const double *dx, magma_int_t incx,
    const double *dy, magma_int_t incy )
{
    return cublasDdot( n, dx, incx, dy, incy );
}

#ifdef COMPLEX
// --------------------
extern "C"
double magma_ddotu(
    magma_int_t n,
    const double *dx, magma_int_t incx,
    const double *dy, magma_int_t incy )
{
    return cublasDdotu( n, dx, incx, dy, incy );
}
#endif

// --------------------
extern "C"
double magma_dnrm2(
    magma_int_t n,
    const double *dx, magma_int_t incx )
{
    return cublasDnrm2( n, dx, incx );
}

// --------------------
extern "C"
void magma_drot(
    magma_int_t n,
    double *dx, magma_int_t incx,
    double *dy, magma_int_t incy,
    double dc, double ds )
{
    cublasDrot( n, dx, incx, dy, incy, dc, ds );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_drot(
    magma_int_t n,
    double *dx, magma_int_t incx,
    double *dy, magma_int_t incy,
    double dc, double ds )
{
    cublasDrot( n, dx, incx, dy, incy, dc, ds );
}
#endif

#ifdef REAL
// --------------------
extern "C"
void magma_drotm(
    magma_int_t n,
    double *dx, magma_int_t incx,
    double *dy, magma_int_t incy,
    const double *param )
{
    cublasDrotm( n, dx, incx, dy, incy, param );
}

// --------------------
extern "C"
void magma_drotmg(
    double *d1, double       *d2,
    double *x1, const double *y1,
    double *param )
{
    cublasDrotmg( d1, d2, x1, y1, param );
}
#endif

// --------------------
extern "C"
void magma_dscal(
    magma_int_t n,
    double alpha,
    double *dx, magma_int_t incx )
{
    cublasDscal( n, alpha, dx, incx );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_dscal(
    magma_int_t n,
    double alpha,
    double *dx, magma_int_t incx )
{
    cublasDdscal( n, alpha, dx, incx );
}
#endif

// --------------------
extern "C"
void magma_dswap(
    magma_int_t n,
    double *dx, magma_int_t incx,
    double *dy, magma_int_t incy )
{
    cublasDswap( n, dx, incx, dy, incy );
}


// ========================================
// Level 2 BLAS

// --------------------
extern "C"
void magma_dgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha,
    const double *dA, magma_int_t ldda,
    const double *dx, magma_int_t incx,
    double beta,
    double       *dy, magma_int_t incy )
{
    cublasDgemv(
        cublas_trans_const( transA ),
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
extern "C"
void magma_dger(
    magma_int_t m, magma_int_t n,
    double alpha,
    const double *dx, magma_int_t incx,
    const double *dy, magma_int_t incy,
    double       *dA, magma_int_t ldda )
{
    cublasDger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_dger(
    magma_int_t m, magma_int_t n,
    double alpha,
    const double *dx, magma_int_t incx,
    const double *dy, magma_int_t incy,
    double       *dA, magma_int_t ldda )
{
    cublasDger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}
#endif

// --------------------
extern "C"
void magma_dsymv(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    const double *dA, magma_int_t ldda,
    const double *dx, magma_int_t incx,
    double beta,
    double       *dy, magma_int_t incy )
{
    cublasDsymv(
        cublas_uplo_const( uplo ),
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
extern "C"
void magma_dsyr(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    const double *dx, magma_int_t incx,
    double       *dA, magma_int_t ldda )
{
    cublasDsyr(
        cublas_uplo_const( uplo ),
        n,
        alpha, dx, incx,
               dA, ldda );
}

// --------------------
extern "C"
void magma_dsyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    const double *dx, magma_int_t incx,
    const double *dy, magma_int_t incy,
    double       *dA, magma_int_t ldda )
{
    cublasDsyr2(
        cublas_uplo_const( uplo ),
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}

// --------------------
extern "C"
void magma_dtrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    const double *dA, magma_int_t ldda,
    double       *dx, magma_int_t incx )
{
    cublasDtrmv(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        n,
        dA, ldda,
        dx, incx );
}

// --------------------
extern "C"
void magma_dtrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    const double *dA, magma_int_t ldda,
    double       *dx, magma_int_t incx )
{
    cublasDtrsv(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        n,
        dA, ldda,
        dx, incx );
}

// ========================================
// Level 3 BLAS

// --------------------
extern "C"
void magma_dgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    const double *dA, magma_int_t ldda,
    const double *dB, magma_int_t lddb,
    double beta,
    double       *dC, magma_int_t lddc )
{
    cublasDgemm(
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    const double *dA, magma_int_t ldda,
    const double *dB, magma_int_t lddb,
    double beta,
    double       *dC, magma_int_t lddc )
{
    cublasDsymm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    const double *dA, magma_int_t ldda,
    double beta,
    double       *dC, magma_int_t lddc )
{
    cublasDsyrk(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    const double *dA, magma_int_t ldda,
    const double *dB, magma_int_t lddb,
    double beta,
    double       *dC, magma_int_t lddc )
{
    cublasDsyr2k(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    const double *dA, magma_int_t ldda,
    const double *dB, magma_int_t lddb,
    double beta,
    double       *dC, magma_int_t lddc )
{
    cublasDsymm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    const double *dA, magma_int_t ldda,
    double beta,
    double       *dC, magma_int_t lddc )
{
    cublasDsyrk(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    const double *dA, magma_int_t ldda,
    const double *dB, magma_int_t lddb,
    double beta,
    double       *dC, magma_int_t lddc )
{
    cublasDsyr2k(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}
#endif // COMPLEX

// --------------------
extern "C"
void magma_dtrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    const double *dA, magma_int_t ldda,
    double       *dB, magma_int_t lddb )
{
    cublasDtrmm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        m, n,
        alpha, dA, ldda,
               dB, lddb );
}

// --------------------
extern "C"
void magma_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    const double *dA, magma_int_t ldda,
    double       *dB, magma_int_t lddb )
{
    cublasDtrsm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        m, n,
        alpha, dA, ldda,
               dB, lddb );
}

#endif // HAVE_CUBLAS

#undef REAL
