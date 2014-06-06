/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
 
       @author Mark Gates
       @generated s Tue Dec 17 13:18:37 2013
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
magma_int_t magma_isamax(
    magma_int_t n,
    const float *dx, magma_int_t incx )
{
    return cublasIsamax( n, dx, incx );
}

// --------------------
extern "C"
magma_int_t magma_isamin(
    magma_int_t n,
    const float *dx, magma_int_t incx )
{
    return cublasIsamin( n, dx, incx );
}

// --------------------
extern "C"
float magma_sasum(
    magma_int_t n,
    const float *dx, magma_int_t incx )
{
    return cublasSasum( n, dx, incx );
}

// --------------------
extern "C"
void magma_saxpy(
    magma_int_t n,
    float alpha,
    const float *dx, magma_int_t incx,
    float       *dy, magma_int_t incy )
{
    cublasSaxpy( n, alpha, dx, incx, dy, incy );
}

// --------------------
extern "C"
void magma_scopy(
    magma_int_t n,
    const float *dx, magma_int_t incx,
    float       *dy, magma_int_t incy )
{
    cublasScopy( n, dx, incx, dy, incy );
}

// --------------------
extern "C"
float magma_sdot(
    magma_int_t n,
    const float *dx, magma_int_t incx,
    const float *dy, magma_int_t incy )
{
    return cublasSdot( n, dx, incx, dy, incy );
}

#ifdef COMPLEX
// --------------------
extern "C"
float magma_sdotu(
    magma_int_t n,
    const float *dx, magma_int_t incx,
    const float *dy, magma_int_t incy )
{
    return cublasSdotu( n, dx, incx, dy, incy );
}
#endif

// --------------------
extern "C"
float magma_snrm2(
    magma_int_t n,
    const float *dx, magma_int_t incx )
{
    return cublasSnrm2( n, dx, incx );
}

// --------------------
extern "C"
void magma_srot(
    magma_int_t n,
    float *dx, magma_int_t incx,
    float *dy, magma_int_t incy,
    float dc, float ds )
{
    cublasSrot( n, dx, incx, dy, incy, dc, ds );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_srot(
    magma_int_t n,
    float *dx, magma_int_t incx,
    float *dy, magma_int_t incy,
    float dc, float ds )
{
    cublasSrot( n, dx, incx, dy, incy, dc, ds );
}
#endif

#ifdef REAL
// --------------------
extern "C"
void magma_srotm(
    magma_int_t n,
    float *dx, magma_int_t incx,
    float *dy, magma_int_t incy,
    const float *param )
{
    cublasSrotm( n, dx, incx, dy, incy, param );
}

// --------------------
extern "C"
void magma_srotmg(
    float *d1, float       *d2,
    float *x1, const float *y1,
    float *param )
{
    cublasSrotmg( d1, d2, x1, y1, param );
}
#endif

// --------------------
extern "C"
void magma_sscal(
    magma_int_t n,
    float alpha,
    float *dx, magma_int_t incx )
{
    cublasSscal( n, alpha, dx, incx );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_sscal(
    magma_int_t n,
    float alpha,
    float *dx, magma_int_t incx )
{
    cublasSsscal( n, alpha, dx, incx );
}
#endif

// --------------------
extern "C"
void magma_sswap(
    magma_int_t n,
    float *dx, magma_int_t incx,
    float *dy, magma_int_t incy )
{
    cublasSswap( n, dx, incx, dy, incy );
}


// ========================================
// Level 2 BLAS

// --------------------
extern "C"
void magma_sgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha,
    const float *dA, magma_int_t ldda,
    const float *dx, magma_int_t incx,
    float beta,
    float       *dy, magma_int_t incy )
{
    cublasSgemv(
        cublas_trans_const( transA ),
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
extern "C"
void magma_sger(
    magma_int_t m, magma_int_t n,
    float alpha,
    const float *dx, magma_int_t incx,
    const float *dy, magma_int_t incy,
    float       *dA, magma_int_t ldda )
{
    cublasSger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_sger(
    magma_int_t m, magma_int_t n,
    float alpha,
    const float *dx, magma_int_t incx,
    const float *dy, magma_int_t incy,
    float       *dA, magma_int_t ldda )
{
    cublasSger(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}
#endif

// --------------------
extern "C"
void magma_ssymv(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    const float *dA, magma_int_t ldda,
    const float *dx, magma_int_t incx,
    float beta,
    float       *dy, magma_int_t incy )
{
    cublasSsymv(
        cublas_uplo_const( uplo ),
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
extern "C"
void magma_ssyr(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    const float *dx, magma_int_t incx,
    float       *dA, magma_int_t ldda )
{
    cublasSsyr(
        cublas_uplo_const( uplo ),
        n,
        alpha, dx, incx,
               dA, ldda );
}

// --------------------
extern "C"
void magma_ssyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    const float *dx, magma_int_t incx,
    const float *dy, magma_int_t incy,
    float       *dA, magma_int_t ldda )
{
    cublasSsyr2(
        cublas_uplo_const( uplo ),
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}

// --------------------
extern "C"
void magma_strmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    const float *dA, magma_int_t ldda,
    float       *dx, magma_int_t incx )
{
    cublasStrmv(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        n,
        dA, ldda,
        dx, incx );
}

// --------------------
extern "C"
void magma_strsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    const float *dA, magma_int_t ldda,
    float       *dx, magma_int_t incx )
{
    cublasStrsv(
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
void magma_sgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    const float *dA, magma_int_t ldda,
    const float *dB, magma_int_t lddb,
    float beta,
    float       *dC, magma_int_t lddc )
{
    cublasSgemm(
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    const float *dA, magma_int_t ldda,
    const float *dB, magma_int_t lddb,
    float beta,
    float       *dC, magma_int_t lddc )
{
    cublasSsymm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    const float *dA, magma_int_t ldda,
    float beta,
    float       *dC, magma_int_t lddc )
{
    cublasSsyrk(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    const float *dA, magma_int_t ldda,
    const float *dB, magma_int_t lddb,
    float beta,
    float       *dC, magma_int_t lddc )
{
    cublasSsyr2k(
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
void magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    const float *dA, magma_int_t ldda,
    const float *dB, magma_int_t lddb,
    float beta,
    float       *dC, magma_int_t lddc )
{
    cublasSsymm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    const float *dA, magma_int_t ldda,
    float beta,
    float       *dC, magma_int_t lddc )
{
    cublasSsyrk(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    const float *dA, magma_int_t ldda,
    const float *dB, magma_int_t lddb,
    float beta,
    float       *dC, magma_int_t lddc )
{
    cublasSsyr2k(
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
void magma_strmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    const float *dA, magma_int_t ldda,
    float       *dB, magma_int_t lddb )
{
    cublasStrmm(
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
void magma_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    const float *dA, magma_int_t ldda,
    float       *dB, magma_int_t lddb )
{
    cublasStrsm(
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
