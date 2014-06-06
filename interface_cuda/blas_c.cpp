/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
 
       @author Mark Gates
       @generated c Tue Dec 17 13:18:37 2013
*/

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"
#include "error.h"

#define COMPLEX

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
magma_int_t magma_icamax(
    magma_int_t n,
    const magmaFloatComplex *dx, magma_int_t incx )
{
    return cublasIcamax( n, dx, incx );
}

// --------------------
extern "C"
magma_int_t magma_icamin(
    magma_int_t n,
    const magmaFloatComplex *dx, magma_int_t incx )
{
    return cublasIcamin( n, dx, incx );
}

// --------------------
extern "C"
float magma_scasum(
    magma_int_t n,
    const magmaFloatComplex *dx, magma_int_t incx )
{
    return cublasScasum( n, dx, incx );
}

// --------------------
extern "C"
void magma_caxpy(
    magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex       *dy, magma_int_t incy )
{
    cublasCaxpy( n, alpha, dx, incx, dy, incy );
}

// --------------------
extern "C"
void magma_ccopy(
    magma_int_t n,
    const magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex       *dy, magma_int_t incy )
{
    cublasCcopy( n, dx, incx, dy, incy );
}

// --------------------
extern "C"
magmaFloatComplex magma_cdotc(
    magma_int_t n,
    const magmaFloatComplex *dx, magma_int_t incx,
    const magmaFloatComplex *dy, magma_int_t incy )
{
    return cublasCdotc( n, dx, incx, dy, incy );
}

#ifdef COMPLEX
// --------------------
extern "C"
magmaFloatComplex magma_cdotu(
    magma_int_t n,
    const magmaFloatComplex *dx, magma_int_t incx,
    const magmaFloatComplex *dy, magma_int_t incy )
{
    return cublasCdotu( n, dx, incx, dy, incy );
}
#endif

// --------------------
extern "C"
float magma_scnrm2(
    magma_int_t n,
    const magmaFloatComplex *dx, magma_int_t incx )
{
    return cublasScnrm2( n, dx, incx );
}

// --------------------
extern "C"
void magma_crot(
    magma_int_t n,
    magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex *dy, magma_int_t incy,
    float dc, magmaFloatComplex ds )
{
    cublasCrot( n, dx, incx, dy, incy, dc, ds );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_csrot(
    magma_int_t n,
    magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex *dy, magma_int_t incy,
    float dc, float ds )
{
    cublasCsrot( n, dx, incx, dy, incy, dc, ds );
}
#endif

#ifdef REAL
// --------------------
extern "C"
void magma_crotm(
    magma_int_t n,
    float *dx, magma_int_t incx,
    float *dy, magma_int_t incy,
    const float *param )
{
    cublasCrotm( n, dx, incx, dy, incy, param );
}

// --------------------
extern "C"
void magma_crotmg(
    float *d1, float       *d2,
    float *x1, const float *y1,
    float *param )
{
    cublasCrotmg( d1, d2, x1, y1, param );
}
#endif

// --------------------
extern "C"
void magma_cscal(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex *dx, magma_int_t incx )
{
    cublasCscal( n, alpha, dx, incx );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_csscal(
    magma_int_t n,
    float alpha,
    magmaFloatComplex *dx, magma_int_t incx )
{
    cublasCsscal( n, alpha, dx, incx );
}
#endif

// --------------------
extern "C"
void magma_cswap(
    magma_int_t n,
    magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex *dy, magma_int_t incy )
{
    cublasCswap( n, dx, incx, dy, incy );
}


// ========================================
// Level 2 BLAS

// --------------------
extern "C"
void magma_cgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    const magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex       *dy, magma_int_t incy )
{
    cublasCgemv(
        cublas_trans_const( transA ),
        m, n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
extern "C"
void magma_cgerc(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dx, magma_int_t incx,
    const magmaFloatComplex *dy, magma_int_t incy,
    magmaFloatComplex       *dA, magma_int_t ldda )
{
    cublasCgerc(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}

#ifdef COMPLEX
// --------------------
extern "C"
void magma_cgeru(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dx, magma_int_t incx,
    const magmaFloatComplex *dy, magma_int_t incy,
    magmaFloatComplex       *dA, magma_int_t ldda )
{
    cublasCgeru(
        m, n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}
#endif

// --------------------
extern "C"
void magma_chemv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    const magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex       *dy, magma_int_t incy )
{
    cublasChemv(
        cublas_uplo_const( uplo ),
        n,
        alpha, dA, ldda,
               dx, incx,
        beta,  dy, incy );
}

// --------------------
extern "C"
void magma_cher(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    const magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex       *dA, magma_int_t ldda )
{
    cublasCher(
        cublas_uplo_const( uplo ),
        n,
        alpha, dx, incx,
               dA, ldda );
}

// --------------------
extern "C"
void magma_cher2(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dx, magma_int_t incx,
    const magmaFloatComplex *dy, magma_int_t incy,
    magmaFloatComplex       *dA, magma_int_t ldda )
{
    cublasCher2(
        cublas_uplo_const( uplo ),
        n,
        alpha, dx, incx,
               dy, incy,
               dA, ldda );
}

// --------------------
extern "C"
void magma_ctrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    const magmaFloatComplex *dA, magma_int_t ldda,
    magmaFloatComplex       *dx, magma_int_t incx )
{
    cublasCtrmv(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        n,
        dA, ldda,
        dx, incx );
}

// --------------------
extern "C"
void magma_ctrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    const magmaFloatComplex *dA, magma_int_t ldda,
    magmaFloatComplex       *dx, magma_int_t incx )
{
    cublasCtrsv(
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
void magma_cgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    const magmaFloatComplex *dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex       *dC, magma_int_t lddc )
{
    cublasCgemm(
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        m, n, k,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_csymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    const magmaFloatComplex *dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex       *dC, magma_int_t lddc )
{
    cublasCsymm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_csyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex       *dC, magma_int_t lddc )
{
    cublasCsyrk(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_csyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    const magmaFloatComplex *dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex       *dC, magma_int_t lddc )
{
    cublasCsyr2k(
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
void magma_chemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    const magmaFloatComplex *dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex       *dC, magma_int_t lddc )
{
    cublasChemm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        alpha, dA, ldda,
               dB, lddb,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_cherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    float beta,
    magmaFloatComplex       *dC, magma_int_t lddc )
{
    cublasCherk(
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        alpha, dA, ldda,
        beta,  dC, lddc );
}

// --------------------
extern "C"
void magma_cher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    const magmaFloatComplex *dB, magma_int_t lddb,
    float beta,
    magmaFloatComplex       *dC, magma_int_t lddc )
{
    cublasCher2k(
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
void magma_ctrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    magmaFloatComplex       *dB, magma_int_t lddb )
{
    cublasCtrmm(
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
void magma_ctrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t ldda,
    magmaFloatComplex       *dB, magma_int_t lddb )
{
    cublasCtrsm(
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        m, n,
        alpha, dA, ldda,
               dB, lddb );
}

#endif // HAVE_CUBLAS

#undef COMPLEX
