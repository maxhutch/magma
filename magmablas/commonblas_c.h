/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:45 2013
*/

#ifndef COMMONBLAS_C_H
#define COMMONBLAS_C_H

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * Internal prototypes
 */

// Tesla GEMM kernels
#define MAGMABLAS_CGEMM( name ) \
void magmablas_cgemm_##name( \
    magmaFloatComplex *C, const magmaFloatComplex *A, const magmaFloatComplex *B, \
    magma_int_t m, magma_int_t n, magma_int_t k, \
    magma_int_t lda, magma_int_t ldb, magma_int_t ldc, \
    magmaFloatComplex alpha, magmaFloatComplex beta )

MAGMABLAS_CGEMM( a_0  );
MAGMABLAS_CGEMM( ab_0 );
MAGMABLAS_CGEMM( N_N_64_16_16_16_4_special );
MAGMABLAS_CGEMM( N_N_64_16_16_16_4         );
MAGMABLAS_CGEMM( N_T_64_16_4_16_4          );
MAGMABLAS_CGEMM( T_N_32_32_8_8_8           );
MAGMABLAS_CGEMM( T_T_64_16_16_16_4_special );
MAGMABLAS_CGEMM( T_T_64_16_16_16_4         );
                   
void magmablas_cgemm_tesla(
    char transA, char transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    const magmaFloatComplex *A, magma_int_t lda,
    const magmaFloatComplex *B, magma_int_t ldb,
    magmaFloatComplex beta,
    magmaFloatComplex *C, magma_int_t ldc );

void magmablas_cgemv_tesla(
    char trans, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    const magmaFloatComplex *A, magma_int_t lda,
    const magmaFloatComplex *x, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex *y, magma_int_t incy );

// for tesla, z is not available, and chemv doesn't have _work interface
void magmablas_chemv_tesla(
    char uplo, magma_int_t n, magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t lda,
    const magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex *dy, magma_int_t incy );

//void magmablas_chemv_tesla_work(
//    char uplo, magma_int_t n, magmaFloatComplex alpha,
//    const magmaFloatComplex *dA, magma_int_t lda,
//    const magmaFloatComplex *dx, magma_int_t incx,
//    magmaFloatComplex beta,
//    magmaFloatComplex *dy, magma_int_t incy,
//    magmaFloatComplex *dwork, magma_int_t lwork );

void magmablas_csymv_tesla(
    char uplo, magma_int_t n, magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t lda,
    const magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex *dy, magma_int_t incy );

void magmablas_csymv_tesla_work(
    char uplo, magma_int_t n, magmaFloatComplex alpha,
    const magmaFloatComplex *dA, magma_int_t lda,
    const magmaFloatComplex *dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex *dy, magma_int_t incy,
    magmaFloatComplex *dwork, magma_int_t lwork );

#ifdef __cplusplus
}
#endif

#endif /* COMMONBLAS_C_H */
