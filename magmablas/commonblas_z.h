/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal z -> s d c
*/

#ifndef COMMONBLAS_Z_H
#define COMMONBLAS_Z_H

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * Internal prototypes
 */

// Tesla GEMM kernels
#define MAGMABLAS_ZGEMM( name ) \
void magmablas_zgemm_##name( \
    magmaDoubleComplex *C, const magmaDoubleComplex *A, const magmaDoubleComplex *B, \
    magma_int_t m, magma_int_t n, magma_int_t k, \
    magma_int_t lda, magma_int_t ldb, magma_int_t ldc, \
    magmaDoubleComplex alpha, magmaDoubleComplex beta )

MAGMABLAS_ZGEMM( a_0  );
MAGMABLAS_ZGEMM( ab_0 );
MAGMABLAS_ZGEMM( N_N_64_16_16_16_4_special );
MAGMABLAS_ZGEMM( N_N_64_16_16_16_4         );
MAGMABLAS_ZGEMM( N_T_64_16_4_16_4          );
MAGMABLAS_ZGEMM( T_N_32_32_8_8_8           );
MAGMABLAS_ZGEMM( T_T_64_16_16_16_4_special );
MAGMABLAS_ZGEMM( T_T_64_16_16_16_4         );
                   
void magmablas_zgemm_tesla(
    char transA, char transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    const magmaDoubleComplex *B, magma_int_t ldb,
    magmaDoubleComplex beta,
    magmaDoubleComplex *C, magma_int_t ldc );

void magmablas_zgemv_tesla(
    char trans, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    const magmaDoubleComplex *x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, magma_int_t incy );

// for tesla, z is not available, and chemv doesn't have _work interface
void magmablas_zhemv_tesla(
    char uplo, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t lda,
    const magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *dy, magma_int_t incy );

//void magmablas_zhemv_tesla_work(
//    char uplo, magma_int_t n, magmaDoubleComplex alpha,
//    const magmaDoubleComplex *dA, magma_int_t lda,
//    const magmaDoubleComplex *dx, magma_int_t incx,
//    magmaDoubleComplex beta,
//    magmaDoubleComplex *dy, magma_int_t incy,
//    magmaDoubleComplex *dwork, magma_int_t lwork );

void magmablas_zsymv_tesla(
    char uplo, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t lda,
    const magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *dy, magma_int_t incy );

void magmablas_zsymv_tesla_work(
    char uplo, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t lda,
    const magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *dy, magma_int_t incy,
    magmaDoubleComplex *dwork, magma_int_t lwork );

#ifdef __cplusplus
}
#endif

#endif /* COMMONBLAS_Z_H */
