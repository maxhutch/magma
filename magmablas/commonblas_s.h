/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from commonblas_z.h normal z -> s, Fri May 30 10:40:44 2014
*/

#ifndef COMMONBLAS_S_H
#define COMMONBLAS_S_H

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * Internal prototypes
 */

// Tesla GEMM kernels
#define MAGMABLAS_SGEMM( name ) \
void magmablas_sgemm_##name( \
    float *C, const float *A, const float *B, \
    magma_int_t m, magma_int_t n, magma_int_t k, \
    magma_int_t lda, magma_int_t ldb, magma_int_t ldc, \
    float alpha, float beta )

MAGMABLAS_SGEMM( a_0  );
MAGMABLAS_SGEMM( ab_0 );
MAGMABLAS_SGEMM( N_N_64_16_16_16_4_special );
MAGMABLAS_SGEMM( N_N_64_16_16_16_4         );
MAGMABLAS_SGEMM( N_T_64_16_4_16_4          );
MAGMABLAS_SGEMM( T_N_32_32_8_8_8           );
MAGMABLAS_SGEMM( T_T_64_16_16_16_4_special );
MAGMABLAS_SGEMM( T_T_64_16_16_16_4         );
                   
void magmablas_sgemm_tesla(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    const float *A, magma_int_t lda,
    const float *B, magma_int_t ldb,
    float beta,
    float *C, magma_int_t ldc );

void magmablas_sgemv_tesla(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    float alpha,
    const float *A, magma_int_t lda,
    const float *x, magma_int_t incx,
    float beta,
    float *y, magma_int_t incy );

// for tesla, z is not available, and chemv doesn't have _work interface
void magmablas_ssymv_tesla(
    magma_uplo_t uplo, magma_int_t n, float alpha,
    const float *dA, magma_int_t lda,
    const float *dx, magma_int_t incx,
    float beta,
    float *dy, magma_int_t incy );

//void magmablas_ssymv_tesla_work(
//    magma_uplo_t uplo, magma_int_t n, float alpha,
//    const float *dA, magma_int_t lda,
//    const float *dx, magma_int_t incx,
//    float beta,
//    float *dy, magma_int_t incy,
//    float *dwork, magma_int_t lwork );

void magmablas_ssymv_tesla(
    magma_uplo_t uplo, magma_int_t n, float alpha,
    const float *dA, magma_int_t lda,
    const float *dx, magma_int_t incx,
    float beta,
    float *dy, magma_int_t incy );

void magmablas_ssymv_tesla_work(
    magma_uplo_t uplo, magma_int_t n, float alpha,
    const float *dA, magma_int_t lda,
    const float *dx, magma_int_t incx,
    float beta,
    float *dy, magma_int_t incy,
    float *dwork, magma_int_t lwork );

#ifdef __cplusplus
}
#endif

#endif /* COMMONBLAS_S_H */
