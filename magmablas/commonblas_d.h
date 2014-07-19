/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from commonblas_z.h normal z -> d, Fri Jul 18 17:34:13 2014
*/

#ifndef COMMONBLAS_D_H
#define COMMONBLAS_D_H

#ifdef __cplusplus
extern "C" {
#endif

/* ======================================================================
 * Internal prototypes
 */

// Tesla GEMM kernels
#define MAGMABLAS_DGEMM( name ) \
void magmablas_dgemm_##name( \
    double *C, const double *A, const double *B, \
    magma_int_t m, magma_int_t n, magma_int_t k, \
    magma_int_t lda, magma_int_t ldb, magma_int_t ldc, \
    double alpha, double beta )

MAGMABLAS_DGEMM( a_0  );
MAGMABLAS_DGEMM( ab_0 );
MAGMABLAS_DGEMM( N_N_64_16_16_16_4_special );
MAGMABLAS_DGEMM( N_N_64_16_16_16_4         );
MAGMABLAS_DGEMM( N_T_64_16_4_16_4          );
MAGMABLAS_DGEMM( T_N_32_32_8_8_8           );
MAGMABLAS_DGEMM( T_T_64_16_16_16_4_special );
MAGMABLAS_DGEMM( T_T_64_16_16_16_4         );
                   
void magmablas_dgemm_tesla(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    const double *A, magma_int_t lda,
    const double *B, magma_int_t ldb,
    double beta,
    double *C, magma_int_t ldc );

void magmablas_dgemv_tesla(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    const double *A, magma_int_t lda,
    const double *x, magma_int_t incx,
    double beta,
    double *y, magma_int_t incy );

#ifdef __cplusplus
}
#endif

#endif /* COMMONBLAS_D_H */
