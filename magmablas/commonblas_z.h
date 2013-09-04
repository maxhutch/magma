/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @precisions normal z -> s d c
*/

#ifndef COMMONBLAS_Z_H
#define COMMONBLAS_Z_H

#ifdef __cplusplus
extern "C" {
#endif

void magmablas_zgemv_MLU( magma_int_t, magma_int_t, magmaDoubleComplex *, magma_int_t, magmaDoubleComplex *, magmaDoubleComplex *);
void magmablas_zgemv32_tesla(char, magma_int_t, magma_int_t, magmaDoubleComplex, magmaDoubleComplex *, magma_int_t, magmaDoubleComplex *, magmaDoubleComplex *);
void magmablas_zgemvt1_tesla(magma_int_t,magma_int_t,magmaDoubleComplex,magmaDoubleComplex *, magma_int_t,magmaDoubleComplex *,magmaDoubleComplex *);
void magmablas_zgemvt2_tesla(magma_int_t,magma_int_t,magmaDoubleComplex,magmaDoubleComplex *, magma_int_t,magmaDoubleComplex *,magmaDoubleComplex *);
void magmablas_zgemvt_tesla(magma_int_t,magma_int_t,magmaDoubleComplex,magmaDoubleComplex *, magma_int_t,magmaDoubleComplex *,magmaDoubleComplex *);
void magmablas_zgemv_tesla(magma_int_t M, magma_int_t N, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *X, magmaDoubleComplex *);

void magmablas_zsymv6(char, magma_int_t, magmaDoubleComplex, magmaDoubleComplex *, magma_int_t, magmaDoubleComplex *, magma_int_t, magmaDoubleComplex, magmaDoubleComplex *, magma_int_t, magmaDoubleComplex *, magma_int_t);

#define MAGMABLAS_ZGEMM( name ) void magmablas_zgemm_kernel_##name(magmaDoubleComplex *C, const magmaDoubleComplex *A, const magmaDoubleComplex *B, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t lda, magma_int_t ldb, magma_int_t ldc, magmaDoubleComplex alpha, magmaDoubleComplex beta)
MAGMABLAS_ZGEMM( a_0                       );
MAGMABLAS_ZGEMM( ab_0                      );
MAGMABLAS_ZGEMM( N_N_64_16_16_16_4         );
MAGMABLAS_ZGEMM( N_N_64_16_16_16_4_special );
MAGMABLAS_ZGEMM( N_T_64_16_4_16_4          );
MAGMABLAS_ZGEMM( T_N_32_32_8_8_8           );
MAGMABLAS_ZGEMM( T_T_64_16_16_16_4         );
MAGMABLAS_ZGEMM( T_T_64_16_16_16_4_v2      );

void magmablas_zgemm_fermi80(char tA, char tB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, const magmaDoubleComplex *A, magma_int_t lda, const magmaDoubleComplex *B, magma_int_t ldb, magmaDoubleComplex beta, magmaDoubleComplex *C, magma_int_t ldc);
void magmablas_zgemm_fermi64(char tA, char tB, magma_int_t m, magma_int_t n, magma_int_t k, magmaDoubleComplex alpha, const magmaDoubleComplex *A, magma_int_t lda, const magmaDoubleComplex *B, magma_int_t ldb, magmaDoubleComplex beta, magmaDoubleComplex *C, magma_int_t ldc);

#ifdef __cplusplus
}
#endif

#endif /* COMMONBLAS_Z_H */
