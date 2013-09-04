/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @generated c Tue Aug 13 16:45:27 2013
*/

#ifndef COMMONBLAS_C_H
#define COMMONBLAS_C_H

#ifdef __cplusplus
extern "C" {
#endif

void magmablas_cgemv_MLU( magma_int_t, magma_int_t, magmaFloatComplex *, magma_int_t, magmaFloatComplex *, magmaFloatComplex *);
void magmablas_cgemv32_tesla(char, magma_int_t, magma_int_t, magmaFloatComplex, magmaFloatComplex *, magma_int_t, magmaFloatComplex *, magmaFloatComplex *);
void magmablas_cgemvt1_tesla(magma_int_t,magma_int_t,magmaFloatComplex,magmaFloatComplex *, magma_int_t,magmaFloatComplex *,magmaFloatComplex *);
void magmablas_cgemvt2_tesla(magma_int_t,magma_int_t,magmaFloatComplex,magmaFloatComplex *, magma_int_t,magmaFloatComplex *,magmaFloatComplex *);
void magmablas_cgemvt_tesla(magma_int_t,magma_int_t,magmaFloatComplex,magmaFloatComplex *, magma_int_t,magmaFloatComplex *,magmaFloatComplex *);
void magmablas_cgemv_tesla(magma_int_t M, magma_int_t N, magmaFloatComplex *A, magma_int_t lda, magmaFloatComplex *X, magmaFloatComplex *);

void magmablas_csymv6(char, magma_int_t, magmaFloatComplex, magmaFloatComplex *, magma_int_t, magmaFloatComplex *, magma_int_t, magmaFloatComplex, magmaFloatComplex *, magma_int_t, magmaFloatComplex *, magma_int_t);

#define MAGMABLAS_CGEMM( name ) void magmablas_cgemm_kernel_##name(magmaFloatComplex *C, const magmaFloatComplex *A, const magmaFloatComplex *B, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t lda, magma_int_t ldb, magma_int_t ldc, magmaFloatComplex alpha, magmaFloatComplex beta)
MAGMABLAS_CGEMM( a_0                       );
MAGMABLAS_CGEMM( ab_0                      );
MAGMABLAS_CGEMM( N_N_64_16_16_16_4         );
MAGMABLAS_CGEMM( N_N_64_16_16_16_4_special );
MAGMABLAS_CGEMM( N_T_64_16_4_16_4          );
MAGMABLAS_CGEMM( T_N_32_32_8_8_8           );
MAGMABLAS_CGEMM( T_T_64_16_16_16_4         );
MAGMABLAS_CGEMM( T_T_64_16_16_16_4_v2      );

void magmablas_cgemm_fermi80(char tA, char tB, magma_int_t m, magma_int_t n, magma_int_t k, magmaFloatComplex alpha, const magmaFloatComplex *A, magma_int_t lda, const magmaFloatComplex *B, magma_int_t ldb, magmaFloatComplex beta, magmaFloatComplex *C, magma_int_t ldc);
void magmablas_cgemm_fermi64(char tA, char tB, magma_int_t m, magma_int_t n, magma_int_t k, magmaFloatComplex alpha, const magmaFloatComplex *A, magma_int_t lda, const magmaFloatComplex *B, magma_int_t ldb, magmaFloatComplex beta, magmaFloatComplex *C, magma_int_t ldc);

#ifdef __cplusplus
}
#endif

#endif /* COMMONBLAS_C_H */
