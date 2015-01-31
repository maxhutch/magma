/* 
    -- MAGMA (version 1.6.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       May 2013 
 
       @author: Simplice Donfack 
*/
#ifndef MAGMA_INSERT_CORE_D_H
#define MAGMA_INSERT_CORE_D_H
/*CPU task wrapper*/
void magma_insert_core_dmemset(double *ptr, double value, magma_int_t n);

void magma_insert_core_dgetrf(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, void *colptr);

void magma_insert_core_dgetrf_rec(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, magma_int_t num_threads, void *colptr);

void magma_insert_core_dtslu(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, magma_int_t num_threads, void *colptr);

void magma_insert_core_dlaswp(magma_int_t n, double *A, magma_int_t LDA, magma_int_t K1, magma_int_t K2, magma_int_t *ipiv, magma_int_t incx, void *colptr);
     
void magma_insert_core_dtrsm(char side, char uplo, char transa, char diag, 
                            magma_int_t m, magma_int_t n, double alpha, 
                            double *A, magma_int_t LDA, double *B, magma_int_t LDB, 
                            void *colptr);

/*The different with the classic insert_dtrsm is that it allows many dtrsm to be performed on the same column at once*/
void  magma_insert_core_dtrsm_gatherv(char side, char uplo, char transa, char diag, 
                            magma_int_t m, magma_int_t n, double alpha, 
                            double *A, magma_int_t LDA, double *B, magma_int_t LDB, void *colptr);

void magma_insert_core_dgemm(char transa, char transb, 
                           magma_int_t m, magma_int_t n, magma_int_t k, double alpha, 
                           double *A, magma_int_t LDA, double *B, magma_int_t LDB, double beta, double *C, magma_int_t LDC, 
                           void *A_colptr, void *C_colptr);
#endif

