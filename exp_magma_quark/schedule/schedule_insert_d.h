#ifndef SCHEDULE_INSERT_D
#define SCHEDULE_INSERT_D

#include "common_magma.h"

/*CPU task wrapper*/
void schedule_insert_dmemset(double *ptr, double value, magma_int_t n);

void schedule_insert_dgetrf(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, void *colptr);

void schedule_insert_dgetrf_rec(magma_int_t m, magma_int_t n, double *A, magma_int_t LDA, magma_int_t *ipiv, magma_int_t *info, magma_int_t num_threads, void *colptr);

void schedule_insert_dlaswp(magma_int_t n, double *A, magma_int_t LDA, magma_int_t K1, magma_int_t K2, magma_int_t *ipiv, magma_int_t incx, void *colptr);
     
void schedule_insert_dtrsm(char side, char uplo, char transa, char diag, 
                            magma_int_t m, magma_int_t n, double alpha, 
                            double *A, magma_int_t LDA, double *B, magma_int_t LDB, 
                            void *colptr);

void schedule_insert_dgemm(char transa, char transb, 
                           magma_int_t m, magma_int_t n, magma_int_t k, double alpha, 
                           double *A, magma_int_t LDA, double *B, magma_int_t LDB, double beta, double *C, magma_int_t LDC, 
                           void *A_colptr, void *C_colptr);
 
 /*GPU task wrapper*/
void schedule_insert_magma_dmalloc_pinned(magma_int_t size, double **A, void *A_dep_ptr);
void schedule_insert_magma_dfree_pinned(double *A, void *A_dep_ptr);
void schedule_insert_magma_dfree_pinned_index(double **A, int index, void *A_dep_ptr);

void schedule_insert_magma_dsetmatrix(magma_int_t m, magma_int_t nb, double *A_src, magma_int_t LDA, double *dA_dst, magma_int_t dA_LD);

void schedule_insert_magma_dgetmatrix(magma_int_t m, magma_int_t nb, double *dA_src, magma_int_t dA_LD, double *A_dst, magma_int_t LDA);

void schedule_insert_magma_dsetmatrix_transpose(magma_int_t m, magma_int_t nb, double *A_src, magma_int_t LDA, double *dA_dst, magma_int_t dA_LD, double *dwork, magma_int_t dwork_LD, void *A_src_dep_ptr, void *dA_dst_dep_ptr);

void schedule_insert_magma_dgetmatrix_transpose(magma_int_t m, magma_int_t nb, double *dA_src, magma_int_t dA_LD, double *A_dst, magma_int_t LDA, double *dwork, magma_int_t dwork_LD, void *A_dst_dep_ptr);

void schedule_insert_magma_dlaswp( magma_int_t n, double *dA, magma_int_t lda, magma_int_t i1, magma_int_t i2, magma_int_t *ipiv, magma_int_t inci, void *dA_dep_ptr);
void schedule_insert_magma_dtrsm(char side, char uplo, char trans, char diag, 
                                 magma_int_t m, magma_int_t n, double alpha, 
                                 double *dA, magma_int_t lda, double *dB, magma_int_t ldb );
void schedule_insert_magma_dgemm(char transA, char transB, 
                                 magma_int_t m, magma_int_t n, magma_int_t k, double alpha, 
                                 double *dA, magma_int_t lda, double *dB, magma_int_t ldb, double beta, double *dC, magma_int_t ldc );
#endif
