/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from magma_zbatched.h normal z -> c, Sat Nov 15 19:53:54 2014
*/

#ifndef MAGMA_CBATCHED_H
#define MAGMA_CBATCHED_H

#include "magma_types.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  local auxiliary routines
   */
void 
cset_pointer_int(magma_int_t **output_array,
        magma_int_t *input,
        magma_int_t lda,
        magma_int_t row, magma_int_t column, 
        magma_int_t batchSize,
        magma_int_t batchCount);

void 
cset_pointer(magmaFloatComplex **output_array,
                 magmaFloatComplex *input,
                 magma_int_t lda,
                 magma_int_t row, magma_int_t column,
                 magma_int_t batchSize,
                 magma_int_t batchCount);


void 
cset_array(magmaFloatComplex **output_array,
               magmaFloatComplex **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column,
               magma_int_t batchCount);

void 
magma_cdisplace_pointers(magmaFloatComplex **output_array,
               magmaFloatComplex **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount);

  /*
   *  LAPACK batched routines
   */

  /*
   *  BLAS batched routines
   */

void
magmablas_cgemm_batched(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );

void
magmablas_cgemm_batched_lg(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );
void
magmablas_cgemm_batched_k32(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );

void 
magmablas_cgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magmaFloatComplex **dx_array, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex **dy_array, magma_int_t incy,
    magma_int_t batchCount);

void 
magmablas_cherk_NC_batched( magma_trans_t TRANSA, magma_trans_t TRANSB, int m , int n , int k , 
                       magmaFloatComplex alpha, magmaFloatComplex **dA_array, int lda, 
                       magmaFloatComplex **B_array, int ldb, 
                       magmaFloatComplex beta,        magmaFloatComplex **C_array, int ldc, 
                       magma_int_t batchCount);

void
magmablas_cherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    float beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );

void
magmablas_cherk_batched_lg(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    float beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );

void
magmablas_cherk_batched_k32(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    float beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );






magma_int_t 
magma_cpotf2_tile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount);

magma_int_t 
magma_cpotrf_panel(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex* dinvA, magmaFloatComplex** dinvA_array, magma_int_t invA_msize,
    magmaFloatComplex* x, magmaFloatComplex** x_array,  magma_int_t x_msize,
    magma_int_t *info_array,  magma_int_t batchCount, magma_int_t matrixSize);


void 
magmablas_ctrtri_diag_batched_q(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex const * const *dA_array, magma_int_t ldda,
    magmaFloatComplex **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_ctrtri_diag_batched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex const * const *dA_array, magma_int_t ldda,
    magmaFloatComplex **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount);


void 
magmablas_ctrsm_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dB_array,    magma_int_t lddb,
    magma_int_t batchCount);


void 
magmablas_ctrsm_work_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    magmaFloatComplex alpha, 
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dB_array,    magma_int_t lddb,
    magmaFloatComplex** dX_array,    magma_int_t lddx, 
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dA_displ, magmaFloatComplex** dB_displ, 
    magmaFloatComplex** dX_displ, magmaFloatComplex** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount);

void 
magmablas_ctrsm_outofplace_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    magmaFloatComplex alpha, 
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dB_array,    magma_int_t lddb,
    magmaFloatComplex** dX_array,    magma_int_t lddx, 
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dA_displ, magmaFloatComplex** dB_displ, 
    magmaFloatComplex** dX_displ, magmaFloatComplex** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount);

magma_int_t 
magma_cpotrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array,  magma_int_t batchCount);


magma_int_t 
magma_cpotf2_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magmaFloatComplex **dA_displ, 
    magmaFloatComplex **dW_displ,
    magmaFloatComplex **dB_displ, 
    magmaFloatComplex **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_cpotrf_panel_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dX_array,    magma_int_t dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dW0_displ, magmaFloatComplex** dW1_displ, 
    magmaFloatComplex** dW2_displ, magmaFloatComplex** dW3_displ,
    magmaFloatComplex** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_cpotrf_recpanel_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dX_array,    magma_int_t dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dW0_displ, magmaFloatComplex** dW1_displ,  
    magmaFloatComplex** dW2_displ, magmaFloatComplex** dW3_displ,
    magmaFloatComplex** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_cpotrf_rectile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dX_array,    magma_int_t dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dW0_displ, magmaFloatComplex** dW1_displ,  
    magmaFloatComplex** dW2_displ, magmaFloatComplex** dW3_displ,
    magmaFloatComplex** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_cpotrs_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  magmaFloatComplex **dA_array, magma_int_t ldda,
                  magmaFloatComplex **dB_array, magma_int_t lddb,
                  magma_int_t batchCount);

magma_int_t 
magma_cposv_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  magmaFloatComplex **dA_array, magma_int_t ldda,
                  magmaFloatComplex **dB_array, magma_int_t lddb,
                  magma_int_t *dinfo_array,
                  magma_int_t batchCount);

magma_int_t 
magma_cgetrs_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  magmaFloatComplex **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  magmaFloatComplex **dB_array, magma_int_t lddb,
                  magma_int_t batchCount);



void 
magma_claswp_rowparallel_batched( magma_int_t n, magmaFloatComplex** input_array, magma_int_t ldi,
                   magmaFloatComplex** output_array, magma_int_t ldo,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **pivinfo_array, 
                   magma_int_t batchCount );

void 
magma_claswp_rowserial_batched(magma_int_t n, magmaFloatComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount);

void 
magma_claswp_columnserial_batched(magma_int_t n, magmaFloatComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount);

void 
magmablas_ctranspose_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array,  magma_int_t ldda,
    magmaFloatComplex **dAT_array, magma_int_t lddat, magma_int_t batchCount );

void 
magmablas_claset_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount );

void 
magmablas_claset_batched_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_cmemset_batched(magma_int_t length, 
        magmaFloatComplex_ptr dAarray[], magmaFloatComplex val, 
        magma_int_t batchCount);

magma_int_t 
magma_cgetf2_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magmaFloatComplex **GERA_array,
    magmaFloatComplex **GERB_array,
    magmaFloatComplex **GERC_array,
    magma_int_t **ipiv_array,
    magma_int_t *info_array, 
    magma_int_t gbstep,            
    magma_int_t batchCount,
    cublasHandle_t myhandle);

magma_int_t 
magma_cgetrf_recpanel_batched_q(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t** dpivinfo_array,
    magmaFloatComplex** dX_array,    magma_int_t dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dW1_displ, magmaFloatComplex** dW2_displ,  
    magmaFloatComplex** dW3_displ, magmaFloatComplex** dW4_displ,
    magmaFloatComplex** dW5_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t stream, cublasHandle_t myhandle);

magma_int_t 
magma_cgetrf_batched(
        magma_int_t m, magma_int_t n,
        magmaFloatComplex **dA_array, 
        magma_int_t lda,
        magma_int_t **ipiv_array, 
        magma_int_t *info_array, 
        magma_int_t batchCount);

magma_int_t 
magma_cgetri_outofplace_batched( magma_int_t n, 
                  magmaFloatComplex **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  magmaFloatComplex **dinvA_array, magma_int_t lddia,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

void 
magma_cdisplace_intpointers(magma_int_t **output_array,
               magma_int_t **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount);





void 
magmablas_icamax_atomic_batched(magma_int_t n, magmaFloatComplex** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);

void 
magmablas_icamax_tree_batched(magma_int_t n, magmaFloatComplex** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);



void 
magmablas_icamax_batched(magma_int_t n, magmaFloatComplex** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);

void 
magmablas_icamax(magma_int_t n, magmaFloatComplex* x, magma_int_t incx, magma_int_t *max_id);


magma_int_t 
magma_icamax_batched(magma_int_t length, 
        magmaFloatComplex **x_array, magma_int_t incx, magma_int_t step,  magma_int_t lda,
        magma_int_t** ipiv_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount);

magma_int_t 
magma_cswap_batched(magma_int_t n, magmaFloatComplex **x_array, magma_int_t incx, magma_int_t j, 
                 magma_int_t** ipiv_array, magma_int_t batchCount);

magma_int_t 
magma_cscal_cgeru_batched(magma_int_t m, magma_int_t n, magma_int_t step,
                                      magmaFloatComplex **dA_array, magma_int_t lda,
                                      magma_int_t *info_array, magma_int_t gbstep, 
                                      magma_int_t batchCount);

magma_int_t 
magma_ccomputecolumn_batched(magma_int_t m, magma_int_t paneloffset, magma_int_t step, 
                                        magmaFloatComplex **dA_array,  magma_int_t lda,
                                        magma_int_t **ipiv_array, 
                                        magma_int_t *info_array, magma_int_t gbstep, 
                                        magma_int_t batchCount);

void 
magma_cgetf2trsm_batched(magma_int_t ib, magma_int_t n, magmaFloatComplex **dA_array,  magma_int_t j, magma_int_t lda,
                       magma_int_t batchCount);


magma_int_t 
magma_cgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magmaFloatComplex **dW0_displ,
    magmaFloatComplex **dW1_displ,
    magmaFloatComplex **dW2_displ,
    magma_int_t *info_array,            
    magma_int_t gbstep, 
    magma_int_t batchCount,
    cublasHandle_t myhandle);

magma_int_t 
magma_cgetrf_panel_nopiv_batched_q(
    magma_int_t m, magma_int_t nb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dX_array,    magma_int_t dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dW0_displ, magmaFloatComplex** dW1_displ,  
    magmaFloatComplex** dW2_displ, magmaFloatComplex** dW3_displ,
    magmaFloatComplex** dW4_displ,     
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t stream, cublasHandle_t myhandle);

magma_int_t 
magma_cgetrf_recpanel_nopiv_batched_q(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dX_array,    magma_int_t dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dW1_displ, magmaFloatComplex** dW2_displ,  
    magmaFloatComplex** dW3_displ, magmaFloatComplex** dW4_displ,
    magmaFloatComplex** dW5_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t stream, cublasHandle_t myhandle);


magma_int_t 
magma_cgetrf_nopiv_batched(
        magma_int_t m, magma_int_t n,
        magmaFloatComplex **dA_array, 
        magma_int_t lda,
        magma_int_t *info_array, 
        magma_int_t batchCount);

magma_int_t 
magma_cgetrs_nopiv_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  magmaFloatComplex **dA_array, magma_int_t ldda,
                  magmaFloatComplex **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

magma_int_t 
magma_cgesv_nopiv_batched(
                  magma_int_t n, magma_int_t nrhs,
                  magmaFloatComplex **dA_array, magma_int_t ldda,
                  magmaFloatComplex **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

magma_int_t 
magma_cgesv_rbt_batched(
                  magma_int_t n, magma_int_t nrhs,
                  magmaFloatComplex **dA_array, magma_int_t ldda,
                  magmaFloatComplex **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

magma_int_t 
magma_cgesv_batched(
                  magma_int_t n, magma_int_t nrhs,
                  magmaFloatComplex **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  magmaFloatComplex **dB_array, magma_int_t lddb,
                  magma_int_t *dinfo_array,
                  magma_int_t batchCount);

magma_int_t
magma_cgerbt_batched(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magmaFloatComplex *U, magmaFloatComplex *V,
    magma_int_t *info, magma_int_t batchCount);

void 
magmablas_cprbt_batched(
    magma_int_t n, 
    magmaFloatComplex **dA_array, magma_int_t ldda, 
    magmaFloatComplex *du, magmaFloatComplex *dv,
    magma_int_t batchCount);

void
magmablas_cprbt_mv_batched(
    magma_int_t n, 
    magmaFloatComplex *dv, magmaFloatComplex **db_array, magma_int_t batchCount);


void
magmablas_cprbt_mtv_batched(
    magma_int_t n, 
    magmaFloatComplex *du, magmaFloatComplex **db_array, magma_int_t batchCount);





void 
magma_clacgv_batched(magma_int_t n, magmaFloatComplex **x_array, magma_int_t incx, int offset, int batchCount);

void 
magma_cpotf2_csscal_batched(magma_int_t n, magmaFloatComplex **x_array, magma_int_t incx, magma_int_t offset, magma_int_t *info_array, magma_int_t batchCount);

void 
magma_cpotf2_cdotc_batched(magma_int_t n, magmaFloatComplex **x_array, magma_int_t incx, magma_int_t offset, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount);


void 
setup_pivinfo_q( magma_int_t *pivinfo, magma_int_t *ipiv, 
                      magma_int_t m, magma_int_t nb, 
                      magma_queue_t stream);

void 
setup_pivinfo( magma_int_t *pivinfo, magma_int_t *ipiv, 
                    magma_int_t m, magma_int_t nb);


void
magmablas_cgeadd_batched_q(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloatComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_clacpy_batched_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloatComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );
void
magmablas_cgeadd_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloatComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount );

void
magmablas_clacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloatComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount );

void
magmablas_cgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t ldda,
    magmaFloatComplex_ptr dx_array[], magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t incy,
    magma_int_t batchCount);


// for debugging purpose
void 
cset_stepinit_ipiv(magma_int_t **ipiv_array,
                 magma_int_t pm,
                 magma_int_t batchCount);



#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif  /* MAGMA_CBATCHED_H */
