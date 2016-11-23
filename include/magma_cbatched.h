/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from include/magma_zbatched.h, normal z -> c, Sun Nov 20 20:20:47 2016
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
magma_cset_pointer(
    magmaFloatComplex **output_array,
    magmaFloatComplex *input,
    magma_int_t lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batch_offset,
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_cdisplace_pointers(
    magmaFloatComplex **output_array,
    magmaFloatComplex **input_array, magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_crecommend_cublas_gemm_batched(
    magma_trans_t transa, magma_trans_t transb, 
    magma_int_t m, magma_int_t n, magma_int_t k);

magma_int_t 
magma_crecommend_cublas_gemm_stream(
    magma_trans_t transa, magma_trans_t transb, 
    magma_int_t m, magma_int_t n, magma_int_t k);

void magma_get_cpotrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb);

magma_int_t magma_get_cpotrf_batched_crossover();

void magma_get_cgetrf_batched_nbparam(magma_int_t n, magma_int_t *nb, magma_int_t *recnb);

magma_int_t magma_get_cgeqrf_batched_nb(magma_int_t m);

void
magmablas_cswapdblk_batched(
    magma_int_t n, magma_int_t nb,
    magmaFloatComplex **dA, magma_int_t ldda, magma_int_t inca,
    magmaFloatComplex **dB, magma_int_t lddb, magma_int_t incb,
    magma_int_t batchCount, magma_queue_t queue );

  /*
   *  LAPACK batched routines
   */

  /*
   *  BLAS batched routines
   */
void
magmablas_cgemm_batched_core(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, 
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
    magma_int_t batchCount, magma_queue_t queue );

void
magma_cgemm_batched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cgemm_batched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_csyrk_internal_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cherk_internal_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_csyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magma_cherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    float beta,
    magmaFloatComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    float beta,
    magmaFloatComplex **dC_array, magma_int_t lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cher2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb, 
    float beta, magmaFloatComplex **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_csyr2k_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t lddb, 
    magmaFloatComplex beta, magmaFloatComplex **dC_array, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t 
magma_cpotf2_tile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cpotrf_panel(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex* dinvA, magmaFloatComplex** dinvA_array, magma_int_t invA_msize,
    magmaFloatComplex* x, magmaFloatComplex** x_array,  magma_int_t x_msize,
    magma_int_t *info_array, 
    magma_int_t batchCount, magma_int_t matrixSize, magma_queue_t queue);

void 
magmablas_ctrtri_diag_batched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex const * const *dA_array, magma_int_t ldda,
    magmaFloatComplex **dinvA_array, 
    magma_int_t resetozero,
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_ctrsm_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dB_array,    magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_ctrsm_work_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    magmaFloatComplex alpha, 
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dB_array,    magma_int_t lddb,
    magmaFloatComplex** dX_array,    magma_int_t lddx, 
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dA_displ, magmaFloatComplex** dB_displ, 
    magmaFloatComplex** dX_displ, magmaFloatComplex** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, magma_queue_t queue);

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
    magma_int_t resetozero,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_ctrsv_batched(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dB_array,    magma_int_t incb,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_ctrsv_work_batched(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dB_array,    magma_int_t incb,
    magmaFloatComplex** dX_array,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_ctrsv_outofplace_batched(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n, 
    magmaFloatComplex ** A_array, magma_int_t lda,
    magmaFloatComplex **b_array, magma_int_t incb, 
    magmaFloatComplex **x_array, 
    magma_int_t batchCount, magma_queue_t queue, magma_int_t flag);

void 
magmablas_ctrmm_batched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t ldda,
        magmaFloatComplex **dB_array, magma_int_t lddb, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_ctrmm_batched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t m, magma_int_t n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t ldda,
        magmaFloatComplex **dB_array, magma_int_t lddb, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_chemm_batched_core(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t m, magma_int_t n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t ldda,
        magmaFloatComplex **dB_array, magma_int_t lddb, 
        magmaFloatComplex beta, 
        magmaFloatComplex **dC_array, magma_int_t lddc, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_chemm_batched(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t m, magma_int_t n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t ldda,
        magmaFloatComplex **dB_array, magma_int_t lddb, 
        magmaFloatComplex beta, 
        magmaFloatComplex **dC_array, magma_int_t lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_chemv_batched_core(
        magma_uplo_t uplo, magma_int_t n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t ldda,
        magmaFloatComplex **dX_array, magma_int_t incx,
        magmaFloatComplex beta, 
        magmaFloatComplex **dY_array, magma_int_t incy,
        magma_int_t offA, magma_int_t offX, magma_int_t offY, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_chemv_batched(
        magma_uplo_t uplo, magma_int_t n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t ldda,
        magmaFloatComplex **dX_array, magma_int_t incx,
        magmaFloatComplex beta, 
        magmaFloatComplex **dY_array, magma_int_t incy,
        magma_int_t batchCount, magma_queue_t queue );

magma_int_t 
magma_cpotrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cpotf2_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magmaFloatComplex **dA_displ, 
    magmaFloatComplex **dW_displ,
    magmaFloatComplex **dB_displ, 
    magmaFloatComplex **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue);

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
    magma_int_t batchCount, magma_queue_t queue);

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
    magma_int_t batchCount, magma_queue_t queue);

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
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cpotrs_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cposv_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magma_int_t *dinfo_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgetrs_batched(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array, 
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_claswp_rowparallel_batched(
    magma_int_t n, magmaFloatComplex** input_array, magma_int_t ldi,
    magmaFloatComplex** output_array, magma_int_t ldo,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **pivinfo_array, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magma_claswp_rowserial_batched(
    magma_int_t n, magmaFloatComplex** dA_array, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **ipiv_array, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_claswp_columnserial_batched(
    magma_int_t n, magmaFloatComplex** dA_array, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    magma_int_t **ipiv_array, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_ctranspose_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array,  magma_int_t ldda,
    magmaFloatComplex **dAT_array, magma_int_t lddat,
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_claset_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue);

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
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgetrf_recpanel_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t** dpivinfo_array,
    magmaFloatComplex** dX_array,    magma_int_t dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dW1_displ, magmaFloatComplex** dW2_displ,  
    magmaFloatComplex** dW3_displ, magmaFloatComplex** dW4_displ,
    magmaFloatComplex** dW5_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgetrf_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, 
    magma_int_t lda,
    magma_int_t **ipiv_array, 
    magma_int_t *info_array, 
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgetri_outofplace_batched(
    magma_int_t n, 
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array, 
    magmaFloatComplex **dinvA_array, magma_int_t lddia,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_cdisplace_intpointers(
    magma_int_t **output_array,
    magma_int_t **input_array, magma_int_t lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_icamax_atomic_batched(
    magma_int_t n,
    magmaFloatComplex** x_array, magma_int_t incx,
    magma_int_t **max_id_array,
    magma_int_t batchCount);

void 
magmablas_icamax_tree_batched(
    magma_int_t n,
    magmaFloatComplex** x_array, magma_int_t incx,
    magma_int_t **max_id_array,
    magma_int_t batchCount);

void 
magmablas_icamax_batched(
    magma_int_t n,
    magmaFloatComplex** x_array, magma_int_t incx,
    magma_int_t **max_id_array,
    magma_int_t batchCount);

void 
magmablas_icamax(
    magma_int_t n,
    magmaFloatComplex* x, magma_int_t incx,
    magma_int_t *max_id);

magma_int_t 
magma_icamax_batched(
    magma_int_t length, 
    magmaFloatComplex **x_array, magma_int_t incx, magma_int_t step,  magma_int_t lda,
    magma_int_t** ipiv_array, magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cswap_batched(
    magma_int_t n, magmaFloatComplex **x_array, magma_int_t incx, magma_int_t j, 
    magma_int_t** ipiv_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cscal_cgeru_batched(
    magma_int_t m, magma_int_t n, magma_int_t step,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_ccomputecolumn_batched(
    magma_int_t m, magma_int_t paneloffset, magma_int_t step, 
    magmaFloatComplex **dA_array,  magma_int_t lda,
    magma_int_t **ipiv_array, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_cgetf2trsm_batched(
    magma_int_t ib, magma_int_t n,
    magmaFloatComplex **dA_array,  magma_int_t j, magma_int_t lda,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda,
    magmaFloatComplex **dW0_displ,
    magmaFloatComplex **dW1_displ,
    magmaFloatComplex **dW2_displ,
    magma_int_t *info_array,            
    magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgetrf_panel_nopiv_batched(
    magma_int_t m, magma_int_t nb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dX_array,    magma_int_t dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dW0_displ, magmaFloatComplex** dW1_displ,  
    magmaFloatComplex** dW2_displ, magmaFloatComplex** dW3_displ,
    magmaFloatComplex** dW4_displ,     
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgetrf_recpanel_nopiv_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** dX_array,    magma_int_t dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t dinvA_length,
    magmaFloatComplex** dW1_displ, magmaFloatComplex** dW2_displ,  
    magmaFloatComplex** dW3_displ, magmaFloatComplex** dW4_displ,
    magmaFloatComplex** dW5_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgetrf_nopiv_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, 
    magma_int_t lda,
    magma_int_t *info_array, 
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgetrs_nopiv_batched(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgesv_nopiv_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgesv_rbt_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgesv_batched(
    magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magma_int_t **dipiv_array, 
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magma_int_t *dinfo_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_cgerbt_batched(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magmaFloatComplex *U, magmaFloatComplex *V,
    magma_int_t *info,
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_cprbt_batched(
    magma_int_t n, 
    magmaFloatComplex **dA_array, magma_int_t ldda, 
    magmaFloatComplex *du, magmaFloatComplex *dv,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_cprbt_mv_batched(
    magma_int_t n, 
    magmaFloatComplex *dv, magmaFloatComplex **db_array,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_cprbt_mtv_batched(
    magma_int_t n, 
    magmaFloatComplex *du, magmaFloatComplex **db_array,
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_clacgv_batched(
    magma_int_t n,
    magmaFloatComplex **x_array, magma_int_t incx, magma_int_t offset,
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_cpotf2_csscal_batched(
    magma_int_t n,
    magmaFloatComplex **x_array, magma_int_t incx, magma_int_t offset,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_cpotf2_cdotc_batched(
    magma_int_t n,
    magmaFloatComplex **x_array, magma_int_t incx, magma_int_t offset,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

void 
setup_pivinfo(
    magma_int_t *pivinfo, magma_int_t *ipiv, 
    magma_int_t m, magma_int_t nb, 
    magma_queue_t queue);

void
magmablas_cgeadd_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloatComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_clacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloatComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cgemv_batched_template(
    magma_trans_t trans, magma_int_t m, magma_int_t n, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t ldda, 
    magmaFloatComplex_ptr dx_array[], magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t incy, 
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_cgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t ldda,
    magmaFloatComplex_ptr dx_array[], magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgeqrf_batched(
    magma_int_t m, magma_int_t n, 
    magmaFloatComplex **dA_array,
    magma_int_t lda, 
    magmaFloatComplex **dtau_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_cgeqrf_expert_batched(
    magma_int_t m, magma_int_t n, 
    magmaFloatComplex **dA_array, magma_int_t ldda, 
    magmaFloatComplex **dR_array, magma_int_t lddr,
    magmaFloatComplex **dT_array, magma_int_t lddt,
    magmaFloatComplex **dtau_array, magma_int_t provide_RT,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgeqrf_batched_v4(
    magma_int_t m, magma_int_t n, 
    magmaFloatComplex **dA_array,
    magma_int_t lda, 
    magmaFloatComplex **tau_array,
    magma_int_t *info_array,
    magma_int_t batchCount);

magma_int_t
magma_cgeqrf_panel_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,    
    magmaFloatComplex** dA_array,    magma_int_t ldda,
    magmaFloatComplex** tau_array, 
    magmaFloatComplex** dT_array, magma_int_t ldt, 
    magmaFloatComplex** dR_array, magma_int_t ldr,
    magmaFloatComplex** dW0_displ, 
    magmaFloatComplex** dW1_displ,
    magmaFloatComplex *dwork,  
    magmaFloatComplex** W_array, 
    magmaFloatComplex** W2_array,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cgels_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex **dA_array, magma_int_t ldda,
    magmaFloatComplex **dB_array, magma_int_t lddb,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magma_int_t *info,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_cgeqr2x_batched_v4(
    magma_int_t m, magma_int_t n, magmaFloatComplex **dA_array,
    magma_int_t lda, 
    magmaFloatComplex **tau_array,
    magmaFloatComplex **dR_array, magma_int_t ldr,
    float **dwork_array,  
    magma_int_t *info,
    magma_int_t batchCount);

magma_int_t
magma_cgeqr2_batched(
    magma_int_t m, magma_int_t n, magmaFloatComplex **dA_array,
    magma_int_t lda, 
    magmaFloatComplex **tau_array,
    magma_int_t *info,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_clarfb_gemm_batched(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV_array[],    magma_int_t lddv,
    magmaFloatComplex_const_ptr dT_array[],    magma_int_t lddt,
    magmaFloatComplex_ptr dC_array[],          magma_int_t lddc,
    magmaFloatComplex_ptr dwork_array[],       magma_int_t ldwork,
    magmaFloatComplex_ptr dworkvt_array[],     magma_int_t ldworkvt,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_clarft_batched_vold(
    magma_int_t n, magma_int_t k, magmaFloatComplex **v_array, magma_int_t ldv,
    magmaFloatComplex **tau_array,
    magmaFloatComplex **T_array, magma_int_t ldt, 
    magma_int_t batchCount);

magma_int_t
magma_clarft_batched(
    magma_int_t n, magma_int_t k, magma_int_t stair_T, 
    magmaFloatComplex **v_array, magma_int_t ldv,
    magmaFloatComplex **tau_array,
    magmaFloatComplex **T_array, magma_int_t ldt, 
    magmaFloatComplex **work_array, magma_int_t lwork,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_clarft_sm32x32_batched(
    magma_int_t n, magma_int_t k,
    magmaFloatComplex **v_array, magma_int_t ldv,
    magmaFloatComplex **tau_array,
    magmaFloatComplex **T_array, magma_int_t ldt, 
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_clarft_recctrmv_sm32x32(
    magma_int_t m, magma_int_t n, 
    magmaFloatComplex *tau, 
    magmaFloatComplex *Trec, magma_int_t ldtrec, 
    magmaFloatComplex *Ttri, magma_int_t ldttri,
    magma_queue_t queue);

void magmablas_clarft_recctrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n, 
    magmaFloatComplex **tau_array, 
    magmaFloatComplex **Trec_array, magma_int_t ldtrec, 
    magmaFloatComplex **Ttri_array, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_clarft_ctrmv_sm32x32(
    magma_int_t m, magma_int_t n, 
    magmaFloatComplex *tau, 
    magmaFloatComplex *Tin, magma_int_t ldtin, 
    magmaFloatComplex *Tout, magma_int_t ldtout,
    magma_queue_t queue);

void magmablas_clarft_ctrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n, 
    magmaFloatComplex **tau_array, 
    magmaFloatComplex **Tin_array, magma_int_t ldtin, 
    magmaFloatComplex **Tout_array, magma_int_t ldtout,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_clarft_gemv_loop_inside(
    magma_int_t n, magma_int_t k, 
    magmaFloatComplex *tau, 
    magmaFloatComplex *v, magma_int_t ldv, 
    magmaFloatComplex *T, magma_int_t ldt,
    magma_queue_t queue);

void magmablas_clarft_gemv_loop_inside_batched(
    magma_int_t n, magma_int_t k, 
    magmaFloatComplex **tau_array, 
    magmaFloatComplex **v_array, magma_int_t ldv, 
    magmaFloatComplex **T_array, magma_int_t ldt,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_clarft_gemvrowwise(
    magma_int_t m, magma_int_t i, 
    magmaFloatComplex *tau, 
    magmaFloatComplex *v, magma_int_t ldv, 
    magmaFloatComplex *T, magma_int_t ldt,
    magmaFloatComplex *W,
    magma_queue_t queue);

void magmablas_clarft_gemvrowwise_batched(
    magma_int_t m, magma_int_t i, 
    magmaFloatComplex **tau_array, 
    magmaFloatComplex **v_array, magma_int_t ldv, 
    magmaFloatComplex **T_array, magma_int_t ldt,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_clarft_gemvcolwise(
    magma_int_t m,  magma_int_t step,
    magmaFloatComplex *v, magma_int_t ldv, 
    magmaFloatComplex *T,  magma_int_t ldt,
    magmaFloatComplex *tau,
    magma_queue_t queue);

void magmablas_clarft_gemvcolwise_batched(
    magma_int_t m,  magma_int_t step,
    magmaFloatComplex **v_array, magma_int_t ldv, 
    magmaFloatComplex **T_array,  magma_int_t ldt,
    magmaFloatComplex **tau_array,
    magma_int_t batchCount, magma_queue_t queue);

void cgeqrf_copy_upper_batched(                
    magma_int_t n, magma_int_t nb,
    magmaFloatComplex **dV_array, magma_int_t ldv,
    magmaFloatComplex **dR_array, magma_int_t ldr,
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_scnrm2_cols_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t lda, 
    float **dxnorm_array,
    magma_int_t batchCount);
 
void 
magma_clarfgx_batched(
    magma_int_t n, magmaFloatComplex **dx0_array, magmaFloatComplex **dx_array, 
    magmaFloatComplex **dtau_array, float **dxnorm_array, 
    magmaFloatComplex **dR_array, magma_int_t it,
    magma_int_t batchCount);

void 
magma_clarfx_batched_v4(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **v_array,
    magmaFloatComplex **tau_array,
    magmaFloatComplex **C_array, magma_int_t ldc, float **xnorm_array, 
    magma_int_t step, 
    magma_int_t batchCount);

void 
magmablas_clarfg_batched(
    magma_int_t n,
    magmaFloatComplex** dalpha_array,
    magmaFloatComplex** dx_array, magma_int_t incx,
    magmaFloatComplex** dtau_array,
    magma_int_t batchCount );

magma_int_t
magma_cpotrf_lpout_batched(
    magma_uplo_t uplo, magma_int_t n, 
    magmaFloatComplex **dA_array, magma_int_t lda, magma_int_t gbstep,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_cpotrf_lpin_batched(
    magma_uplo_t uplo, magma_int_t n, 
    magmaFloatComplex **dA_array, magma_int_t lda, magma_int_t gbstep,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_cpotrf_v33_batched(
    magma_uplo_t uplo, magma_int_t n, 
    magmaFloatComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue);

// for debugging purpose
void 
cset_stepinit_ipiv(
    magma_int_t **ipiv_array,
    magma_int_t pm,
    magma_int_t batchCount);

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif  /* MAGMA_CBATCHED_H */
