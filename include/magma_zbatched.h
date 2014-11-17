/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/

#ifndef MAGMA_ZBATCHED_H
#define MAGMA_ZBATCHED_H

#include "magma_types.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  local auxiliary routines
   */
void 
zset_pointer_int(magma_int_t **output_array,
        magma_int_t *input,
        magma_int_t lda,
        magma_int_t row, magma_int_t column, 
        magma_int_t batchSize,
        magma_int_t batchCount);

void 
zset_pointer(magmaDoubleComplex **output_array,
                 magmaDoubleComplex *input,
                 magma_int_t lda,
                 magma_int_t row, magma_int_t column,
                 magma_int_t batchSize,
                 magma_int_t batchCount);


void 
zset_array(magmaDoubleComplex **output_array,
               magmaDoubleComplex **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column,
               magma_int_t batchCount);

void 
magma_zdisplace_pointers(magmaDoubleComplex **output_array,
               magmaDoubleComplex **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount);

  /*
   *  LAPACK batched routines
   */

  /*
   *  BLAS batched routines
   */

void
magmablas_zgemm_batched(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );

void
magmablas_zgemm_batched_lg(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );
void
magmablas_zgemm_batched_k32(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dB_array, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );

void 
magmablas_zgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dx_array, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dy_array, magma_int_t incy,
    magma_int_t batchCount);

void 
magmablas_zherk_NC_batched( magma_trans_t TRANSA, magma_trans_t TRANSB, int m , int n , int k , 
                       magmaDoubleComplex alpha, magmaDoubleComplex **dA_array, int lda, 
                       magmaDoubleComplex **B_array, int ldb, 
                       magmaDoubleComplex beta,        magmaDoubleComplex **C_array, int ldc, 
                       magma_int_t batchCount);

void
magmablas_zherk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    double beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );

void
magmablas_zherk_batched_lg(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    double beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );

void
magmablas_zherk_batched_k32(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    double beta,
    magmaDoubleComplex **dC_array, magma_int_t lddc, magma_int_t batchCount );






magma_int_t 
magma_zpotf2_tile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount);

magma_int_t 
magma_zpotrf_panel(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex* dinvA, magmaDoubleComplex** dinvA_array, magma_int_t invA_msize,
    magmaDoubleComplex* x, magmaDoubleComplex** x_array,  magma_int_t x_msize,
    magma_int_t *info_array,  magma_int_t batchCount, magma_int_t matrixSize);


void 
magmablas_ztrtri_diag_batched_q(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex const * const *dA_array, magma_int_t ldda,
    magmaDoubleComplex **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_ztrtri_diag_batched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex const * const *dA_array, magma_int_t ldda,
    magmaDoubleComplex **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount);


void 
magmablas_ztrsm_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t lddb,
    magma_int_t batchCount);


void 
magmablas_ztrsm_work_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha, 
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t lddb,
    magmaDoubleComplex** dX_array,    magma_int_t lddx, 
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ, 
    magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount);

void 
magmablas_ztrsm_outofplace_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha, 
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dB_array,    magma_int_t lddb,
    magmaDoubleComplex** dX_array,    magma_int_t lddx, 
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dA_displ, magmaDoubleComplex** dB_displ, 
    magmaDoubleComplex** dX_displ, magmaDoubleComplex** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount);

magma_int_t 
magma_zpotrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    magma_int_t *info_array,  magma_int_t batchCount);


magma_int_t 
magma_zpotf2_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    magmaDoubleComplex **dA_displ, 
    magmaDoubleComplex **dW_displ,
    magmaDoubleComplex **dB_displ, 
    magmaDoubleComplex **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_zpotrf_panel_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ, 
    magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ,
    magmaDoubleComplex** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_zpotrf_recpanel_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ,  
    magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ,
    magmaDoubleComplex** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_zpotrf_rectile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ,  
    magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ,
    magmaDoubleComplex** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t 
magma_zpotrs_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t batchCount);

magma_int_t 
magma_zposv_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t *dinfo_array,
                  magma_int_t batchCount);

magma_int_t 
magma_zgetrs_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t batchCount);



void 
magma_zlaswp_rowparallel_batched( magma_int_t n, magmaDoubleComplex** input_array, magma_int_t ldi,
                   magmaDoubleComplex** output_array, magma_int_t ldo,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **pivinfo_array, 
                   magma_int_t batchCount );

void 
magma_zlaswp_rowserial_batched(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount);

void 
magma_zlaswp_columnserial_batched(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount);

void 
magmablas_ztranspose_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array,  magma_int_t ldda,
    magmaDoubleComplex **dAT_array, magma_int_t lddat, magma_int_t batchCount );

void 
magmablas_zlaset_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount );

void 
magmablas_zlaset_batched_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_zmemset_batched(magma_int_t length, 
        magmaDoubleComplex_ptr dAarray[], magmaDoubleComplex val, 
        magma_int_t batchCount);

magma_int_t 
magma_zgetf2_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    magmaDoubleComplex **GERA_array,
    magmaDoubleComplex **GERB_array,
    magmaDoubleComplex **GERC_array,
    magma_int_t **ipiv_array,
    magma_int_t *info_array, 
    magma_int_t gbstep,            
    magma_int_t batchCount,
    cublasHandle_t myhandle);

magma_int_t 
magma_zgetrf_recpanel_batched_q(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t** dpivinfo_array,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW1_displ, magmaDoubleComplex** dW2_displ,  
    magmaDoubleComplex** dW3_displ, magmaDoubleComplex** dW4_displ,
    magmaDoubleComplex** dW5_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t stream, cublasHandle_t myhandle);

magma_int_t 
magma_zgetrf_batched(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex **dA_array, 
        magma_int_t lda,
        magma_int_t **ipiv_array, 
        magma_int_t *info_array, 
        magma_int_t batchCount);

magma_int_t 
magma_zgetri_outofplace_batched( magma_int_t n, 
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  magmaDoubleComplex **dinvA_array, magma_int_t lddia,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

void 
magma_zdisplace_intpointers(magma_int_t **output_array,
               magma_int_t **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount);





void 
magmablas_izamax_atomic_batched(magma_int_t n, magmaDoubleComplex** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);

void 
magmablas_izamax_tree_batched(magma_int_t n, magmaDoubleComplex** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);



void 
magmablas_izamax_batched(magma_int_t n, magmaDoubleComplex** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);

void 
magmablas_izamax(magma_int_t n, magmaDoubleComplex* x, magma_int_t incx, magma_int_t *max_id);


magma_int_t 
magma_izamax_batched(magma_int_t length, 
        magmaDoubleComplex **x_array, magma_int_t incx, magma_int_t step,  magma_int_t lda,
        magma_int_t** ipiv_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount);

magma_int_t 
magma_zswap_batched(magma_int_t n, magmaDoubleComplex **x_array, magma_int_t incx, magma_int_t j, 
                 magma_int_t** ipiv_array, magma_int_t batchCount);

magma_int_t 
magma_zscal_zgeru_batched(magma_int_t m, magma_int_t n, magma_int_t step,
                                      magmaDoubleComplex **dA_array, magma_int_t lda,
                                      magma_int_t *info_array, magma_int_t gbstep, 
                                      magma_int_t batchCount);

magma_int_t 
magma_zcomputecolumn_batched(magma_int_t m, magma_int_t paneloffset, magma_int_t step, 
                                        magmaDoubleComplex **dA_array,  magma_int_t lda,
                                        magma_int_t **ipiv_array, 
                                        magma_int_t *info_array, magma_int_t gbstep, 
                                        magma_int_t batchCount);

void 
magma_zgetf2trsm_batched(magma_int_t ib, magma_int_t n, magmaDoubleComplex **dA_array,  magma_int_t j, magma_int_t lda,
                       magma_int_t batchCount);


magma_int_t 
magma_zgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t lda,
    magmaDoubleComplex **dW0_displ,
    magmaDoubleComplex **dW1_displ,
    magmaDoubleComplex **dW2_displ,
    magma_int_t *info_array,            
    magma_int_t gbstep, 
    magma_int_t batchCount,
    cublasHandle_t myhandle);

magma_int_t 
magma_zgetrf_panel_nopiv_batched_q(
    magma_int_t m, magma_int_t nb,    
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW0_displ, magmaDoubleComplex** dW1_displ,  
    magmaDoubleComplex** dW2_displ, magmaDoubleComplex** dW3_displ,
    magmaDoubleComplex** dW4_displ,     
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t stream, cublasHandle_t myhandle);

magma_int_t 
magma_zgetrf_recpanel_nopiv_batched_q(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    magmaDoubleComplex** dA_array,    magma_int_t ldda,
    magmaDoubleComplex** dX_array,    magma_int_t dX_length,
    magmaDoubleComplex** dinvA_array, magma_int_t dinvA_length,
    magmaDoubleComplex** dW1_displ, magmaDoubleComplex** dW2_displ,  
    magmaDoubleComplex** dW3_displ, magmaDoubleComplex** dW4_displ,
    magmaDoubleComplex** dW5_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t stream, cublasHandle_t myhandle);


magma_int_t 
magma_zgetrf_nopiv_batched(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex **dA_array, 
        magma_int_t lda,
        magma_int_t *info_array, 
        magma_int_t batchCount);

magma_int_t 
magma_zgetrs_nopiv_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

magma_int_t 
magma_zgesv_nopiv_batched(
                  magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

magma_int_t 
magma_zgesv_rbt_batched(
                  magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount);

magma_int_t 
magma_zgesv_batched(
                  magma_int_t n, magma_int_t nrhs,
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  magmaDoubleComplex **dB_array, magma_int_t lddb,
                  magma_int_t *dinfo_array,
                  magma_int_t batchCount);

magma_int_t
magma_zgerbt_batched(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb,
    magmaDoubleComplex *U, magmaDoubleComplex *V,
    magma_int_t *info, magma_int_t batchCount);

void 
magmablas_zprbt_batched(
    magma_int_t n, 
    magmaDoubleComplex **dA_array, magma_int_t ldda, 
    magmaDoubleComplex *du, magmaDoubleComplex *dv,
    magma_int_t batchCount);

void
magmablas_zprbt_mv_batched(
    magma_int_t n, 
    magmaDoubleComplex *dv, magmaDoubleComplex **db_array, magma_int_t batchCount);


void
magmablas_zprbt_mtv_batched(
    magma_int_t n, 
    magmaDoubleComplex *du, magmaDoubleComplex **db_array, magma_int_t batchCount);





void 
magma_zlacgv_batched(magma_int_t n, magmaDoubleComplex **x_array, magma_int_t incx, int offset, int batchCount);

void 
magma_zpotf2_zdscal_batched(magma_int_t n, magmaDoubleComplex **x_array, magma_int_t incx, magma_int_t offset, magma_int_t *info_array, magma_int_t batchCount);

void 
magma_zpotf2_zdotc_batched(magma_int_t n, magmaDoubleComplex **x_array, magma_int_t incx, magma_int_t offset, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount);


void 
setup_pivinfo_q( magma_int_t *pivinfo, magma_int_t *ipiv, 
                      magma_int_t m, magma_int_t nb, 
                      magma_queue_t stream);

void 
setup_pivinfo( magma_int_t *pivinfo, magma_int_t *ipiv, 
                    magma_int_t m, magma_int_t nb);


void
magmablas_zgeadd_batched_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDoubleComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_zlacpy_batched_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDoubleComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );
void
magmablas_zgeadd_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDoubleComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount );

void
magmablas_zlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr  const dAarray[], magma_int_t ldda,
    magmaDoubleComplex_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount );

void
magmablas_zgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t ldda,
    magmaDoubleComplex_ptr dx_array[], magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t incy,
    magma_int_t batchCount);


// for debugging purpose
void 
zset_stepinit_ipiv(magma_int_t **ipiv_array,
                 magma_int_t pm,
                 magma_int_t batchCount);



#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif  /* MAGMA_ZBATCHED_H */
