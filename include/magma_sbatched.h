/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Azzam Haidar
       @author Tingxing Dong

       @generated from magma_zbatched.h normal z -> s, Fri Jan 30 19:00:06 2015
*/

#ifndef MAGMA_SBATCHED_H
#define MAGMA_SBATCHED_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  local auxiliary routines
   */
void 
sset_pointer_int(magma_int_t **output_array,
        magma_int_t *input,
        magma_int_t lda,
        magma_int_t row, magma_int_t column, 
        magma_int_t batchSize,
        magma_int_t batchCount, magma_queue_t queue);

void 
sset_pointer(float **output_array,
                 float *input,
                 magma_int_t lda,
                 magma_int_t row, magma_int_t column,
                 magma_int_t batchSize,
                 magma_int_t batchCount, magma_queue_t queue);


void 
sset_array(float **output_array,
               float **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column,
               magma_int_t batchCount, magma_queue_t queue);

void 
magma_sdisplace_pointers(float **output_array,
               float **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue);

  /*
   *  LAPACK batched routines
   */

  /*
   *  BLAS batched routines
   */

void
magmablas_sgemm_batched(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float const * const * dB_array, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_batched_lg(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float const * const * dB_array, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );
void
magmablas_sgemm_batched_k32(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float const * const * dB_array, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );



void 
magmablas_ssyrk_NC_batched( magma_trans_t TRANSA, magma_trans_t TRANSB, int m , int n , int k , 
                       float alpha, float **dA_array, int lda, 
                       float **B_array, int ldb, 
                       float beta,        float **C_array, int ldc, 
                       magma_int_t batchCount);

void
magmablas_ssyrk_batched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float beta,
    float **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_batched_lg(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float beta,
    float **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_batched_k32(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    float const * const * dA_array, magma_int_t ldda,
    float beta,
    float **dC_array, magma_int_t lddc, magma_int_t batchCount, magma_queue_t queue );






magma_int_t 
magma_spotf2_tile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t lda,
    magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_spotrf_panel(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    float *A, magma_int_t lda,
    float* dinvA, float** dinvA_array, magma_int_t invA_msize,
    float* x, float** x_array,  magma_int_t x_msize,
    magma_int_t *info_array,  magma_int_t batchCount, magma_int_t matrixSize, magma_queue_t queue);


void 
magmablas_strtri_diag_batched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    float const * const *dA_array, magma_int_t ldda,
    float **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);



void 
magmablas_strsm_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue);


void 
magmablas_strsm_work_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    float alpha, 
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t lddb,
    float** dX_array,    magma_int_t lddx, 
    float** dinvA_array, magma_int_t dinvA_length,
    float** dA_displ, float** dB_displ, 
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_strsm_outofplace_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    float alpha, 
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t lddb,
    float** dX_array,    magma_int_t lddx, 
    float** dinvA_array, magma_int_t dinvA_length,
    float** dA_displ, float** dB_displ, 
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_spotrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    float **dA_array, magma_int_t lda,
    magma_int_t *info_array,  magma_int_t batchCount, magma_queue_t queue);


magma_int_t 
magma_spotf2_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t lda,
    float **dA_displ, 
    float **dW_displ,
    float **dB_displ, 
    float **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);

magma_int_t 
magma_spotrf_panel_batched(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,     
    float** dA_array,    magma_int_t ldda,
    float** dX_array,    magma_int_t dX_length,
    float** dinvA_array, magma_int_t dinvA_length,
    float** dW0_displ, float** dW1_displ, 
    float** dW2_displ, float** dW3_displ,
    float** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);

magma_int_t 
magma_spotrf_recpanel_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    float** dA_array,    magma_int_t ldda,
    float** dX_array,    magma_int_t dX_length,
    float** dinvA_array, magma_int_t dinvA_length,
    float** dW0_displ, float** dW1_displ,  
    float** dW2_displ, float** dW3_displ,
    float** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);

magma_int_t 
magma_spotrf_rectile_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, 
    magma_int_t min_recpnb,    
    float** dA_array,    magma_int_t ldda,
    float** dX_array,    magma_int_t dX_length,
    float** dinvA_array, magma_int_t dinvA_length,
    float** dW0_displ, float** dW1_displ,  
    float** dW2_displ, float** dW3_displ,
    float** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);

magma_int_t 
magma_spotrs_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  float **dA_array, magma_int_t ldda,
                  float **dB_array, magma_int_t lddb,
                  magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sposv_batched(
                  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                  float **dA_array, magma_int_t ldda,
                  float **dB_array, magma_int_t lddb,
                  magma_int_t *dinfo_array,
                  magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sgetrs_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  float **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  float **dB_array, magma_int_t lddb,
                  magma_int_t batchCount, magma_queue_t queue);



void 
magma_slaswp_rowparallel_batched( magma_int_t n, float** input_array, magma_int_t ldi,
                   float** output_array, magma_int_t ldo,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **pivinfo_array, 
                   magma_int_t batchCount, magma_queue_t queue );

void 
magma_slaswp_rowserial_batched(magma_int_t n, float** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount, magma_queue_t queue);

void 
magma_slaswp_columnserial_batched(magma_int_t n, float** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array, 
                   magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_stranspose_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array,  magma_int_t ldda,
    float **dAT_array, magma_int_t lddat, magma_int_t batchCount );


void 
magmablas_slaset_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    magmaFloat_ptr dAarray[], magma_int_t ldda,
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_smemset_batched(magma_int_t length, 
        magmaFloat_ptr dAarray[], float val, 
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sgetf2_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t lda,
    float **GERA_array,
    float **GERB_array,
    float **GERC_array,
    magma_int_t **ipiv_array,
    magma_int_t *info_array, 
    magma_int_t gbstep,            
    magma_int_t batchCount,
    cublasHandle_t myhandle, magma_queue_t queue);

magma_int_t 
magma_sgetrf_recpanel_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    float** dA_array,    magma_int_t ldda,
    magma_int_t** dipiv_array, magma_int_t** dpivinfo_array,
    float** dX_array,    magma_int_t dX_length,
    float** dinvA_array, magma_int_t dinvA_length,
    float** dW1_displ, float** dW2_displ,  
    float** dW3_displ, float** dW4_displ,
    float** dW5_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);

magma_int_t 
magma_sgetrf_batched(
        magma_int_t m, magma_int_t n,
        float **dA_array, 
        magma_int_t lda,
        magma_int_t **ipiv_array, 
        magma_int_t *info_array, 
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sgetri_outofplace_batched( magma_int_t n, 
                  float **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  float **dinvA_array, magma_int_t lddia,
                  magma_int_t *info_array,
                  magma_int_t batchCount, magma_queue_t queue);

void 
magma_sdisplace_intpointers(magma_int_t **output_array,
               magma_int_t **input_array, magma_int_t lda,
               magma_int_t row, magma_int_t column, 
               magma_int_t batchCount, magma_queue_t queue);





void 
magmablas_isamax_atomic_batched(magma_int_t n, float** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);

void 
magmablas_isamax_tree_batched(magma_int_t n, float** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);



void 
magmablas_isamax_batched(magma_int_t n, float** x_array, magma_int_t incx, magma_int_t **max_id_array, magma_int_t batchCount);

void 
magmablas_isamax(magma_int_t n, float* x, magma_int_t incx, magma_int_t *max_id);


magma_int_t 
magma_isamax_batched(magma_int_t length, 
        float **x_array, magma_int_t incx, magma_int_t step,  magma_int_t lda,
        magma_int_t** ipiv_array, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sswap_batched(magma_int_t n, float **x_array, magma_int_t incx, magma_int_t j, 
                 magma_int_t** ipiv_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sscal_sger_batched(magma_int_t m, magma_int_t n, magma_int_t step,
                                      float **dA_array, magma_int_t lda,
                                      magma_int_t *info_array, magma_int_t gbstep, 
                                      magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_scomputecolumn_batched(magma_int_t m, magma_int_t paneloffset, magma_int_t step, 
                                        float **dA_array,  magma_int_t lda,
                                        magma_int_t **ipiv_array, 
                                        magma_int_t *info_array, magma_int_t gbstep, 
                                        magma_int_t batchCount, magma_queue_t queue);

void 
magma_sgetf2trsm_batched(magma_int_t ib, magma_int_t n, float **dA_array,  magma_int_t j, magma_int_t lda,
                       magma_int_t batchCount, magma_queue_t queue);


magma_int_t 
magma_sgetf2_nopiv_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t lda,
    float **dW0_displ,
    float **dW1_displ,
    float **dW2_displ,
    magma_int_t *info_array,            
    magma_int_t gbstep, 
    magma_int_t batchCount,
    cublasHandle_t myhandle, magma_queue_t queue);

magma_int_t 
magma_sgetrf_panel_nopiv_batched(
    magma_int_t m, magma_int_t nb,    
    float** dA_array,    magma_int_t ldda,
    float** dX_array,    magma_int_t dX_length,
    float** dinvA_array, magma_int_t dinvA_length,
    float** dW0_displ, float** dW1_displ,  
    float** dW2_displ, float** dW3_displ,
    float** dW4_displ,     
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);

magma_int_t 
magma_sgetrf_recpanel_nopiv_batched(
    magma_int_t m, magma_int_t n, magma_int_t min_recpnb,    
    float** dA_array,    magma_int_t ldda,
    float** dX_array,    magma_int_t dX_length,
    float** dinvA_array, magma_int_t dinvA_length,
    float** dW1_displ, float** dW2_displ,  
    float** dW3_displ, float** dW4_displ,
    float** dW5_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);


magma_int_t 
magma_sgetrf_nopiv_batched(
        magma_int_t m, magma_int_t n,
        float **dA_array, 
        magma_int_t lda,
        magma_int_t *info_array, 
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sgetrs_nopiv_batched(
                  magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                  float **dA_array, magma_int_t ldda,
                  float **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sgesv_nopiv_batched(
                  magma_int_t n, magma_int_t nrhs,
                  float **dA_array, magma_int_t ldda,
                  float **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sgesv_rbt_batched(
                  magma_int_t n, magma_int_t nrhs,
                  float **dA_array, magma_int_t ldda,
                  float **dB_array, magma_int_t lddb,
                  magma_int_t *info_array,
                  magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sgesv_batched(
                  magma_int_t n, magma_int_t nrhs,
                  float **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  float **dB_array, magma_int_t lddb,
                  magma_int_t *dinfo_array,
                  magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgerbt_batched(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs,
    float **dA_array, magma_int_t ldda,
    float **dB_array, magma_int_t lddb,
    float *U, float *V,
    magma_int_t *info, magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_sprbt_batched(
    magma_int_t n, 
    float **dA_array, magma_int_t ldda, 
    float *du, float *dv,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sprbt_mv_batched(
    magma_int_t n, 
    float *dv, float **db_array, magma_int_t batchCount, magma_queue_t queue);


void
magmablas_sprbt_mtv_batched(
    magma_int_t n, 
    float *du, float **db_array, magma_int_t batchCount, magma_queue_t queue);





void 
magma_slacgv_batched(magma_int_t n, float **x_array, magma_int_t incx, int offset, int batchCount, magma_queue_t queue);

void 
magma_spotf2_sscal_batched(magma_int_t n, float **x_array, magma_int_t incx, magma_int_t offset, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

void 
magma_spotf2_sdot_batched(magma_int_t n, float **x_array, magma_int_t incx, magma_int_t offset, magma_int_t *info_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);


void 
setup_pivinfo( magma_int_t *pivinfo, magma_int_t *ipiv, 
                      magma_int_t m, magma_int_t nb, 
                      magma_queue_t queue);


void
magmablas_sgeadd_batched_q(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloat_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_slacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloat_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgeadd_batched(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr  const dAarray[], magma_int_t ldda,
    magmaFloat_ptr              dBarray[], magma_int_t lddb,
    magma_int_t batchCount );


void
magmablas_sgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t ldda,
    magmaFloat_ptr dx_array[], magma_int_t incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue);


magma_int_t 
magma_sgeqrf_batched(
    magma_int_t m, magma_int_t n, 
    float **dA_array,
    magma_int_t lda, 
    float **tau_array,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t 
magma_sgeqrf_batched_v4(
    magma_int_t m, magma_int_t n, 
    float **dA_array,
    magma_int_t lda, 
    float **tau_array,
    magma_int_t *info_array, magma_int_t batchCount);

magma_int_t
magma_sgeqrf_panel_batched(
    magma_int_t m, magma_int_t n, magma_int_t nb,    
    float** dA_array,    magma_int_t ldda,
    float** tau_array, 
    float** dT_array, magma_int_t ldt, 
    float** dR_array, magma_int_t ldr,
    float** dW0_displ, 
    float** dW1_displ,
    float *dwork,  
    float** W_array, 
    float** W2_array,
    magma_int_t *info_array,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);

magma_int_t
magma_sgeqrf_panel_batched_v4(
    magma_int_t m, magma_int_t n, magma_int_t nb,    
    float** dA_array,    magma_int_t ldda,
    float** tau_array, 
    float** dT_array, magma_int_t ldt, 
    float** dR_array, magma_int_t ldr,
    float** dnorm_array,  
    float** dW0_displ, 
    float** dW1_displ,
    float *dwork,  
    float** W_array, 
    float** W2_array,
    magma_int_t *info_array,
    magma_int_t batchCount, cublasHandle_t myhandle);

magma_int_t
magma_sgeqr2x_batched_v4(magma_int_t m, magma_int_t n, float **dA_array,
                  magma_int_t lda, 
                  float **tau_array,
                  float **dR_array, magma_int_t ldr,
                  float **dwork_array,  
                  magma_int_t *info, magma_int_t batchCount);

magma_int_t
magma_sgeqr2_batched(magma_int_t m, magma_int_t n, float **dA_array,
                  magma_int_t lda, 
                  float **tau_array,
                  magma_int_t *info, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_slarfb_sgemm_batched(
                  cublasHandle_t myhandle,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  float **dV_array,    magma_int_t ldv,
                  float **dT_array,    magma_int_t ldt,
                  float **dA_array,    magma_int_t lda,
                  float **W_array,     magma_int_t ldw,
                  float **W2_array,    magma_int_t ldw2,
                  magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_slarfb_gemm_batched(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV_array[],    magma_int_t lddv,
    magmaFloat_const_ptr dT_array[],    magma_int_t lddt,
    magmaFloat_ptr dC_array[],          magma_int_t lddc,
    magmaFloat_ptr dwork_array[],       magma_int_t ldwork,
    magmaFloat_ptr dworkvt_array[],     magma_int_t ldworkvt,
    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);




void
magma_slarft_batched_vold(magma_int_t n, magma_int_t k, float **v_array, magma_int_t ldv,
                    float **tau_array,
                    float **T_array, magma_int_t ldt, 
                    magma_int_t batchCount);





magma_int_t
magma_slarft_batched(magma_int_t n, magma_int_t k, magma_int_t stair_T, 
                float **v_array, magma_int_t ldv,
                float **tau_array, float **T_array, magma_int_t ldt, 
                float **work_array, magma_int_t lwork, magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);

void
magma_slarft_sm32x32_batched(magma_int_t n, magma_int_t k, float **v_array, magma_int_t ldv,
                    float **tau_array, float **T_array, magma_int_t ldt, 
                    magma_int_t batchCount, cublasHandle_t myhandle, magma_queue_t queue);



void magmablas_slarft_recstrmv_sm32x32(
    magma_int_t m, magma_int_t n, 
    float *tau, 
    float *Trec, magma_int_t ldtrec, 
    float *Ttri, magma_int_t ldttri);


void magmablas_slarft_recstrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n, 
    float **tau_array, 
    float **Trec_array, magma_int_t ldtrec, 
    float **Ttri_array, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_slarft_strmv_sm32x32(
    magma_int_t m, magma_int_t n, 
    float *tau, 
    float *Tin, magma_int_t ldtin, 
    float *Tout, magma_int_t ldtout);

void magmablas_slarft_strmv_sm32x32_batched(
    magma_int_t m, magma_int_t n, 
    float **tau_array, 
    float **Tin_array, magma_int_t ldtin, 
    float **Tout_array, magma_int_t ldtout,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_slarft_gemv_loop_inside(
    int n, int k, 
    float *tau, 
    float *v, int ldv, 
    float *T, int ldt);

void magmablas_slarft_gemv_loop_inside_batched(
    int n, int k, 
    float **tau_array, 
    float **v_array, int ldv, 
    float **T_array, int ldt, magma_int_t batchCount, magma_queue_t queue);

void magmablas_slarft_gemvrowwise(
    magma_int_t m, magma_int_t i, 
    float *tau, 
    float *v, magma_int_t ldv, 
    float *T, magma_int_t ldt,
    float *W);

void magmablas_slarft_gemvrowwise_batched(
    magma_int_t m, magma_int_t i, 
    float **tau_array, 
    float **v_array, magma_int_t ldv, 
    float **T_array, magma_int_t ldt,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_slarft_gemvcolwise(
    magma_int_t m,  magma_int_t step,
    float *v, magma_int_t ldv, 
    float *T,  magma_int_t ldt,
    float *tau);

void magmablas_slarft_gemvcolwise_batched(
    magma_int_t m,  magma_int_t step,
    float **v_array, magma_int_t ldv, 
    float **T_array,  magma_int_t ldt,
    float **tau_array, magma_int_t batchCount, magma_queue_t queue);




void sgeqrf_copy_upper_batched(                
                  magma_int_t n, magma_int_t nb,
                  float **dV_array,    magma_int_t ldv,
                  float **dR_array,    magma_int_t ldr,
          magma_int_t batchCount, magma_queue_t queue);



void 
magmablas_snrm2_cols_batched(
    magma_int_t m, magma_int_t n,
    float **dA_array, magma_int_t lda, 
    float **dxnorm_array, magma_int_t batchCount);
 
void 
magma_slarfgx_batched(magma_int_t n, float **dx0_array, float **dx_array, 
                  float **dtau_array, float **dxnorm_array, 
                  float **dR_array, magma_int_t it, magma_int_t batchCount);


void 
magma_slarfx_batched_v4(magma_int_t m, magma_int_t n, float **v_array, float **tau_array,
                float **C_array, magma_int_t ldc, float **xnorm_array, 
                magma_int_t step, 
                magma_int_t batchCount);


void 
magmablas_slarfg_batched(
    magma_int_t n,
    float** dalpha_array, float** dx_array, magma_int_t incx,
    float** dtau_array, magma_int_t batchCount );





// for debugging purpose
void 
sset_stepinit_ipiv(magma_int_t **ipiv_array,
                 magma_int_t pm,
                 magma_int_t batchCount);



#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMA_SBATCHED_H */
