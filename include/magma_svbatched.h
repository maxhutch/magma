/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from include/magma_zvbatched.h, normal z -> s, Sun Nov 20 20:20:47 2016
*/

#ifndef MAGMA_SVBATCHED_H
#define MAGMA_SVBATCHED_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  LAPACK vbatched routines
   */
magma_int_t
magma_spotrf_lpout_vbatched(
    magma_uplo_t uplo, magma_int_t *n, magma_int_t max_n,  
    float **dA_array, magma_int_t *lda, magma_int_t gbstep,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotf2_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    float **dA_array, magma_int_t* lda,
    float **dA_displ, 
    float **dW_displ,
    float **dB_displ, 
    float **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_panel_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    magma_int_t *ibvec, magma_int_t nb,  
    float** dA_array,    magma_int_t* ldda,
    float** dX_array,    magma_int_t* dX_length,
    float** dinvA_array, magma_int_t* dinvA_length,
    float** dW0_displ, float** dW1_displ, 
    float** dW2_displ, float** dW3_displ,
    float** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t *n, 
    float **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount, 
    magma_int_t max_n, magma_queue_t queue);

magma_int_t
magma_spotrf_vbatched(
    magma_uplo_t uplo, magma_int_t *n, 
    float **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount, 
    magma_queue_t queue);
  /*
   *  BLAS vbatched routines
   */
/* Level 3 */
void 
magmablas_sgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
    magma_int_t spec_m, magma_int_t spec_n, magma_int_t spec_k, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_vbatched_max(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_vbatched_nocheck(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_sgemm_vbatched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magmablas_ssyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_ssyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, 
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_ssyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, 
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb, 
    float beta, float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb, 
    float beta, float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb, 
    float beta, float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magmablas_ssyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb, 
    float beta, float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void 
magmablas_ssyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb, 
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb, 
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
    
void
magmablas_ssyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb, 
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magmablas_ssyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb, 
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_strmm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        float alpha, 
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, 
        magma_int_t spec_m, magma_int_t spec_n, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_strmm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        float alpha, 
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_strmm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        float alpha, 
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_strmm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        float alpha, 
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strmm_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        float alpha, 
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue );
   
void magmablas_strsm_outofplace_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, 
    magma_int_t *m, magma_int_t* n,
    float alpha, 
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    float** dX_array,    magma_int_t* lddx, 
    float** dinvA_array, magma_int_t* dinvA_length,
    float** dA_displ, float** dB_displ, 
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void magmablas_strsm_work_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, 
    magma_int_t* m, magma_int_t* n, 
    float alpha, 
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    float** dX_array,    magma_int_t* lddx, 
    float** dinvA_array, magma_int_t* dinvA_length,
    float** dA_displ, float** dB_displ, 
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void magmablas_strsm_vbatched_max_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void 
magmablas_strsm_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void
magmablas_strsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_queue_t queue);

void
magmablas_strsm_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_queue_t queue);

void
magmablas_strtri_diag_vbatched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t nmax, magma_int_t *n,
    float const * const *dA_array, magma_int_t *ldda,
    float **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_ssymm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        float alpha, 
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb, 
        float beta, 
        float **dC_array, magma_int_t *lddc, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
        magma_int_t specM, magma_int_t specN, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_ssymm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        float alpha, 
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb, 
        float beta, 
        float **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_ssymm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        float alpha, 
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb, 
        float beta, 
        float **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_ssymm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        float alpha, 
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb, 
        float beta, 
        float **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssymm_vbatched(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        float alpha, 
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb, 
        float beta, 
        float **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue );

/* Level 2 */
void
magmablas_sgemv_vbatched_max_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda, 
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);

void
magmablas_sgemv_vbatched_max(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda, 
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);
    
void
magmablas_sgemv_vbatched_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda, 
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sgemv_vbatched(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda, 
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_ssymv_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t* n, float alpha, 
    float **dA_array, magma_int_t* ldda,
    float **dX_array, magma_int_t* incx,
    float beta,  
    float **dY_array, magma_int_t* incy, 
    magma_int_t max_n, magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssymv_vbatched_max(
    magma_uplo_t uplo, magma_int_t* n, 
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda, 
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_queue_t queue);

void
magmablas_ssymv_vbatched_nocheck(
    magma_uplo_t uplo, magma_int_t* n, 
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda, 
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_ssymv_vbatched(
    magma_uplo_t uplo, magma_int_t* n, 
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda, 
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);
/* Level 1 */
/* Auxiliary routines */
void magma_sset_pointer_var_cc(
    float **output_array,
    float *input,
    magma_int_t *lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t *batch_offset,
    magma_int_t batchCount, 
    magma_queue_t queue);

void 
magma_sdisplace_pointers_var_cc(float **output_array,
    float **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_sdisplace_pointers_var_cv(float **output_array,
    float **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t* column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_sdisplace_pointers_var_vc(float **output_array,
    float **input_array, magma_int_t* lda,
    magma_int_t *row, magma_int_t column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_sdisplace_pointers_var_vv(float **output_array,
    float **input_array, magma_int_t* lda,
    magma_int_t* row, magma_int_t* column, 
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_slaset_vbatched(
    magma_uplo_t uplo, magma_int_t max_m, magma_int_t max_n, 
    magma_int_t* m, magma_int_t* n,
    float offdiag, float diag,
    magmaFloat_ptr dAarray[], magma_int_t* ldda,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_slacpy_vbatched(
    magma_uplo_t uplo, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_int_t* m, magma_int_t* n,
    float const * const * dAarray, magma_int_t* ldda,
    float**               dBarray, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue );

  /*
   *  Aux. vbatched routines
   */    
magma_int_t magma_get_spotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMA_SVBATCHED_H */
