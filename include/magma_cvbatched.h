/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from include/magma_zvbatched.h, normal z -> c, Sun Nov 20 20:20:47 2016
*/

#ifndef MAGMA_CVBATCHED_H
#define MAGMA_CVBATCHED_H

#include "magma_types.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  LAPACK vbatched routines
   */
magma_int_t
magma_cpotrf_lpout_vbatched(
    magma_uplo_t uplo, magma_int_t *n, magma_int_t max_n,  
    magmaFloatComplex **dA_array, magma_int_t *lda, magma_int_t gbstep,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_cpotf2_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    magmaFloatComplex **dA_array, magma_int_t* lda,
    magmaFloatComplex **dA_displ, 
    magmaFloatComplex **dW_displ,
    magmaFloatComplex **dB_displ, 
    magmaFloatComplex **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_cpotrf_panel_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    magma_int_t *ibvec, magma_int_t nb,  
    magmaFloatComplex** dA_array,    magma_int_t* ldda,
    magmaFloatComplex** dX_array,    magma_int_t* dX_length,
    magmaFloatComplex** dinvA_array, magma_int_t* dinvA_length,
    magmaFloatComplex** dW0_displ, magmaFloatComplex** dW1_displ, 
    magmaFloatComplex** dW2_displ, magmaFloatComplex** dW3_displ,
    magmaFloatComplex** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_cpotrf_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t *n, 
    magmaFloatComplex **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount, 
    magma_int_t max_n, magma_queue_t queue);

magma_int_t
magma_cpotrf_vbatched(
    magma_uplo_t uplo, magma_int_t *n, 
    magmaFloatComplex **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount, 
    magma_queue_t queue);
  /*
   *  BLAS vbatched routines
   */
/* Level 3 */
void 
magmablas_cgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
    magma_int_t spec_m, magma_int_t spec_n, magma_int_t spec_k, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cgemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cgemm_vbatched_max(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cgemm_vbatched_nocheck(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_cgemm_vbatched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cherk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magmablas_csyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_cherk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    float alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    float beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_cherk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        float alpha,
        magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
        float beta,
        magmaFloatComplex **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, 
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_cherk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        float alpha,
        magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
        float beta,
        magmaFloatComplex **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cherk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        float alpha,
        magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
        float beta,
        magmaFloatComplex **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_csyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_csyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        magmaFloatComplex alpha,
        magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
        magmaFloatComplex beta,
        magmaFloatComplex **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, 
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_csyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        magmaFloatComplex alpha,
        magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
        magmaFloatComplex beta,
        magmaFloatComplex **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_csyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        magmaFloatComplex alpha,
        magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
        magmaFloatComplex beta,
        magmaFloatComplex **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_cher2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb, 
    float beta, magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_cher2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb, 
    float beta, magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_cher2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb, 
    float beta, magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magmablas_cher2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb, 
    float beta, magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void 
magmablas_csyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb, 
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_csyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb, 
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
    
void
magmablas_csyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb, 
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magmablas_csyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    magmaFloatComplex alpha,
    magmaFloatComplex const * const * dA_array, magma_int_t* ldda,
    magmaFloatComplex const * const * dB_array, magma_int_t* lddb, 
    magmaFloatComplex beta,
    magmaFloatComplex **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_ctrmm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t* ldda,
        magmaFloatComplex **dB_array, magma_int_t* lddb, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, 
        magma_int_t spec_m, magma_int_t spec_n, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_ctrmm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t* ldda,
        magmaFloatComplex **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_ctrmm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t* ldda,
        magmaFloatComplex **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_ctrmm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t* ldda,
        magmaFloatComplex **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ctrmm_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t* ldda,
        magmaFloatComplex **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue );
   
void magmablas_ctrsm_outofplace_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, 
    magma_int_t *m, magma_int_t* n,
    magmaFloatComplex alpha, 
    magmaFloatComplex** dA_array,    magma_int_t* ldda,
    magmaFloatComplex** dB_array,    magma_int_t* lddb,
    magmaFloatComplex** dX_array,    magma_int_t* lddx, 
    magmaFloatComplex** dinvA_array, magma_int_t* dinvA_length,
    magmaFloatComplex** dA_displ, magmaFloatComplex** dB_displ, 
    magmaFloatComplex** dX_displ, magmaFloatComplex** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void magmablas_ctrsm_work_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, 
    magma_int_t* m, magma_int_t* n, 
    magmaFloatComplex alpha, 
    magmaFloatComplex** dA_array,    magma_int_t* ldda,
    magmaFloatComplex** dB_array,    magma_int_t* lddb,
    magmaFloatComplex** dX_array,    magma_int_t* lddx, 
    magmaFloatComplex** dinvA_array, magma_int_t* dinvA_length,
    magmaFloatComplex** dA_displ, magmaFloatComplex** dB_displ, 
    magmaFloatComplex** dX_displ, magmaFloatComplex** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void magmablas_ctrsm_vbatched_max_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    magmaFloatComplex alpha,
    magmaFloatComplex** dA_array,    magma_int_t* ldda,
    magmaFloatComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void 
magmablas_ctrsm_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    magmaFloatComplex alpha,
    magmaFloatComplex** dA_array,    magma_int_t* ldda,
    magmaFloatComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void
magmablas_ctrsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    magmaFloatComplex alpha,
    magmaFloatComplex** dA_array,    magma_int_t* ldda,
    magmaFloatComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_queue_t queue);

void
magmablas_ctrsm_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    magmaFloatComplex alpha,
    magmaFloatComplex** dA_array,    magma_int_t* ldda,
    magmaFloatComplex** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_queue_t queue);

void
magmablas_ctrtri_diag_vbatched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t nmax, magma_int_t *n,
    magmaFloatComplex const * const *dA_array, magma_int_t *ldda,
    magmaFloatComplex **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_chemm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t *ldda,
        magmaFloatComplex **dB_array, magma_int_t *lddb, 
        magmaFloatComplex beta, 
        magmaFloatComplex **dC_array, magma_int_t *lddc, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
        magma_int_t specM, magma_int_t specN, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_chemm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t *ldda,
        magmaFloatComplex **dB_array, magma_int_t *lddb, 
        magmaFloatComplex beta, 
        magmaFloatComplex **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_chemm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t *ldda,
        magmaFloatComplex **dB_array, magma_int_t *lddb, 
        magmaFloatComplex beta, 
        magmaFloatComplex **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_chemm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t *ldda,
        magmaFloatComplex **dB_array, magma_int_t *lddb, 
        magmaFloatComplex beta, 
        magmaFloatComplex **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_chemm_vbatched(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        magmaFloatComplex alpha, 
        magmaFloatComplex **dA_array, magma_int_t *ldda,
        magmaFloatComplex **dB_array, magma_int_t *lddb, 
        magmaFloatComplex beta, 
        magmaFloatComplex **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue );

/* Level 2 */
void
magmablas_cgemv_vbatched_max_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaFloatComplex_ptr dx_array[], magma_int_t* incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);

void
magmablas_cgemv_vbatched_max(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaFloatComplex_ptr dx_array[], magma_int_t* incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);
    
void
magmablas_cgemv_vbatched_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaFloatComplex_ptr dx_array[], magma_int_t* incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_cgemv_vbatched(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaFloatComplex_ptr dx_array[], magma_int_t* incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_chemv_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t* n, magmaFloatComplex alpha, 
    magmaFloatComplex **dA_array, magma_int_t* ldda,
    magmaFloatComplex **dX_array, magma_int_t* incx,
    magmaFloatComplex beta,  
    magmaFloatComplex **dY_array, magma_int_t* incy, 
    magma_int_t max_n, magma_int_t batchCount, magma_queue_t queue );

void
magmablas_chemv_vbatched_max(
    magma_uplo_t uplo, magma_int_t* n, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaFloatComplex_ptr dx_array[], magma_int_t* incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_queue_t queue);

void
magmablas_chemv_vbatched_nocheck(
    magma_uplo_t uplo, magma_int_t* n, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaFloatComplex_ptr dx_array[], magma_int_t* incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_chemv_vbatched(
    magma_uplo_t uplo, magma_int_t* n, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaFloatComplex_ptr dx_array[], magma_int_t* incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);
/* Level 1 */
/* Auxiliary routines */
void magma_cset_pointer_var_cc(
    magmaFloatComplex **output_array,
    magmaFloatComplex *input,
    magma_int_t *lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t *batch_offset,
    magma_int_t batchCount, 
    magma_queue_t queue);

void 
magma_cdisplace_pointers_var_cc(magmaFloatComplex **output_array,
    magmaFloatComplex **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_cdisplace_pointers_var_cv(magmaFloatComplex **output_array,
    magmaFloatComplex **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t* column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_cdisplace_pointers_var_vc(magmaFloatComplex **output_array,
    magmaFloatComplex **input_array, magma_int_t* lda,
    magma_int_t *row, magma_int_t column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_cdisplace_pointers_var_vv(magmaFloatComplex **output_array,
    magmaFloatComplex **input_array, magma_int_t* lda,
    magma_int_t* row, magma_int_t* column, 
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_claset_vbatched(
    magma_uplo_t uplo, magma_int_t max_m, magma_int_t max_n, 
    magma_int_t* m, magma_int_t* n,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dAarray[], magma_int_t* ldda,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_clacpy_vbatched(
    magma_uplo_t uplo, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_int_t* m, magma_int_t* n,
    magmaFloatComplex const * const * dAarray, magma_int_t* ldda,
    magmaFloatComplex**               dBarray, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue );

  /*
   *  Aux. vbatched routines
   */    
magma_int_t magma_get_cpotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif  /* MAGMA_CVBATCHED_H */
