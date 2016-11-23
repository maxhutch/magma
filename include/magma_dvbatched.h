/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from include/magma_zvbatched.h, normal z -> d, Sun Nov 20 20:20:47 2016
*/

#ifndef MAGMA_DVBATCHED_H
#define MAGMA_DVBATCHED_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  LAPACK vbatched routines
   */
magma_int_t
magma_dpotrf_lpout_vbatched(
    magma_uplo_t uplo, magma_int_t *n, magma_int_t max_n,  
    double **dA_array, magma_int_t *lda, magma_int_t gbstep,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dpotf2_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    double **dA_array, magma_int_t* lda,
    double **dA_displ, 
    double **dW_displ,
    double **dB_displ, 
    double **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dpotrf_panel_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    magma_int_t *ibvec, magma_int_t nb,  
    double** dA_array,    magma_int_t* ldda,
    double** dX_array,    magma_int_t* dX_length,
    double** dinvA_array, magma_int_t* dinvA_length,
    double** dW0_displ, double** dW1_displ, 
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ, 
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dpotrf_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t *n, 
    double **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount, 
    magma_int_t max_n, magma_queue_t queue);

magma_int_t
magma_dpotrf_vbatched(
    magma_uplo_t uplo, magma_int_t *n, 
    double **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount, 
    magma_queue_t queue);
  /*
   *  BLAS vbatched routines
   */
/* Level 3 */
void 
magmablas_dgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
    magma_int_t spec_m, magma_int_t spec_n, magma_int_t spec_k, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dgemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dgemm_vbatched_max(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dgemm_vbatched_nocheck(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_dgemm_vbatched(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magmablas_dsyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t max_n, magma_int_t max_k, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_dsyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, 
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_dsyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, 
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans, 
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb, 
    double beta, double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb, 
    double beta, double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb, 
    double beta, double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magmablas_dsyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb, 
    double beta, double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void 
magmablas_dsyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb, 
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb, 
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );
    
void
magmablas_dsyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb, 
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );
    
void
magmablas_dsyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb, 
    double beta,
    double **dC_array, magma_int_t* lddc, 
    magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_dtrmm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        double alpha, 
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, 
        magma_int_t spec_m, magma_int_t spec_n, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_dtrmm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        double alpha, 
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_dtrmm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        double alpha, 
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_dtrmm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        double alpha, 
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrmm_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
        magma_int_t* m, magma_int_t* n, 
        double alpha, 
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb, 
        magma_int_t batchCount, magma_queue_t queue );
   
void magmablas_dtrsm_outofplace_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, 
    magma_int_t *m, magma_int_t* n,
    double alpha, 
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    double** dX_array,    magma_int_t* lddx, 
    double** dinvA_array, magma_int_t* dinvA_length,
    double** dA_displ, double** dB_displ, 
    double** dX_displ, double** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void magmablas_dtrsm_work_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, 
    magma_int_t* m, magma_int_t* n, 
    double alpha, 
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    double** dX_array,    magma_int_t* lddx, 
    double** dinvA_array, magma_int_t* dinvA_length,
    double** dA_displ, double** dB_displ, 
    double** dX_displ, double** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void magmablas_dtrsm_vbatched_max_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void 
magmablas_dtrsm_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue);

void
magmablas_dtrsm_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_queue_t queue);

void
magmablas_dtrsm_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_queue_t queue);

void
magmablas_dtrtri_diag_vbatched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t nmax, magma_int_t *n,
    double const * const *dA_array, magma_int_t *ldda,
    double **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_dsymm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        double alpha, 
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb, 
        double beta, 
        double **dC_array, magma_int_t *lddc, 
        magma_int_t max_m, magma_int_t max_n, 
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
        magma_int_t specM, magma_int_t specN, 
        magma_int_t batchCount, magma_queue_t queue );

void 
magmablas_dsymm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        double alpha, 
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb, 
        double beta, 
        double **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_dsymm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        double alpha, 
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb, 
        double beta, 
        double **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n, 
        magma_queue_t queue );

void
magmablas_dsymm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        double alpha, 
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb, 
        double beta, 
        double **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsymm_vbatched(
        magma_side_t side, magma_uplo_t uplo, 
        magma_int_t *m, magma_int_t *n, 
        double alpha, 
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb, 
        double beta, 
        double **dC_array, magma_int_t *lddc, 
        magma_int_t batchCount, magma_queue_t queue );

/* Level 2 */
void
magmablas_dgemv_vbatched_max_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);

void
magmablas_dgemv_vbatched_max(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);
    
void
magmablas_dgemv_vbatched_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dgemv_vbatched(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magmablas_dsymv_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t* n, double alpha, 
    double **dA_array, magma_int_t* ldda,
    double **dX_array, magma_int_t* incx,
    double beta,  
    double **dY_array, magma_int_t* incy, 
    magma_int_t max_n, magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsymv_vbatched_max(
    magma_uplo_t uplo, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_n, magma_queue_t queue);

void
magmablas_dsymv_vbatched_nocheck(
    magma_uplo_t uplo, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dsymv_vbatched(
    magma_uplo_t uplo, magma_int_t* n, 
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda, 
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, magma_queue_t queue);
/* Level 1 */
/* Auxiliary routines */
void magma_dset_pointer_var_cc(
    double **output_array,
    double *input,
    magma_int_t *lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t *batch_offset,
    magma_int_t batchCount, 
    magma_queue_t queue);

void 
magma_ddisplace_pointers_var_cc(double **output_array,
    double **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_ddisplace_pointers_var_cv(double **output_array,
    double **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t* column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_ddisplace_pointers_var_vc(double **output_array,
    double **input_array, magma_int_t* lda,
    magma_int_t *row, magma_int_t column, 
    magma_int_t batchCount, magma_queue_t queue);

void 
magma_ddisplace_pointers_var_vv(double **output_array,
    double **input_array, magma_int_t* lda,
    magma_int_t* row, magma_int_t* column, 
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_dlaset_vbatched(
    magma_uplo_t uplo, magma_int_t max_m, magma_int_t max_n, 
    magma_int_t* m, magma_int_t* n,
    double offdiag, double diag,
    magmaDouble_ptr dAarray[], magma_int_t* ldda,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dlacpy_vbatched(
    magma_uplo_t uplo, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_int_t* m, magma_int_t* n,
    double const * const * dAarray, magma_int_t* ldda,
    double**               dBarray, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue );

  /*
   *  Aux. vbatched routines
   */    
magma_int_t magma_get_dpotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMA_DVBATCHED_H */
