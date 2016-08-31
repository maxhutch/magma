/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from include/magma_zvbatched.h, normal z -> s, Tue Aug 30 09:39:21 2016
*/

#ifndef MAGMA_SVBATCHED_H
#define MAGMA_SVBATCHED_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif
  /*
   *  BLAS batched routines
   */
// Level 1

// Level 2

// Level 3
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

#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMA_SVBATCHED_H */
