/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from include/magmablas_z_q.h normal z -> s, Mon May  2 23:31:25 2016
*/

#ifndef MAGMABLAS_S_Q_H
#define MAGMABLAS_S_Q_H
                    
#include "magma_types.h"
#include "magma_copy_q.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
magmablas_stranspose_inplace_q(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_stranspose_inplace_q(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_stranspose_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat,
    magma_queue_t queue );

void
magmablas_stranspose_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat,
    magma_queue_t queue );

void
magmablas_sgetmatrix_transpose_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dAT,   magma_int_t ldda,
    float          *hA,    magma_int_t lda,
    magmaFloat_ptr       dwork, magma_int_t lddwork, magma_int_t nb,
    magma_queue_t queues[2] );

void
magmablas_ssetmatrix_transpose_q(
    magma_int_t m, magma_int_t n,
    const float *hA,    magma_int_t lda,
    magmaFloat_ptr    dAT,   magma_int_t ldda,
    magmaFloat_ptr    dwork, magma_int_t lddwork, magma_int_t nb,
    magma_queue_t queues[2] );

  /*
   * RBT-related functions
   */
void
magmablas_sprbt_q(
    magma_int_t n, 
    magmaFloat_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr du,
    magmaFloat_ptr dv,
    magma_queue_t queue );

void
magmablas_sprbt_mv_q(
    magma_int_t n, 
    magmaFloat_ptr dv,
    magmaFloat_ptr db,
    magma_queue_t queue );

void
magmablas_sprbt_mtv_q(
    magma_int_t n, 
    magmaFloat_ptr du,
    magmaFloat_ptr db,
    magma_queue_t queue );

  /*
   * Multi-GPU copy functions
   */
void
magma_sgetmatrix_1D_col_bcyclic_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr const dA[], magma_int_t ldda,
    float                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queue[ MagmaMaxGPUs ] );

void
magma_ssetmatrix_1D_col_bcyclic_q(
    magma_int_t m, magma_int_t n,
    const float *hA,   magma_int_t lda,
    magmaFloat_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queue[ MagmaMaxGPUs ] );

void
magma_sgetmatrix_1D_row_bcyclic_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr const dA[], magma_int_t ldda,
    float                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queue[ MagmaMaxGPUs ] );

void
magma_ssetmatrix_1D_row_bcyclic_q(
    magma_int_t m, magma_int_t n,
    const float *hA,   magma_int_t lda,
    magmaFloat_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queue[ MagmaMaxGPUs ] );

void
magmablas_sgetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    magmaFloat_const_ptr const dAT[],    magma_int_t ldda,
    float                *hA,       magma_int_t lda,
    magmaFloat_ptr             dwork[],  magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void
magmablas_ssetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    const float *hA,      magma_int_t lda,
    magmaFloat_ptr    dAT[],   magma_int_t ldda,
    magmaFloat_ptr    dwork[], magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb );

// in src/ssytrd_mgpu.cpp
// TODO rename ssetmatrix_sy or similar
magma_int_t
magma_shtodhe(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    float     *A,   magma_int_t lda,
    magmaFloat_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][10],
    magma_int_t *info );

// in src/spotrf3_mgpu.cpp
// TODO same as magma_shtodhe?
magma_int_t
magma_shtodpo(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    float     *A,   magma_int_t lda,
    magmaFloat_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info );

// in src/spotrf3_mgpu.cpp
// TODO rename sgetmatrix_sy or similar
magma_int_t
magma_sdtohpo(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
    float     *A,   magma_int_t lda,
    magmaFloat_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info );


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */
void
magmablas_ssymm_mgpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloat_ptr dB[],    magma_int_t lddb,
    float beta,
    magmaFloat_ptr dC[],    magma_int_t lddc,
    magmaFloat_ptr dwork[], magma_int_t dworksiz,
    //float    *C,       magma_int_t ldc,
    //float    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t events[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t ncmplx );

magma_int_t
magmablas_ssymv_mgpu(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    magmaFloat_const_ptr dx,           magma_int_t incx,
    float beta,             
    magmaFloat_ptr    dy,              magma_int_t incy,
    float       *hwork,           magma_int_t lhwork,
    magmaFloat_ptr    dwork[],         magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] );

magma_int_t
magmablas_ssymv_mgpu_sync(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    magmaFloat_const_ptr dx,           magma_int_t incx,
    float beta,             
    magmaFloat_ptr    dy,              magma_int_t incy,
    float       *hwork,           magma_int_t lhwork,
    magmaFloat_ptr    dwork[],         magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] );

// Ichi's version, in src/ssytrd_mgpu.cpp
void
magma_ssyr2k_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloat_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );

void
magmablas_ssyr2k_mgpu2(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_ptr dA[], magma_int_t ldda, magma_int_t a_offset,
    magmaFloat_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloat_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue );

// in src/spotrf_mgpu_right.cpp
void
magma_ssyrk_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloat_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );

// in src/spotrf_mgpu_right.cpp
void
magma_ssyrk_mgpu2(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloat_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
magmablas_sgeadd_q(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_sgeadd2_q(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_slacpy_q(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_slacpy_conj_q(
    magma_int_t n,
    magmaFloat_ptr dA1, magma_int_t lda1,
    magmaFloat_ptr dA2, magma_int_t lda2,
    magma_queue_t queue );

void
magmablas_slacpy_sym_in_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_slacpy_sym_out_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

float
magmablas_slange_q(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

float
magmablas_slansy_q(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

float
magmablas_slansy_q(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

void
magmablas_slarfg_q(
    magma_int_t n,
    magmaFloat_ptr dalpha,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dtau,
    magma_queue_t queue );

void
magmablas_slascl_q(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slascl_2x2_q(
    magma_type_t type, magma_int_t m,
    magmaFloat_const_ptr dW, magma_int_t lddw,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slascl2_q(
    magma_type_t type,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slascl_diag_q(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD, magma_int_t lddd,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slaset_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_slaset_band_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_slaswp_q(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_slaswp2_q(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_slaswp_sym_q(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_slaswpx_q(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_ssymmetrize_q(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_ssymmetrize_tiles_q(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride,
    magma_queue_t queue );

void
magmablas_strtri_diag_q(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr d_dinvA,
    magma_queue_t queue );

  /*
   * to cleanup (alphabetical order)
   */
magma_int_t
magma_slarfb_gpu_q(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV, magma_int_t lddv,
    magmaFloat_const_ptr dT, magma_int_t lddt,
    magmaFloat_ptr dC,       magma_int_t lddc,
    magmaFloat_ptr dwork,    magma_int_t ldwork,
    magma_queue_t queue );

magma_int_t
magma_slarfb_gpu_gemm_q(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV, magma_int_t lddv,
    magmaFloat_const_ptr dT, magma_int_t lddt,
    magmaFloat_ptr dC,       magma_int_t lddc,
    magmaFloat_ptr dwork,    magma_int_t ldwork,
    magmaFloat_ptr dworkvt,  magma_int_t ldworkvt,
    magma_queue_t queue );

void
magma_slarfbx_gpu_q(
    magma_int_t m, magma_int_t k,
    magmaFloat_ptr V,  magma_int_t ldv,
    magmaFloat_ptr dT, magma_int_t ldt,
    magmaFloat_ptr c,
    magmaFloat_ptr dwork,
    magma_queue_t queue );

void
magma_slarfg_gpu_q(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dAkk,
    magma_queue_t queue );

void
magma_slarfgtx_gpu_q(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dA, magma_int_t iter,
    magmaFloat_ptr V,  magma_int_t ldv,
    magmaFloat_ptr T,  magma_int_t ldt,
    magmaFloat_ptr dwork,
    magma_queue_t queue );

void
magma_slarfgx_gpu_q(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dA, magma_int_t iter,
    magma_queue_t queue );

void
magma_slarfx_gpu_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr v,
    magmaFloat_ptr tau,
    magmaFloat_ptr C, magma_int_t ldc,
    magmaFloat_ptr        xnorm,
    magmaFloat_ptr dT, magma_int_t iter,
    magmaFloat_ptr work,
    magma_queue_t queue );

  /*
   * Level 1 BLAS (alphabetical order)
   */
void
magmablas_saxpycp_q(
    magma_int_t m,
    magmaFloat_ptr dr,
    magmaFloat_ptr dx,
    magmaFloat_const_ptr db,
    magma_queue_t queue );

void
magmablas_sswap_q(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magma_queue_t queue );

void
magmablas_sswapblk_q(
    magma_order_t order,
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset,
    magma_queue_t queue );

void
magmablas_sswapdblk_q(
    magma_int_t n, magma_int_t nb,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloat_ptr dB, magma_int_t lddb, magma_int_t incb,
    magma_queue_t queue );

void
magmablas_snrm2_adjust_q(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dc,
    magma_queue_t queue );

void
magmablas_snrm2_check_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc,
    magma_queue_t queue );

void
magmablas_snrm2_check_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc,
    magma_queue_t queue );

void
magmablas_snrm2_cols_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magma_queue_t queue );

void
magmablas_snrm2_row_check_adjust_q(
    magma_int_t k, float tol,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dxnorm2,
    magmaFloat_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dlsticc,
    magma_queue_t queue );

  /*
   * Level 2 BLAS (alphabetical order)
   */
// trsv were always queue versions
void
magmablas_strsv(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       db, magma_int_t incb,
    magma_queue_t queue );

// todo: move flag before queue?
void
magmablas_strsv_outofplace(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr db,       magma_int_t incb,
    magmaFloat_ptr dx,
    magma_queue_t queue,
    magma_int_t flag );

void
magmablas_strsv_work_batched(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_ptr *dA_array, magma_int_t lda,
    magmaFloat_ptr *db_array, magma_int_t incb,
    magmaFloat_ptr *dx_array,
    magma_int_t batchCount,
    magma_queue_t queue );

void
magmablas_sgemv_q(
    magma_trans_t trans, magma_int_t m, magma_int_t n, 
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy, 
    magma_queue_t queue );

void
magmablas_sgemv_conj_q(
    magma_int_t m, magma_int_t n, float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy,
    magma_queue_t queue );

magma_int_t
magmablas_ssymv_q(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

magma_int_t
magmablas_ssymv_q(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

// hemv/symv_work were always queue versions
magma_int_t
magmablas_ssymv_work(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magmaFloat_ptr       dwork, magma_int_t lwork,
    magma_queue_t queue );

magma_int_t
magmablas_ssymv_work(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magmaFloat_ptr       dwork, magma_int_t lwork,
    magma_queue_t queue );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
magmablas_sgemm_q(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_sgemm_reduce_q(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssymm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssymm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssyr2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssyr2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float  beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssyrk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssyrk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float  alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float  beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_strsm_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_strsm_outofplace_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magmaFloat_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloat_ptr d_dinvA, magma_int_t dinvA_length,
    magma_queue_t queue );

void
magmablas_strsm_work_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magmaFloat_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloat_ptr d_dinvA, magma_int_t dinvA_length,
    magma_queue_t queue );


  /*
   * Wrappers for platform independence.
   * These wrap CUBLAS or AMD OpenCL BLAS functions.
   */

// ========================================
// copying vectors
// set  copies host   to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_ssetvector_q(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_ssetvector_q_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_sgetvector_q(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_sgetvector_q_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_scopyvector_q(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_scopyvector_q_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_ssetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_ssetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_sgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_sgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_scopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_scopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_ssetvector_q_internal(
    magma_int_t n,
    float const    *hx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setvector_q_internal( n, sizeof(float), hx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void
magma_sgetvector_q_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    float          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getvector_q_internal( n, sizeof(float), dx_src, incx, hy_dst, incy, queue, func, file, line ); }

static inline void
magma_scopyvector_q_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copyvector_q_internal( n, sizeof(float), dx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void
magma_ssetvector_async_internal(
    magma_int_t n,
    float const    *hx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setvector_async_internal( n, sizeof(float), hx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void
magma_sgetvector_async_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    float          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getvector_async_internal( n, sizeof(float), dx_src, incx, hy_dst, incy, queue, func, file, line ); }

static inline void
magma_scopyvector_async_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copyvector_async_internal( n, sizeof(float), dx_src, incx, dy_dst, incy, queue, func, file, line ); }


// ========================================
// copying sub-matrices (contiguous columns)

#define magma_ssetmatrix_q(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_ssetmatrix_q_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_sgetmatrix_q(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_sgetmatrix_q_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_scopymatrix_q(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_scopymatrix_q_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_ssetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        magma_ssetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_sgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        magma_sgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

#define magma_scopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_scopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_ssetmatrix_q_internal(
    magma_int_t m, magma_int_t n,
    float const    *hA_src, magma_int_t lda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setmatrix_q_internal( m, n, sizeof(float), hA_src, lda, dB_dst, lddb, queue, func, file, line ); }

static inline void
magma_sgetmatrix_q_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    float          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getmatrix_q_internal( m, n, sizeof(float), dA_src, ldda, hB_dst, ldb, queue, func, file, line ); }

static inline void
magma_scopymatrix_q_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copymatrix_q_internal( m, n, sizeof(float), dA_src, ldda, dB_dst, lddb, queue, func, file, line ); }

static inline void
magma_ssetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    float const    *hA_src, magma_int_t lda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setmatrix_async_internal( m, n, sizeof(float), hA_src, lda, dB_dst, lddb, queue, func, file, line ); }

static inline void
magma_sgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    float          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getmatrix_async_internal( m, n, sizeof(float), dA_src, ldda, hB_dst, ldb, queue, func, file, line ); }

static inline void
magma_scopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copymatrix_async_internal( m, n, sizeof(float), dA_src, ldda, dB_dst, lddb, queue, func, file, line ); }


// ========================================
// Level 1 BLAS (alphabetical order)

// in cublas_v2, result returned through output argument
magma_int_t
magma_isamax_q(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
magma_int_t
magma_isamin_q(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
float
magma_sasum_q(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_saxpy_q(
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_scopy_q(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
float
magma_sdot_q(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
float
magma_sdot_q(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
float
magma_snrm2_q(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_srot_q(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    float dc, float ds,
    magma_queue_t queue );

void
magma_srot_q(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    float dc, float ds,
    magma_queue_t queue );

#ifdef REAL
void
magma_srotm_q(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magmaFloat_const_ptr param,
    magma_queue_t queue );

void
magma_srotmg_q(
    magmaFloat_ptr d1, magmaFloat_ptr       d2,
    magmaFloat_ptr x1, magmaFloat_const_ptr y1,
    magmaFloat_ptr param,
    magma_queue_t queue );
#endif

void
magma_sscal_q(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_sscal_q(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_sswap_q(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magma_queue_t queue );

// ========================================
// Level 2 BLAS (alphabetical order)

void
magma_sgemv_q(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_sger_q(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_sger_q(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_ssymv_q(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_ssyr_q(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_ssyr2_q(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_strmv_q(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_strsv_q(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dx, magma_int_t incx,
    magma_queue_t queue );

// ========================================
// Level 3 BLAS (alphabetical order)

void
magma_sgemm_q(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssymm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssymm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssyr2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssyr2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssyrk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssyrk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_strmm_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magma_strsm_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );


#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMABLAS_S_H */
