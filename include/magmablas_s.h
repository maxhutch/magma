/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magmablas_z.h, normal z -> s, Sun Nov 20 20:20:47 2016
*/

#ifndef MAGMABLAS_S_H
#define MAGMABLAS_S_H

#include "magma_types.h"
#include "magma_copy.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
magmablas_stranspose_inplace(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_stranspose_inplace(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_stranspose(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat,
    magma_queue_t queue );

void
magmablas_stranspose(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat,
    magma_queue_t queue );

void
magmablas_sgetmatrix_transpose(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloat_const_ptr dAT,   magma_int_t ldda,
    float          *hA,    magma_int_t lda,
    magmaFloat_ptr       dwork, magma_int_t lddw,
    magma_queue_t queues[2] );

void
magmablas_ssetmatrix_transpose(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const float *hA,    magma_int_t lda,
    magmaFloat_ptr    dAT,   magma_int_t ldda,
    magmaFloat_ptr    dwork, magma_int_t lddw,
    magma_queue_t queues[2] );

  /*
   * RBT-related functions
   */
void
magmablas_sprbt(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr du,
    magmaFloat_ptr dv,
    magma_queue_t queue );

void
magmablas_sprbt_mv(
    magma_int_t n,
    magmaFloat_ptr dv,
    magmaFloat_ptr db,
    magma_queue_t queue );

void
magmablas_sprbt_mtv(
    magma_int_t n,
    magmaFloat_ptr du,
    magmaFloat_ptr db,
    magma_queue_t queue );

  /*
   * Multi-GPU copy functions
   */
void
magma_sgetmatrix_1D_col_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloat_const_ptr const dA[], magma_int_t ldda,
    float                *hA,   magma_int_t lda,
    magma_queue_t queue[] );

void
magma_ssetmatrix_1D_col_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const float *hA,   magma_int_t lda,
    magmaFloat_ptr    dA[], magma_int_t ldda,
    magma_queue_t queue[] );

void
magma_sgetmatrix_1D_row_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloat_const_ptr const dA[], magma_int_t ldda,
    float                *hA,   magma_int_t lda,
    magma_queue_t queue[] );

void
magma_ssetmatrix_1D_row_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const float *hA,   magma_int_t lda,
    magmaFloat_ptr    dA[], magma_int_t ldda,
    magma_queue_t queue[] );

void
magmablas_sgetmatrix_transpose_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloat_const_ptr const dAT[],    magma_int_t ldda,
    float                *hA,       magma_int_t lda,
    magmaFloat_ptr             dwork[],  magma_int_t lddw,
    magma_queue_t queues[][2] );

void
magmablas_ssetmatrix_transpose_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const float *hA,      magma_int_t lda,
    magmaFloat_ptr    dAT[],   magma_int_t ldda,
    magmaFloat_ptr    dwork[], magma_int_t lddw,
    magma_queue_t queues[][2] );

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
magmablas_sgeadd(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_sgeadd2(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_slacpy(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_slacpy_conj(
    magma_int_t n,
    magmaFloat_ptr dA1, magma_int_t lda1,
    magmaFloat_ptr dA2, magma_int_t lda2,
    magma_queue_t queue );

void
magmablas_slacpy_sym_in(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_slacpy_sym_out(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

float
magmablas_slange(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

float
magmablas_slansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

float
magmablas_slansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

void
magmablas_slarfg(
    magma_int_t n,
    magmaFloat_ptr dalpha,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dtau,
    magma_queue_t queue );

void
magmablas_slascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slascl_2x2(
    magma_type_t type, magma_int_t m,
    magmaFloat_const_ptr dW, magma_int_t lddw,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slascl2(
    magma_type_t type,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD, magma_int_t lddd,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_slaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_slaset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_slaswp(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_slaswp2(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_slaswp_sym(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_slaswpx(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_ssymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_ssymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride,
    magma_queue_t queue );

void
magmablas_strtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr d_dinvA,
    magma_queue_t queue );

  /*
   * to cleanup (alphabetical order)
   */
magma_int_t
magma_slarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV, magma_int_t lddv,
    magmaFloat_const_ptr dT, magma_int_t lddt,
    magmaFloat_ptr dC,       magma_int_t lddc,
    magmaFloat_ptr dwork,    magma_int_t ldwork,
    magma_queue_t queue );

magma_int_t
magma_slarfb_gpu_gemm(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV, magma_int_t lddv,
    magmaFloat_const_ptr dT, magma_int_t lddt,
    magmaFloat_ptr dC,       magma_int_t lddc,
    magmaFloat_ptr dwork,    magma_int_t ldwork,
    magmaFloat_ptr dworkvt,  magma_int_t ldworkvt,
    magma_queue_t queue );

void
magma_slarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaFloat_ptr V,  magma_int_t ldv,
    magmaFloat_ptr dT, magma_int_t ldt,
    magmaFloat_ptr c,
    magmaFloat_ptr dwork,
    magma_queue_t queue );

void
magma_slarfg_gpu(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dAkk,
    magma_queue_t queue );

void
magma_slarfgtx_gpu(
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
magma_slarfgx_gpu(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dA, magma_int_t iter,
    magma_queue_t queue );

void
magma_slarfx_gpu(
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
magmablas_saxpycp(
    magma_int_t m,
    magmaFloat_ptr dr,
    magmaFloat_ptr dx,
    magmaFloat_const_ptr db,
    magma_queue_t queue );

void
magmablas_sswap(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magma_queue_t queue );

void
magmablas_sswapblk(
    magma_order_t order,
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset,
    magma_queue_t queue );

void
magmablas_sswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloat_ptr dB, magma_int_t lddb, magma_int_t incb,
    magma_queue_t queue );

void
magmablas_snrm2_adjust(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dc,
    magma_queue_t queue );

void
magmablas_snrm2_check(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc,
    magma_queue_t queue );

void
magmablas_snrm2_check(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc,
    magma_queue_t queue );

void
magmablas_snrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magma_queue_t queue );

void
magmablas_snrm2_row_check_adjust(
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
magmablas_sgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy,
    magma_queue_t queue );

void
magmablas_sgemv_conj(
    magma_int_t m, magma_int_t n, float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy,
    magma_queue_t queue );

magma_int_t
magmablas_ssymv(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

magma_int_t
magmablas_ssymv(
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
magmablas_sgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_sgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float  beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float  alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float  beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_strsm_outofplace(
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
magmablas_strsm_work(
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

// =============================================================================
// copying vectors
// set  copies host   to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

/// Type-safe version of magma_setvector() for float arrays.
/// @ingroup magma_setvector
#define magma_ssetvector(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_ssetvector_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector() for float arrays.
/// @ingroup magma_getvector
#define magma_sgetvector(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_sgetvector_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector() for float arrays.
/// @ingroup magma_copyvector
#define magma_scopyvector(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_scopyvector_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setvector_async() for float arrays.
/// @ingroup magma_setvector
#define magma_ssetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_ssetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector_async() for float arrays.
/// @ingroup magma_getvector
#define magma_sgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_sgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector_async() for float arrays.
/// @ingroup magma_copyvector
#define magma_scopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_scopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_ssetvector_internal(
    magma_int_t n,
    float const    *hx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_internal( n, sizeof(float),
                              hx_src, incx,
                              dy_dst, incy, queue,
                              func, file, line );
}

static inline void
magma_sgetvector_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    float          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_internal( n, sizeof(float),
                              dx_src, incx,
                              hy_dst, incy, queue,
                              func, file, line );
}

static inline void
magma_scopyvector_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_internal( n, sizeof(float),
                               dx_src, incx,
                               dy_dst, incy, queue,
                               func, file, line );
}

static inline void
magma_ssetvector_async_internal(
    magma_int_t n,
    float const    *hx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_async_internal( n, sizeof(float),
                                    hx_src, incx,
                                    dy_dst, incy, queue,
                                    func, file, line );
}

static inline void
magma_sgetvector_async_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    float          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_async_internal( n, sizeof(float),
                                    dx_src, incx,
                                    hy_dst, incy, queue,
                                    func, file, line );
}

static inline void
magma_scopyvector_async_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_async_internal( n, sizeof(float),
                                     dx_src, incx,
                                     dy_dst, incy, queue,
                                     func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

/// Type-safe version of magma_setmatrix() for float arrays.
/// @ingroup magma_setmatrix
#define magma_ssetmatrix(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_ssetmatrix_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix() for float arrays.
/// @ingroup magma_getmatrix
#define magma_sgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_sgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix() for float arrays.
/// @ingroup magma_copymatrix
#define magma_scopymatrix(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_scopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setmatrix_async() for float arrays.
/// @ingroup magma_setmatrix
#define magma_ssetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        magma_ssetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix_async() for float arrays.
/// @ingroup magma_getmatrix
#define magma_sgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        magma_sgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix_async() for float arrays.
/// @ingroup magma_copymatrix
#define magma_scopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_scopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_ssetmatrix_internal(
    magma_int_t m, magma_int_t n,
    float const    *hA_src, magma_int_t lda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_internal( m, n, sizeof(float),
                              hA_src, lda,
                              dB_dst, lddb, queue,
                              func, file, line );
}

static inline void
magma_sgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    float          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_internal( m, n, sizeof(float),
                              dA_src, ldda,
                              hB_dst, ldb, queue,
                              func, file, line );
}

static inline void
magma_scopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_internal( m, n, sizeof(float),
                               dA_src, ldda,
                               dB_dst, lddb, queue,
                               func, file, line );
}

static inline void
magma_ssetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    float const    *hA_src, magma_int_t lda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_async_internal( m, n, sizeof(float),
                                    hA_src, lda,
                                    dB_dst, lddb, queue,
                                    func, file, line );
}

static inline void
magma_sgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    float          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_async_internal( m, n, sizeof(float),
                                    dA_src, ldda,
                                    hB_dst, ldb, queue,
                                    func, file, line );
}

static inline void
magma_scopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_async_internal( m, n, sizeof(float),
                                     dA_src, ldda,
                                     dB_dst, lddb, queue,
                                     func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

magma_int_t
magma_isamax(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

magma_int_t
magma_isamin(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

float
magma_sasum(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_saxpy(
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_scopy(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

float
magma_sdot(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magma_queue_t queue );

float
magma_sdot(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magma_queue_t queue );

float
magma_snrm2(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_srot(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    float dc, float ds,
    magma_queue_t queue );

void
magma_srot(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    float dc, float ds,
    magma_queue_t queue );

void
magma_srotg(
    magmaFloat_ptr a,
    magmaFloat_ptr b,
    magmaFloat_ptr        c,
    magmaFloat_ptr s,
    magma_queue_t queue );

#ifdef REAL
void
magma_srotm(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magmaFloat_const_ptr param,
    magma_queue_t queue );

void
magma_srotmg(
    magmaFloat_ptr       d1,
    magmaFloat_ptr       d2,
    magmaFloat_ptr       x1,
    magmaFloat_const_ptr y1,
    magmaFloat_ptr param,
    magma_queue_t queue );
#endif

void
magma_sscal(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_sscal(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_sswap(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magma_queue_t queue );

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
magma_sgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_sger(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

#ifdef COMPLEX
void
magma_sger(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_ssymv(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_ssyr(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_ssyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );
#endif // COMPLEX

void
magma_ssymv(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_ssyr(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_ssyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_strmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_strsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dx, magma_int_t incx,
    magma_queue_t queue );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
magma_sgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_strmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magma_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

magma_int_t
magma_spotf2_lpout(
        magma_uplo_t uplo, magma_int_t n, 
        float *dA, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *dinfo, magma_queue_t queue);

magma_int_t
magma_spotf2_lpin(
        magma_uplo_t uplo, magma_int_t n, 
        float *dA, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *dinfo, magma_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef REAL

#endif // MAGMABLAS_S_H
