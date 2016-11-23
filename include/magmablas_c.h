/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magmablas_z.h, normal z -> c, Sun Nov 20 20:20:47 2016
*/

#ifndef MAGMABLAS_C_H
#define MAGMABLAS_C_H

#include "magma_types.h"
#include "magma_copy.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
magmablas_ctranspose_inplace(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_ctranspose_conj_inplace(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_ctranspose(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, magma_int_t lddat,
    magma_queue_t queue );

void
magmablas_ctranspose_conj(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, magma_int_t lddat,
    magma_queue_t queue );

void
magmablas_cgetmatrix_transpose(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloatComplex_const_ptr dAT,   magma_int_t ldda,
    magmaFloatComplex          *hA,    magma_int_t lda,
    magmaFloatComplex_ptr       dwork, magma_int_t lddw,
    magma_queue_t queues[2] );

void
magmablas_csetmatrix_transpose(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const magmaFloatComplex *hA,    magma_int_t lda,
    magmaFloatComplex_ptr    dAT,   magma_int_t ldda,
    magmaFloatComplex_ptr    dwork, magma_int_t lddw,
    magma_queue_t queues[2] );

  /*
   * RBT-related functions
   */
void
magmablas_cprbt(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr du,
    magmaFloatComplex_ptr dv,
    magma_queue_t queue );

void
magmablas_cprbt_mv(
    magma_int_t n,
    magmaFloatComplex_ptr dv,
    magmaFloatComplex_ptr db,
    magma_queue_t queue );

void
magmablas_cprbt_mtv(
    magma_int_t n,
    magmaFloatComplex_ptr du,
    magmaFloatComplex_ptr db,
    magma_queue_t queue );

  /*
   * Multi-GPU copy functions
   */
void
magma_cgetmatrix_1D_col_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloatComplex_const_ptr const dA[], magma_int_t ldda,
    magmaFloatComplex                *hA,   magma_int_t lda,
    magma_queue_t queue[] );

void
magma_csetmatrix_1D_col_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const magmaFloatComplex *hA,   magma_int_t lda,
    magmaFloatComplex_ptr    dA[], magma_int_t ldda,
    magma_queue_t queue[] );

void
magma_cgetmatrix_1D_row_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloatComplex_const_ptr const dA[], magma_int_t ldda,
    magmaFloatComplex                *hA,   magma_int_t lda,
    magma_queue_t queue[] );

void
magma_csetmatrix_1D_row_bcyclic(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const magmaFloatComplex *hA,   magma_int_t lda,
    magmaFloatComplex_ptr    dA[], magma_int_t ldda,
    magma_queue_t queue[] );

void
magmablas_cgetmatrix_transpose_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloatComplex_const_ptr const dAT[],    magma_int_t ldda,
    magmaFloatComplex                *hA,       magma_int_t lda,
    magmaFloatComplex_ptr             dwork[],  magma_int_t lddw,
    magma_queue_t queues[][2] );

void
magmablas_csetmatrix_transpose_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb,
    const magmaFloatComplex *hA,      magma_int_t lda,
    magmaFloatComplex_ptr    dAT[],   magma_int_t ldda,
    magmaFloatComplex_ptr    dwork[], magma_int_t lddw,
    magma_queue_t queues[][2] );

// in src/chetrd_mgpu.cpp
// TODO rename csetmatrix_sy or similar
magma_int_t
magma_chtodhe(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaFloatComplex     *A,   magma_int_t lda,
    magmaFloatComplex_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][10],
    magma_int_t *info );

// in src/cpotrf3_mgpu.cpp
// TODO same as magma_chtodhe?
magma_int_t
magma_chtodpo(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaFloatComplex     *A,   magma_int_t lda,
    magmaFloatComplex_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info );

// in src/cpotrf3_mgpu.cpp
// TODO rename cgetmatrix_sy or similar
magma_int_t
magma_cdtohpo(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
    magmaFloatComplex     *A,   magma_int_t lda,
    magmaFloatComplex_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info );


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */
void
magmablas_chemm_mgpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloatComplex_ptr dB[],    magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dC[],    magma_int_t lddc,
    magmaFloatComplex_ptr dwork[], magma_int_t dworksiz,
    //magmaFloatComplex    *C,       magma_int_t ldc,
    //magmaFloatComplex    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t events[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t ncmplx );

magma_int_t
magmablas_chemv_mgpu(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    magmaFloatComplex_const_ptr dx,           magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr    dy,              magma_int_t incy,
    magmaFloatComplex       *hwork,           magma_int_t lhwork,
    magmaFloatComplex_ptr    dwork[],         magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] );

magma_int_t
magmablas_chemv_mgpu_sync(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    magmaFloatComplex_const_ptr dx,           magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr    dy,              magma_int_t incy,
    magmaFloatComplex       *hwork,           magma_int_t lhwork,
    magmaFloatComplex_ptr    dwork[],         magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] );

// Ichi's version, in src/chetrd_mgpu.cpp
void
magma_cher2k_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloatComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );

void
magmablas_cher2k_mgpu2(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA[], magma_int_t ldda, magma_int_t a_offset,
    magmaFloatComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloatComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue );

// in src/cpotrf_mgpu_right.cpp
void
magma_cherk_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloatComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );

// in src/cpotrf_mgpu_right.cpp
void
magma_cherk_mgpu2(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloatComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
magmablas_cgeadd(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_cgeadd2(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_clacpy(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_clacpy_conj(
    magma_int_t n,
    magmaFloatComplex_ptr dA1, magma_int_t lda1,
    magmaFloatComplex_ptr dA2, magma_int_t lda2,
    magma_queue_t queue );

void
magmablas_clacpy_sym_in(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_clacpy_sym_out(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

float
magmablas_clange(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

float
magmablas_clanhe(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

float
magmablas_clansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

void
magmablas_clarfg(
    magma_int_t n,
    magmaFloatComplex_ptr dalpha,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dtau,
    magma_queue_t queue );

void
magmablas_clascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_clascl_2x2(
    magma_type_t type, magma_int_t m,
    magmaFloatComplex_const_ptr dW, magma_int_t lddw,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_clascl2(
    magma_type_t type,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_clascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dD, magma_int_t lddd,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_claset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_claset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_claswp(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_claswp2(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_claswp_sym(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_claswpx(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_csymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_csymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride,
    magma_queue_t queue );

void
magmablas_ctrtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr d_dinvA,
    magma_queue_t queue );

  /*
   * to cleanup (alphabetical order)
   */
magma_int_t
magma_clarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV, magma_int_t lddv,
    magmaFloatComplex_const_ptr dT, magma_int_t lddt,
    magmaFloatComplex_ptr dC,       magma_int_t lddc,
    magmaFloatComplex_ptr dwork,    magma_int_t ldwork,
    magma_queue_t queue );

magma_int_t
magma_clarfb_gpu_gemm(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV, magma_int_t lddv,
    magmaFloatComplex_const_ptr dT, magma_int_t lddt,
    magmaFloatComplex_ptr dC,       magma_int_t lddc,
    magmaFloatComplex_ptr dwork,    magma_int_t ldwork,
    magmaFloatComplex_ptr dworkvt,  magma_int_t ldworkvt,
    magma_queue_t queue );

void
magma_clarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr dT, magma_int_t ldt,
    magmaFloatComplex_ptr c,
    magmaFloatComplex_ptr dwork,
    magma_queue_t queue );

void
magma_clarfg_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dAkk,
    magma_queue_t queue );

void
magma_clarfgtx_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr T,  magma_int_t ldt,
    magmaFloatComplex_ptr dwork,
    magma_queue_t queue );

void
magma_clarfgx_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter,
    magma_queue_t queue );

void
magma_clarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr tau,
    magmaFloatComplex_ptr C, magma_int_t ldc,
    magmaFloat_ptr        xnorm,
    magmaFloatComplex_ptr dT, magma_int_t iter,
    magmaFloatComplex_ptr work,
    magma_queue_t queue );

  /*
   * Level 1 BLAS (alphabetical order)
   */
void
magmablas_caxpycp(
    magma_int_t m,
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_const_ptr db,
    magma_queue_t queue );

void
magmablas_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue );

void
magmablas_cswapblk(
    magma_order_t order,
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset,
    magma_queue_t queue );

void
magmablas_cswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloatComplex_ptr dB, magma_int_t lddb, magma_int_t incb,
    magma_queue_t queue );

void
magmablas_scnrm2_adjust(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloatComplex_ptr dc,
    magma_queue_t queue );

void
magmablas_snrm2_check(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc,
    magma_queue_t queue );

void
magmablas_scnrm2_check(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc,
    magma_queue_t queue );

void
magmablas_scnrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magma_queue_t queue );

void
magmablas_scnrm2_row_check_adjust(
    magma_int_t k, float tol,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dxnorm2,
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dlsticc,
    magma_queue_t queue );

  /*
   * Level 2 BLAS (alphabetical order)
   */
// trsv were always queue versions
void
magmablas_ctrsv(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       db, magma_int_t incb,
    magma_queue_t queue );

// todo: move flag before queue?
void
magmablas_ctrsv_outofplace(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr db,       magma_int_t incb,
    magmaFloatComplex_ptr dx,
    magma_queue_t queue,
    magma_int_t flag );

void
magmablas_cgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue );

void
magmablas_cgemv_conj(
    magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue );

magma_int_t
magmablas_chemv(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

magma_int_t
magmablas_csymv(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

// hemv/symv_work were always queue versions
magma_int_t
magmablas_chemv_work(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magmaFloatComplex_ptr       dwork, magma_int_t lwork,
    magma_queue_t queue );

magma_int_t
magmablas_csymv_work(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magmaFloatComplex_ptr       dwork, magma_int_t lwork,
    magma_queue_t queue );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
magmablas_cgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_cgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_chemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_csymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_csyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_cher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    float  beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_csyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_cherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float  alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    float  beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ctrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_ctrsm_outofplace(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magmaFloatComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloatComplex_ptr d_dinvA, magma_int_t dinvA_length,
    magma_queue_t queue );

void
magmablas_ctrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magmaFloatComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloatComplex_ptr d_dinvA, magma_int_t dinvA_length,
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

/// Type-safe version of magma_setvector() for magmaFloatComplex arrays.
/// @ingroup magma_setvector
#define magma_csetvector(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_csetvector_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector() for magmaFloatComplex arrays.
/// @ingroup magma_getvector
#define magma_cgetvector(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_cgetvector_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector() for magmaFloatComplex arrays.
/// @ingroup magma_copyvector
#define magma_ccopyvector(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_ccopyvector_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setvector_async() for magmaFloatComplex arrays.
/// @ingroup magma_setvector
#define magma_csetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_csetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getvector_async() for magmaFloatComplex arrays.
/// @ingroup magma_getvector
#define magma_cgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_cgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copyvector_async() for magmaFloatComplex arrays.
/// @ingroup magma_copyvector
#define magma_ccopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_ccopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_csetvector_internal(
    magma_int_t n,
    magmaFloatComplex const    *hx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_internal( n, sizeof(magmaFloatComplex),
                              hx_src, incx,
                              dy_dst, incy, queue,
                              func, file, line );
}

static inline void
magma_cgetvector_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_internal( n, sizeof(magmaFloatComplex),
                              dx_src, incx,
                              hy_dst, incy, queue,
                              func, file, line );
}

static inline void
magma_ccopyvector_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_internal( n, sizeof(magmaFloatComplex),
                               dx_src, incx,
                               dy_dst, incy, queue,
                               func, file, line );
}

static inline void
magma_csetvector_async_internal(
    magma_int_t n,
    magmaFloatComplex const    *hx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setvector_async_internal( n, sizeof(magmaFloatComplex),
                                    hx_src, incx,
                                    dy_dst, incy, queue,
                                    func, file, line );
}

static inline void
magma_cgetvector_async_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getvector_async_internal( n, sizeof(magmaFloatComplex),
                                    dx_src, incx,
                                    hy_dst, incy, queue,
                                    func, file, line );
}

static inline void
magma_ccopyvector_async_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copyvector_async_internal( n, sizeof(magmaFloatComplex),
                                     dx_src, incx,
                                     dy_dst, incy, queue,
                                     func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

/// Type-safe version of magma_setmatrix() for magmaFloatComplex arrays.
/// @ingroup magma_setmatrix
#define magma_csetmatrix(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_csetmatrix_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix() for magmaFloatComplex arrays.
/// @ingroup magma_getmatrix
#define magma_cgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_cgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix() for magmaFloatComplex arrays.
/// @ingroup magma_copymatrix
#define magma_ccopymatrix(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_ccopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_setmatrix_async() for magmaFloatComplex arrays.
/// @ingroup magma_setmatrix
#define magma_csetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        magma_csetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_getmatrix_async() for magmaFloatComplex arrays.
/// @ingroup magma_getmatrix
#define magma_cgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        magma_cgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

/// Type-safe version of magma_copymatrix_async() for magmaFloatComplex arrays.
/// @ingroup magma_copymatrix
#define magma_ccopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_ccopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_csetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const    *hA_src, magma_int_t lda,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_internal( m, n, sizeof(magmaFloatComplex),
                              hA_src, lda,
                              dB_dst, lddb, queue,
                              func, file, line );
}

static inline void
magma_cgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_internal( m, n, sizeof(magmaFloatComplex),
                              dA_src, ldda,
                              hB_dst, ldb, queue,
                              func, file, line );
}

static inline void
magma_ccopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_internal( m, n, sizeof(magmaFloatComplex),
                               dA_src, ldda,
                               dB_dst, lddb, queue,
                               func, file, line );
}

static inline void
magma_csetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const    *hA_src, magma_int_t lda,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_setmatrix_async_internal( m, n, sizeof(magmaFloatComplex),
                                    hA_src, lda,
                                    dB_dst, lddb, queue,
                                    func, file, line );
}

static inline void
magma_cgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_getmatrix_async_internal( m, n, sizeof(magmaFloatComplex),
                                    dA_src, ldda,
                                    hB_dst, ldb, queue,
                                    func, file, line );
}

static inline void
magma_ccopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    magma_copymatrix_async_internal( m, n, sizeof(magmaFloatComplex),
                                     dA_src, ldda,
                                     dB_dst, lddb, queue,
                                     func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

magma_int_t
magma_icamax(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

magma_int_t
magma_icamin(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

float
magma_scasum(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_caxpy(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_ccopy(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

magmaFloatComplex
magma_cdotc(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magma_queue_t queue );

magmaFloatComplex
magma_cdotu(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magma_queue_t queue );

float
magma_scnrm2(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_crot(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float dc, magmaFloatComplex ds,
    magma_queue_t queue );

void
magma_csrot(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float dc, float ds,
    magma_queue_t queue );

void
magma_crotg(
    magmaFloatComplex_ptr a,
    magmaFloatComplex_ptr b,
    magmaFloat_ptr        c,
    magmaFloatComplex_ptr s,
    magma_queue_t queue );

#ifdef REAL
void
magma_crotm(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magmaFloat_const_ptr param,
    magma_queue_t queue );

void
magma_crotmg(
    magmaFloat_ptr       d1,
    magmaFloat_ptr       d2,
    magmaFloat_ptr       x1,
    magmaFloat_const_ptr y1,
    magmaFloat_ptr param,
    magma_queue_t queue );
#endif

void
magma_cscal(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_csscal(
    magma_int_t n,
    float alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue );

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
magma_cgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_cgerc(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

#ifdef COMPLEX
void
magma_cgeru(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_chemv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_cher(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_cher2(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );
#endif // COMPLEX

void
magma_csymv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_csyr(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_csyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_ctrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_ctrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx,
    magma_queue_t queue );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
magma_cgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_chemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_cher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_cherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_csymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_csyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_csyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ctrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magma_ctrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

magma_int_t
magma_cpotf2_lpout(
        magma_uplo_t uplo, magma_int_t n, 
        magmaFloatComplex *dA, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *dinfo, magma_queue_t queue);

magma_int_t
magma_cpotf2_lpin(
        magma_uplo_t uplo, magma_int_t n, 
        magmaFloatComplex *dA, magma_int_t lda, magma_int_t gbstep,
        magma_int_t *dinfo, magma_queue_t queue);

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif // MAGMABLAS_C_H
