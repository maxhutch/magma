/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magmablas_z.h normal z -> c, Fri Jan 30 19:00:06 2015
*/

#ifndef MAGMABLAS_C_H
#define MAGMABLAS_C_H

#include "magma_types.h"

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
    magmaFloatComplex_ptr dA, magma_int_t ldda );

void
magmablas_ctranspose(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, magma_int_t lddat );

void
magmablas_cgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dAT,   magma_int_t ldda,
    magmaFloatComplex          *hA,    magma_int_t lda,
    magmaFloatComplex_ptr       dwork, magma_int_t lddwork, magma_int_t nb );

void
magmablas_csetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA,    magma_int_t lda,
    magmaFloatComplex_ptr    dAT,   magma_int_t ldda,
    magmaFloatComplex_ptr    dwork, magma_int_t lddwork, magma_int_t nb );

  /*
   * RBT-related functions
   */
void
magmablas_cprbt(
    magma_int_t n, 
    magmaFloatComplex *dA, magma_int_t ldda, 
    magmaFloatComplex *du, magmaFloatComplex *dv);

void
magmablas_cprbt_mv(
    magma_int_t n, 
    magmaFloatComplex *dv, magmaFloatComplex *db);

void
magmablas_cprbt_mtv(
    magma_int_t n, 
    magmaFloatComplex *du, magmaFloatComplex *db);

void
magmablas_caxpycp2(
    magma_int_t m, magmaFloatComplex *r, magmaFloatComplex *x,
    const magmaFloatComplex *b);

  /*
   * Multi-GPU copy functions
   */
void
magmablas_cgetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    magmaFloatComplex_const_ptr const dAT[],    magma_int_t ldda,
    magmaFloatComplex                *hA,       magma_int_t lda,
    magmaFloatComplex_ptr             dwork[],  magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void
magmablas_csetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    const magmaFloatComplex *hA,      magma_int_t lda,
    magmaFloatComplex_ptr    dAT[],   magma_int_t ldda,
    magmaFloatComplex_ptr    dwork[], magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void
magma_cgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const dA[], magma_int_t ldda,
    magmaFloatComplex                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_csetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA,   magma_int_t lda,
    magmaFloatComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_cgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const dA[], magma_int_t ldda,
    magmaFloatComplex                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_csetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA,   magma_int_t lda,
    magmaFloatComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

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
    magmaFloatComplex    *C,       magma_int_t ldc,
    magmaFloatComplex    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][20], magma_int_t nbevents );

void
magmablas_chemm_mgpu_com(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloatComplex_ptr dB[],    magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dC[],    magma_int_t lddc,
    magmaFloatComplex_ptr dwork[], magma_int_t dworksiz,
    magmaFloatComplex    *C,       magma_int_t ldc,
    magmaFloatComplex    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void
magmablas_chemm_mgpu_spec(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloatComplex_ptr dB[],    magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dC[],    magma_int_t lddc,
    magmaFloatComplex_ptr dwork[], magma_int_t dworksiz,
    magmaFloatComplex    *C,       magma_int_t ldc,
    magmaFloatComplex    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void
magmablas_chemm_mgpu_spec33(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloatComplex_ptr dB[],    magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dC[],    magma_int_t lddc,
    magmaFloatComplex_ptr dVIN[],  magma_int_t lddv, magma_int_t v_offset,
    magmaFloatComplex_ptr dwork[], magma_int_t dworksiz,
    magmaFloatComplex *C,       magma_int_t ldc,
    magmaFloatComplex *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

magma_int_t
magmablas_chemv_mgpu(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    magmaFloatComplex const *x,         magma_int_t incx,
    magmaFloatComplex beta,             
    magmaFloatComplex       *y,         magma_int_t incy,
    magmaFloatComplex       *hwork,     magma_int_t lhwork,
    magmaFloatComplex_ptr    dwork[],   magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] );

magma_int_t
magmablas_chemv_mgpu_sync(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    magmaFloatComplex const *x,         magma_int_t incx,
    magmaFloatComplex beta,             
    magmaFloatComplex       *y,         magma_int_t incy,
    magmaFloatComplex       *hwork,     magma_int_t lhwork,
    magmaFloatComplex_ptr    dwork[],   magma_int_t ldwork,
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
magmablas_cher2k_mgpu_spec(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA[], magma_int_t ldda, magma_int_t a_offset,
    magmaFloatComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloatComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t ngpu, magma_int_t nb, magma_queue_t queues[][20], magma_int_t nqueue );

void
magmablas_cher2k_mgpu_spec324(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dVIN[], magma_int_t lddv, magma_int_t v_offset,
    magmaFloatComplex_ptr dWIN[], magma_int_t lddw, magma_int_t w_offset,
    float beta,
    magmaFloatComplex_ptr dC[],   magma_int_t lddc, magma_int_t c_offset,
    magmaFloatComplex_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

void
magmablas_cher2k_mgpu_spec325(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dVIN[], magma_int_t lddv, magma_int_t v_offset,
    magmaFloatComplex_ptr dWIN[], magma_int_t lddw, magma_int_t w_offset,
    float beta,
    magmaFloatComplex_ptr dC[],   magma_int_t lddc, magma_int_t c_offset,
    magmaFloatComplex_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    magmaFloatComplex *harray[],
    magmaFloatComplex_ptr *darray[],
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

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
    magma_int_t nqueue, magma_queue_t queues[][10]);

// in src/cpotrf_mgpu_right.cpp
void
magma_cherk_mgpu2(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    float beta,
    magmaFloatComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10]);


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
magmablas_cgeadd(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magmablas_clacpy(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magmablas_clacpy_cnjg(
    magma_int_t n, magmaFloatComplex *dA1, magma_int_t lda1,
    magmaFloatComplex *dA2, magma_int_t lda2);

float
magmablas_clange(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork );

float
magmablas_clanhe(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork );

float
magmablas_clansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork );

void
magmablas_clarfg(
    magma_int_t n,
    magmaFloatComplex_ptr dalpha, magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dtau );

void
magmablas_clarfg_work(
    magma_int_t n,
    magmaFloatComplex_ptr dalpha, magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dtau, magmaFloatComplex_ptr dwork );

void
magmablas_clascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_clascl_2x2(
    magma_type_t type, magma_int_t m,
    magmaFloatComplex *dW, magma_int_t lddw,
    magmaFloatComplex *dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_clascl2(
    magma_type_t type,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_clascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dD, magma_int_t lddd,
          magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_claset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda );

void
magmablas_claset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda);

void
magmablas_claswp(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_claswp2(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci );

void
magmablas_claswpx(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_csymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda );

void
magmablas_csymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride );

void
magmablas_ctrtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr d_dinvA);

  /*
   * to cleanup (alphabetical order)
   */
void
magmablas_scnrm2_adjust(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloatComplex_ptr dc);

void
magmablas_scnrm2_check(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc);

void
magmablas_scnrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm);

void
magmablas_scnrm2_row_check_adjust(
    magma_int_t k, float tol,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dxnorm2,
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dlsticc);

void
magma_clarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr dT, magma_int_t ldt,
    magmaFloatComplex_ptr c,
    magmaFloatComplex_ptr dwork);

void
magma_clarfg_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dAkk );

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
    magmaFloatComplex_ptr dwork);

void
magma_clarfgx_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter);

void
magma_clarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr tau,
    magmaFloatComplex_ptr C,  magma_int_t ldc,
    magmaFloat_ptr        xnorm,
    magmaFloatComplex_ptr dT, magma_int_t iter,
    magmaFloatComplex_ptr work);


  /*
   * Level 1 BLAS (alphabetical order)
   */
void
magmablas_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy );

void
magmablas_cswapblk(
    magma_order_t order,
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset );

void
magmablas_cswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloatComplex_ptr dB, magma_int_t lddb, magma_int_t incb );

  /*
   * Level 2 BLAS (alphabetical order)
   */
void
magmablas_cgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );


void
magmablas_cgemv_conjv(
    magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy, magma_int_t incy);

magma_int_t
magmablas_chemv(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

magma_int_t
magmablas_csymv(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

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
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_cgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_chemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_csymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_csyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_cher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    float  beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_csyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_cherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float  alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    float  beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_ctrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magmablas_ctrsm_outofplace(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_int_t flag, magmaFloatComplex_ptr d_dinvA, magmaFloatComplex_ptr dX );

void
magmablas_ctrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_int_t flag, magmaFloatComplex_ptr d_dinvA, magmaFloatComplex_ptr dX );


  /*
   * Wrappers for platform independence.
   * These wrap CUBLAS or AMD OpenCL BLAS functions.
   */

// ========================================
// copying vectors
// set copies host to device
// get copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_csetvector(           n, hx_src, incx, dy_dst, incy ) \
        magma_csetvector_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_cgetvector(           n, dx_src, incx, hy_dst, incy ) \
        magma_cgetvector_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_ccopyvector(          n, dx_src, incx, dy_dst, incy ) \
        magma_ccopyvector_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_csetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_csetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_cgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_cgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_ccopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_ccopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

void
magma_csetvector_internal(
    magma_int_t n,
    magmaFloatComplex const    *hx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void
magma_cgetvector_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex          *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void
magma_ccopyvector_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void
magma_csetvector_async_internal(
    magma_int_t n,
    magmaFloatComplex const    *hx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_cgetvector_async_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_ccopyvector_async_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// copying sub-matrices (contiguous columns)
// set  copies host to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_csetmatrix(           m, n, hA_src, lda, dB_dst, lddb ) \
        magma_csetmatrix_internal(  m, n, hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_cgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb ) \
        magma_cgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_ccopymatrix(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_ccopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_csetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        magma_csetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_cgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        magma_cgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

#define magma_ccopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_ccopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

void
magma_csetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const    *hA_src, magma_int_t ldha,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void
magma_cgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex          *hB_dst, magma_int_t ldhb,
    const char* func, const char* file, int line );

void
magma_ccopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void
magma_csetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const    *hA_src, magma_int_t ldha,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_cgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex          *hB_dst, magma_int_t ldhb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_ccopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// Level 1 BLAS (alphabetical order)

// in cublas_v2, result returned through output argument
magma_int_t
magma_icamax(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t
magma_icamin(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
float
magma_scasum(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

void
magma_caxpy(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

void
magma_ccopy(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaFloatComplex
magma_cdotc(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaFloatComplex
magma_cdotu(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float
magma_scnrm2(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

void
magma_crot(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float dc, magmaFloatComplex ds );

void
magma_csrot(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float dc, float ds );

#ifdef REAL
void
magma_crotm(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magmaFloat_const_ptr param );

void
magma_crotmg(
    magmaFloat_ptr d1, magmaFloat_ptr       d2,
    magmaFloat_ptr x1, magmaFloat_const_ptr y1,
    magmaFloat_ptr param );
#endif

void
magma_cscal(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx );

void
magma_csscal(
    magma_int_t n,
    float alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx );

void
magma_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy );

// ========================================
// Level 2 BLAS (alphabetical order)

void
magma_cgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

void
magma_cgerc(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda );

void
magma_cgeru(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda );

void
magma_chemv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

void
magma_cher(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dA, magma_int_t ldda );

void
magma_cher2(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda );

void
magma_ctrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx );

void
magma_ctrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx );

// ========================================
// Level 3 BLAS (alphabetical order)

void
magma_cgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_csymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_chemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_csyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_cher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_csyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_cherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_ctrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magma_ctrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );


#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif  /* MAGMABLAS_C_H */
