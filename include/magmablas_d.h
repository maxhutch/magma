/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magmablas_z.h normal z -> d, Fri Jan 30 19:00:06 2015
*/

#ifndef MAGMABLAS_D_H
#define MAGMABLAS_D_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
magmablas_dtranspose_inplace(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda );

void
magmablas_dtranspose(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA,  magma_int_t ldda,
    magmaDouble_ptr       dAT, magma_int_t lddat );

void
magmablas_dgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dAT,   magma_int_t ldda,
    double          *hA,    magma_int_t lda,
    magmaDouble_ptr       dwork, magma_int_t lddwork, magma_int_t nb );

void
magmablas_dsetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const double *hA,    magma_int_t lda,
    magmaDouble_ptr    dAT,   magma_int_t ldda,
    magmaDouble_ptr    dwork, magma_int_t lddwork, magma_int_t nb );

  /*
   * RBT-related functions
   */
void
magmablas_dprbt(
    magma_int_t n, 
    double *dA, magma_int_t ldda, 
    double *du, double *dv);

void
magmablas_dprbt_mv(
    magma_int_t n, 
    double *dv, double *db);

void
magmablas_dprbt_mtv(
    magma_int_t n, 
    double *du, double *db);

void
magmablas_daxpycp2(
    magma_int_t m, double *r, double *x,
    const double *b);

  /*
   * Multi-GPU copy functions
   */
void
magmablas_dgetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    magmaDouble_const_ptr const dAT[],    magma_int_t ldda,
    double                *hA,       magma_int_t lda,
    magmaDouble_ptr             dwork[],  magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void
magmablas_dsetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    const double *hA,      magma_int_t lda,
    magmaDouble_ptr    dAT[],   magma_int_t ldda,
    magmaDouble_ptr    dwork[], magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void
magma_dgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const dA[], magma_int_t ldda,
    double                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_dsetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const double *hA,   magma_int_t lda,
    magmaDouble_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_dgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const dA[], magma_int_t ldda,
    double                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_dsetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const double *hA,   magma_int_t lda,
    magmaDouble_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

// in src/dsytrd_mgpu.cpp
// TODO rename dsetmatrix_sy or similar
magma_int_t
magma_dhtodhe(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    double     *A,   magma_int_t lda,
    magmaDouble_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][10],
    magma_int_t *info );

// in src/dpotrf3_mgpu.cpp
// TODO same as magma_dhtodhe?
magma_int_t
magma_dhtodpo(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    double     *A,   magma_int_t lda,
    magmaDouble_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info );

// in src/dpotrf3_mgpu.cpp
// TODO rename dgetmatrix_sy or similar
magma_int_t
magma_ddtohpo(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
    double     *A,   magma_int_t lda,
    magmaDouble_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info );


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */
void
magmablas_dsymm_mgpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[],    magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[],    magma_int_t lddc,
    magmaDouble_ptr dwork[], magma_int_t dworksiz,
    double    *C,       magma_int_t ldc,
    double    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][20], magma_int_t nbevents );

void
magmablas_dsymm_mgpu_com(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[],    magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[],    magma_int_t lddc,
    magmaDouble_ptr dwork[], magma_int_t dworksiz,
    double    *C,       magma_int_t ldc,
    double    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void
magmablas_dsymm_mgpu_spec(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[],    magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[],    magma_int_t lddc,
    magmaDouble_ptr dwork[], magma_int_t dworksiz,
    double    *C,       magma_int_t ldc,
    double    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void
magmablas_dsymm_mgpu_spec33(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[],    magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[],    magma_int_t lddc,
    magmaDouble_ptr dVIN[],  magma_int_t lddv, magma_int_t v_offset,
    magmaDouble_ptr dwork[], magma_int_t dworksiz,
    double *C,       magma_int_t ldc,
    double *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

magma_int_t
magmablas_dsymv_mgpu(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    double const *x,         magma_int_t incx,
    double beta,             
    double       *y,         magma_int_t incy,
    double       *hwork,     magma_int_t lhwork,
    magmaDouble_ptr    dwork[],   magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] );

magma_int_t
magmablas_dsymv_mgpu_sync(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    double const *x,         magma_int_t incx,
    double beta,             
    double       *y,         magma_int_t incy,
    double       *hwork,     magma_int_t lhwork,
    magmaDouble_ptr    dwork[],   magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] );

// Ichi's version, in src/dsytrd_mgpu.cpp
void
magma_dsyr2k_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );

void
magmablas_dsyr2k_mgpu_spec(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_ptr dA[], magma_int_t ldda, magma_int_t a_offset,
    magmaDouble_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t ngpu, magma_int_t nb, magma_queue_t queues[][20], magma_int_t nqueue );

void
magmablas_dsyr2k_mgpu_spec324(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dVIN[], magma_int_t lddv, magma_int_t v_offset,
    magmaDouble_ptr dWIN[], magma_int_t lddw, magma_int_t w_offset,
    double beta,
    magmaDouble_ptr dC[],   magma_int_t lddc, magma_int_t c_offset,
    magmaDouble_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

void
magmablas_dsyr2k_mgpu_spec325(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dVIN[], magma_int_t lddv, magma_int_t v_offset,
    magmaDouble_ptr dWIN[], magma_int_t lddw, magma_int_t w_offset,
    double beta,
    magmaDouble_ptr dC[],   magma_int_t lddc, magma_int_t c_offset,
    magmaDouble_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    double *harray[],
    magmaDouble_ptr *darray[],
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

void
magmablas_dsyr2k_mgpu2(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_ptr dA[], magma_int_t ldda, magma_int_t a_offset,
    magmaDouble_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue );

// in src/dpotrf_mgpu_right.cpp
void
magma_dsyrk_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10]);

// in src/dpotrf_mgpu_right.cpp
void
magma_dsyrk_mgpu2(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10]);


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
magmablas_dgeadd(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb );

void
magmablas_dlacpy(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb );

void
magmablas_dlacpy_cnjg(
    magma_int_t n, double *dA1, magma_int_t lda1,
    double *dA2, magma_int_t lda2);

double
magmablas_dlange(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork );

double
magmablas_dlansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork );

double
magmablas_dlansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork );

void
magmablas_dlarfg(
    magma_int_t n,
    magmaDouble_ptr dalpha, magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dtau );

void
magmablas_dlarfg_work(
    magma_int_t n,
    magmaDouble_ptr dalpha, magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dtau, magmaDouble_ptr dwork );

void
magmablas_dlascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_dlascl_2x2(
    magma_type_t type, magma_int_t m,
    double *dW, magma_int_t lddw,
    double *dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_dlascl2(
    magma_type_t type,
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_dlascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD, magma_int_t lddd,
          magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_dlaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    magmaDouble_ptr dA, magma_int_t ldda );

void
magmablas_dlaset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    double offdiag, double diag,
    magmaDouble_ptr dA, magma_int_t ldda);

void
magmablas_dlaswp(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_dlaswp2(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci );

void
magmablas_dlaswpx(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_dsymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaDouble_ptr dA, magma_int_t ldda );

void
magmablas_dsymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride );

void
magmablas_dtrtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr d_dinvA);

  /*
   * to cleanup (alphabetical order)
   */
void
magmablas_dnrm2_adjust(
    magma_int_t k,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dc);

void
magmablas_dnrm2_check(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc);

void
magmablas_dnrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm);

void
magmablas_dnrm2_row_check_adjust(
    magma_int_t k, double tol,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dxnorm2,
    magmaDouble_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dlsticc);

void
magma_dlarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaDouble_ptr V,  magma_int_t ldv,
    magmaDouble_ptr dT, magma_int_t ldt,
    magmaDouble_ptr c,
    magmaDouble_ptr dwork);

void
magma_dlarfg_gpu(
    magma_int_t n,
    magmaDouble_ptr dx0,
    magmaDouble_ptr dx,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDouble_ptr dAkk );

void
magma_dlarfgtx_gpu(
    magma_int_t n,
    magmaDouble_ptr dx0,
    magmaDouble_ptr dx,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDouble_ptr dA, magma_int_t iter,
    magmaDouble_ptr V,  magma_int_t ldv,
    magmaDouble_ptr T,  magma_int_t ldt,
    magmaDouble_ptr dwork);

void
magma_dlarfgx_gpu(
    magma_int_t n,
    magmaDouble_ptr dx0,
    magmaDouble_ptr dx,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDouble_ptr dA, magma_int_t iter);

void
magma_dlarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr v,
    magmaDouble_ptr tau,
    magmaDouble_ptr C,  magma_int_t ldc,
    magmaDouble_ptr        xnorm,
    magmaDouble_ptr dT, magma_int_t iter,
    magmaDouble_ptr work);


  /*
   * Level 1 BLAS (alphabetical order)
   */
void
magmablas_dswap(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy );

void
magmablas_dswapblk(
    magma_order_t order,
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset );

void
magmablas_dswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaDouble_ptr dB, magma_int_t lddb, magma_int_t incb );

  /*
   * Level 2 BLAS (alphabetical order)
   */
void
magmablas_dgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );


void
magmablas_dgemv_conjv(
    magma_int_t m, magma_int_t n, double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr dy, magma_int_t incy);

magma_int_t
magmablas_dsymv(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );

magma_int_t
magmablas_dsymv(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );

magma_int_t
magmablas_dsymv_work(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy,
    magmaDouble_ptr       dwork, magma_int_t lwork,
    magma_queue_t queue );

magma_int_t
magmablas_dsymv_work(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy,
    magmaDouble_ptr       dwork, magma_int_t lwork,
    magma_queue_t queue );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
magmablas_dgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magmablas_dgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magmablas_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magmablas_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magmablas_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magmablas_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double  beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magmablas_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magmablas_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double  alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double  beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magmablas_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb );

void
magmablas_dtrsm_outofplace(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb,
    magma_int_t flag, magmaDouble_ptr d_dinvA, magmaDouble_ptr dX );

void
magmablas_dtrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb,
    magma_int_t flag, magmaDouble_ptr d_dinvA, magmaDouble_ptr dX );


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

#define magma_dsetvector(           n, hx_src, incx, dy_dst, incy ) \
        magma_dsetvector_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_dgetvector(           n, dx_src, incx, hy_dst, incy ) \
        magma_dgetvector_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_dcopyvector(          n, dx_src, incx, dy_dst, incy ) \
        magma_dcopyvector_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_dsetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_dsetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_dgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_dgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_dcopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_dcopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

void
magma_dsetvector_internal(
    magma_int_t n,
    double const    *hx_src, magma_int_t incx,
    magmaDouble_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void
magma_dgetvector_internal(
    magma_int_t n,
    magmaDouble_const_ptr dx_src, magma_int_t incx,
    double          *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void
magma_dcopyvector_internal(
    magma_int_t n,
    magmaDouble_const_ptr dx_src, magma_int_t incx,
    magmaDouble_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void
magma_dsetvector_async_internal(
    magma_int_t n,
    double const    *hx_src, magma_int_t incx,
    magmaDouble_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_dgetvector_async_internal(
    magma_int_t n,
    magmaDouble_const_ptr dx_src, magma_int_t incx,
    double          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_dcopyvector_async_internal(
    magma_int_t n,
    magmaDouble_const_ptr dx_src, magma_int_t incx,
    magmaDouble_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// copying sub-matrices (contiguous columns)
// set  copies host to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_dsetmatrix(           m, n, hA_src, lda, dB_dst, lddb ) \
        magma_dsetmatrix_internal(  m, n, hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_dgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb ) \
        magma_dgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_dcopymatrix(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_dcopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_dsetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        magma_dsetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_dgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        magma_dgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

#define magma_dcopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_dcopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

void
magma_dsetmatrix_internal(
    magma_int_t m, magma_int_t n,
    double const    *hA_src, magma_int_t ldha,
    magmaDouble_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void
magma_dgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, magma_int_t ldda,
    double          *hB_dst, magma_int_t ldhb,
    const char* func, const char* file, int line );

void
magma_dcopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, magma_int_t ldda,
    magmaDouble_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void
magma_dsetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    double const    *hA_src, magma_int_t ldha,
    magmaDouble_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_dgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, magma_int_t ldda,
    double          *hB_dst, magma_int_t ldhb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void
magma_dcopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, magma_int_t ldda,
    magmaDouble_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// Level 1 BLAS (alphabetical order)

// in cublas_v2, result returned through output argument
magma_int_t
magma_idamax(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t
magma_idamin(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
double
magma_dasum(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

void
magma_daxpy(
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dy, magma_int_t incy );

void
magma_dcopy(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
double
magma_ddot(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
double
magma_ddot(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
double
magma_dnrm2(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

void
magma_drot(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    double dc, double ds );

void
magma_drot(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    double dc, double ds );

#ifdef REAL
void
magma_drotm(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    magmaDouble_const_ptr param );

void
magma_drotmg(
    magmaDouble_ptr d1, magmaDouble_ptr       d2,
    magmaDouble_ptr x1, magmaDouble_const_ptr y1,
    magmaDouble_ptr param );
#endif

void
magma_dscal(
    magma_int_t n,
    double alpha,
    magmaDouble_ptr dx, magma_int_t incx );

void
magma_dscal(
    magma_int_t n,
    double alpha,
    magmaDouble_ptr dx, magma_int_t incx );

void
magma_dswap(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy );

// ========================================
// Level 2 BLAS (alphabetical order)

void
magma_dgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );

void
magma_dger(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda );

void
magma_dger(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda );

void
magma_dsymv(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );

void
magma_dsyr(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dA, magma_int_t ldda );

void
magma_dsyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda );

void
magma_dtrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dx, magma_int_t incx );

void
magma_dtrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dx, magma_int_t incx );

// ========================================
// Level 3 BLAS (alphabetical order)

void
magma_dgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void
magma_dtrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb );

void
magma_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb );


#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMABLAS_D_H */
