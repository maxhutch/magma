/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
*/

#ifndef MAGMABLAS_Z_H
#define MAGMABLAS_Z_H

#include "magma_types.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Interface to clean
   */

// see also zlaswp
// ipiv gets updated
void magmablas_zpermute_long2(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t nb, magma_int_t ind );

// ipiv is not updated (unlike zpermute_long2)
void magmablas_zpermute_long3(
    /*magma_int_t n,*/
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    const magma_int_t *ipiv, magma_int_t nb, magma_int_t ind );

  /*
   * Transpose functions
   */
void magmablas_ztranspose_inplace(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda );

void magmablas_ztranspose(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA,  magma_int_t ldda,
    magmaDoubleComplex_ptr       dAT, magma_int_t lddat );

void magmablas_zgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dAT,   magma_int_t ldda,
    magmaDoubleComplex          *hA,    magma_int_t lda,
    magmaDoubleComplex_ptr       dwork, magma_int_t lddwork, magma_int_t nb );

void magmablas_zsetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,    magma_int_t lda,
    magmaDoubleComplex_ptr    dAT,   magma_int_t ldda,
    magmaDoubleComplex_ptr    dwork, magma_int_t lddwork, magma_int_t nb );

  /*
   * Multi-GPU functions
   */
void magmablas_zgetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t stream[][2],
    magmaDoubleComplex_ptr  dAT[], magma_int_t ldda,
    magmaDoubleComplex     *hA,    magma_int_t lda,
    magmaDoubleComplex_ptr  dB[],  magma_int_t lddb,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void magmablas_zsetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t stream[][2],
    const magmaDoubleComplex *hA,    magma_int_t lda,
    magmaDoubleComplex_ptr    dAT[], magma_int_t ldda,
    magmaDoubleComplex_ptr    dB[],  magma_int_t lddb,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void magma_zgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,
    magmaDoubleComplex    *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void magma_zsetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,   magma_int_t lda,
    magmaDoubleComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

void magma_zgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,
    magmaDoubleComplex    *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void magma_zsetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,   magma_int_t lda,
    magmaDoubleComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

// in src/zhetrd_mgpu.cpp
magma_int_t magma_zhtodhe(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex **dwork, magma_int_t ldda,
    magma_queue_t stream[][10], magma_int_t *info );

// in src/zpotrf3_mgpu.cpp
magma_int_t magma_zhtodpo(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaDoubleComplex  *h_A,   magma_int_t lda,
    magmaDoubleComplex *d_lA[], magma_int_t ldda,
    magma_queue_t stream[][3], magma_int_t *info );

// in src/zpotrf3_mgpu.cpp
magma_int_t magma_zdtohpo(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
    magmaDoubleComplex  *a,     magma_int_t lda,
    magmaDoubleComplex *work[], magma_int_t ldda,
    magma_queue_t stream[][3], magma_int_t *info );

magma_int_t magmablas_zhemv_mgpu_offset(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **X, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **Y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_queue_t stream[][10] );

magma_int_t magmablas_zhemv_mgpu_32_offset(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **X, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **Y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_queue_t stream[][10] );

magma_int_t magmablas_zhemv2_mgpu_offset(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset);

magma_int_t magmablas_zhemv2_mgpu_32_offset(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset);

magma_int_t magmablas_zhemv_mgpu(
    magma_int_t num_gpus, magma_int_t k, magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **da, magma_int_t ldda, magma_int_t offset,
    magmaDoubleComplex **dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dy, magma_int_t incy,
    magmaDoubleComplex **dwork, magma_int_t ldwork,
    magmaDoubleComplex *work, magmaDoubleComplex *w,
    magma_queue_t stream[][10] );

magma_int_t magmablas_zhemv_sync(
    magma_int_t num_gpus, magma_int_t k,
    magma_int_t n, magmaDoubleComplex *work, magmaDoubleComplex *w,
    magma_queue_t stream[][10] );

void magmablas_zhemm_1gpu_old(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc,
    magmaDoubleComplex*    C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_zhemm_1gpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc,
    magmaDoubleComplex*    C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_zhemm_mgpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[],    magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[],    magma_int_t lddc,
    magmaDoubleComplex_ptr dwork[], magma_int_t lddwork,
    magmaDoubleComplex*    C,       magma_int_t ldc,
    magmaDoubleComplex*    work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][20], magma_int_t nbevents );

void magmablas_zhemm_mgpu_com(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[],    magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[],    magma_int_t lddc,
    magmaDoubleComplex_ptr dwork[], magma_int_t lddwork,
    magmaDoubleComplex*    C,       magma_int_t ldc,
    magmaDoubleComplex*    work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void magmablas_zhemm_mgpu_spec(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[],    magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[],    magma_int_t lddc,
    magmaDoubleComplex_ptr dwork[], magma_int_t lddwork,
    magmaDoubleComplex*    C,       magma_int_t ldc,
    magmaDoubleComplex*    work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void magmablas_zhemm_mgpu_spec33(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[],    magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[],    magma_int_t lddc,
    magmaDoubleComplex_ptr dVIN[],  magma_int_t lddv, magma_int_t voffst,
    magmaDoubleComplex_ptr dwork[], magma_int_t lddwork,
    magmaDoubleComplex *C,       magma_int_t ldc,
    magmaDoubleComplex *work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

// Ichi's version, in src/zhetrd_mgpu.cpp
void magma_zher2k_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **db, magma_int_t lddb, magma_int_t boffset,
    double beta,
    magmaDoubleComplex **dc, magma_int_t lddc, magma_int_t offset,
    magma_int_t num_qs, magma_queue_t streams[][10] );

void magmablas_zher2k_mgpu2(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_int_t aoff,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t boff,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t offset,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_zher2k_mgpu_spec(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[], magma_int_t lda, magma_int_t aoff,
    magmaDoubleComplex_ptr dB[], magma_int_t ldb, magma_int_t boff,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t ldc,  magma_int_t offset,
    magma_int_t ngpu, magma_int_t nb, magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_zher2k_mgpu_spec324(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dVIN[], magma_int_t lddv, magma_int_t voff,
    magmaDoubleComplex_ptr dWIN[], magma_int_t lddw, magma_int_t woff,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc,  magma_int_t offset,
    magmaDoubleComplex_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

void magmablas_zher2k_mgpu_spec325(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dVIN[], magma_int_t lddv, magma_int_t voff,
    magmaDoubleComplex_ptr dWIN[], magma_int_t lddw, magma_int_t woff,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc,  magma_int_t offset,
    magmaDoubleComplex_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    magmaDoubleComplex **harray[],
    magmaDoubleComplex_ptr *darray[],
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

  /*
   * LAPACK auxiliary functions
   */
void magmablas_zgeadd(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void magmablas_zgeadd_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr  const *dAarray, magma_int_t ldda,
    magmaDoubleComplex_ptr              *dBarray, magma_int_t lddb,
    magma_int_t batchCount );

void magmablas_zlacpy(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void magmablas_zlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr  const *dAarray, magma_int_t ldda,
    magmaDoubleComplex_ptr              *dBarray, magma_int_t lddb,
    magma_int_t batchCount );

double magmablas_zlange(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork );

double magmablas_zlanhe(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork );

double magmablas_zlansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork );

void magmablas_zlarfg(
    magma_int_t n,
    magmaDoubleComplex *dalpha, magmaDoubleComplex *dx, magma_int_t incx,
    magmaDoubleComplex *dtau );

void magmablas_zlascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info );

void magmablas_zlascl2(
    magma_type_t type,
    magma_int_t m, magma_int_t n, const double *dD,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info );

void magmablas_zlaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda );

void magmablas_zlaset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex *A, magma_int_t lda);

void magmablas_zlaswp(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t i1,  magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci );

void magmablas_zlaswpx(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldx, magma_int_t ldy,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci );

void magmablas_zlaswp2(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *d_ipiv, magma_int_t inci );

void magmablas_zsymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda );

void magmablas_zsymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride );

void magmablas_ztrtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex *d_invA);

  /*
   * to cleanup
   */
void magma_zlarfg_gpu(
    magma_int_t n, magmaDoubleComplex *dx0, magmaDoubleComplex *dx,
    magmaDoubleComplex *dtau, double *dxnorm, magmaDoubleComplex *dAkk );

void magma_zlarfgx_gpu(
    magma_int_t n, magmaDoubleComplex *dx0, magmaDoubleComplex *dx,
    magmaDoubleComplex *dtau, double *dxnorm,
    magmaDoubleComplex *ddx0, magma_int_t iter);

void magma_zlarfx_gpu(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *v, magmaDoubleComplex *tau,
    magmaDoubleComplex *c, magma_int_t ldc, double *xnorm,
    magmaDoubleComplex *dT, magma_int_t iter, magmaDoubleComplex *work);

void magma_zlarfbx_gpu(
    magma_int_t m, magma_int_t k, magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *dT, magma_int_t ldt, magmaDoubleComplex *c,
    magmaDoubleComplex *dwork);

void magma_zlarfgtx_gpu(
    magma_int_t n, magmaDoubleComplex *dx0, magmaDoubleComplex *dx,
    magmaDoubleComplex *dtau, double *dxnorm,
    magmaDoubleComplex *dA, magma_int_t it,
    magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *T, magma_int_t ldt,
    magmaDoubleComplex *dwork);

void magmablas_dznrm2_adjust(
    magma_int_t k, double *xnorm, magmaDoubleComplex *c);

void magmablas_dznrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm);

void magmablas_dznrm2_row_check_adjust(
    magma_int_t k, double tol, double *xnorm, double *xnorm2,
    magmaDoubleComplex *c, magma_int_t ldc, double *lsticc);

void magmablas_dznrm2_check(
    magma_int_t m, magma_int_t num, magmaDoubleComplex *da, magma_int_t ldda,
    double *dxnorm, double *lsticc);

  /*
   * Level 1 BLAS
   */
void magmablas_zswap(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb );

void magmablas_zswapblk(
    magma_order_t order,
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset );

void magmablas_zswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t incb );

  /*
   * Level 2 BLAS
   */
void magmablas_zgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

void magmablas_zgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A_array, magma_int_t ldda,
    magmaDoubleComplex **dx_array, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dy_array, magma_int_t incy,
    magma_int_t batchCount);

void magmablas_zgemv_conjv(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t lda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy);

magma_int_t magmablas_zhemv(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

magma_int_t magmablas_zhemv_work(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dX, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dY, magma_int_t incy,
    magmaDoubleComplex_ptr       dwork, magma_int_t lwork );

magma_int_t magmablas_zsymv(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

magma_int_t magmablas_zsymv_work(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dwork, magma_int_t lwork );

  /*
   * Level 3 BLAS
   */
void magmablas_zgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magmablas_zgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *dA, magma_int_t lda,
    const magmaDoubleComplex *dB, magma_int_t ldb,
    magmaDoubleComplex beta,
    magmaDoubleComplex *dC, magma_int_t ldc );

void magmablas_zhemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magmablas_zsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magmablas_zsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magmablas_zherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double  alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    double  beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magmablas_zsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magmablas_zher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    double  beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magmablas_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void magmablas_ztrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       db, magma_int_t lddb,
    magma_int_t flag, magmaDoubleComplex_ptr d_dinvA, magmaDoubleComplex_ptr dx );


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

#define magma_zsetvector(           n, hx_src, incx, dy_dst, incy ) \
        magma_zsetvector_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_zgetvector(           n, dx_src, incx, hy_dst, incy ) \
        magma_zgetvector_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_zcopyvector(          n, dx_src, incx, dy_dst, incy ) \
        magma_zcopyvector_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_zsetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_zsetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_zgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_zgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_zcopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_zcopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

void magma_zsetvector_internal(
    magma_int_t n,
    magmaDoubleComplex const*    hx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_zgetvector_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex*          hy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_zcopyvector_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_zsetvector_async_internal(
    magma_int_t n,
    magmaDoubleComplex const*    hx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_zgetvector_async_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex*          hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_zcopyvector_async_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// copying sub-matrices (contiguous columns)
// set  copies host to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_zsetmatrix(           m, n, hA_src, lda, dB_dst, lddb ) \
        magma_zsetmatrix_internal(  m, n, hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_zgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb ) \
        magma_zgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_zcopymatrix(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_zcopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_zsetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        magma_zsetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_zgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        magma_zgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

#define magma_zcopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_zcopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

void magma_zsetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const*    hA_src, magma_int_t ldha,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_zgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex*          hB_dst, magma_int_t ldhb,
    const char* func, const char* file, int line );

void magma_zcopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_zsetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const*    hA_src, magma_int_t ldha,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_zgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex*          hB_dst, magma_int_t ldhb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_zcopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// Level 1 BLAS

// in cublas_v2, result returned through output argument
magma_int_t magma_izamax(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t magma_izamin(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
double magma_dzasum(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

void magma_zaxpy(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

void magma_zcopy(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaDoubleComplex
magma_zdotc(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaDoubleComplex
magma_zdotu(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
double magma_dznrm2(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx );

void magma_zrot(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double dc, magmaDoubleComplex ds );

void magma_zdrot(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double dc, double ds );

#ifdef REAL
void magma_zrotm(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    magmaDouble_const_ptr param );

void magma_zrotmg(
    magmaDouble_ptr d1, magmaDouble_ptr       d2,
    magmaDouble_ptr x1, magmaDouble_const_ptr y1,
    magmaDouble_ptr param );
#endif

void magma_zscal(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx );

void magma_zdscal(
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx );

void magma_zswap(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy );

// ========================================
// Level 2 BLAS

void magma_zgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

void magma_zgerc(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda );

void magma_zgeru(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda );

void magma_zhemv(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy );

void magma_zher(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda );

void magma_zher2(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda );

void magma_ztrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, magma_int_t incx );

void magma_ztrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, magma_int_t incx );

// ========================================
// Level 3 BLAS

void magma_zgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magma_zsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magma_zsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magma_zsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magma_zhemm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magma_zherk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magma_zher2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc );

void magma_ztrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

void magma_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb );

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif  /* MAGMABLAS_Z_H */
