/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:17 2013
*/

#ifndef MAGMABLAS_S_H
#define MAGMABLAS_S_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Interface to clean
   */
float cpu_gpu_sdiff(
    magma_int_t m, magma_int_t n,
    const float    *hA, magma_int_t lda,
    magmaFloat_const_ptr dA, magma_int_t ldda );

// see also slaset
void szero_32x32_block(
    magmaFloat_ptr dA, magma_int_t ldda );

void szero_nbxnb_block(
    magma_int_t nb,
    magmaFloat_ptr dA, magma_int_t ldda );

// see also slaswp
// ipiv gets updated
void magmablas_spermute_long2(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t nb, magma_int_t ind );

// ipiv is not updated (unlike spermute_long2)
void magmablas_spermute_long3(
    /*magma_int_t n,*/
    magmaFloat_ptr dAT, magma_int_t ldda,
    const magma_int_t *ipiv, magma_int_t nb, magma_int_t ind );

  /*
   * Transpose functions
   */
void magmablas_stranspose_inplace(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda );

void magmablas_stranspose(
    magmaFloat_ptr       odata, magma_int_t ldo,
    magmaFloat_const_ptr idata, magma_int_t ldi,
    magma_int_t m, magma_int_t n );

void magmablas_stranspose2(
    magmaFloat_ptr       odata, magma_int_t ldo,
    magmaFloat_const_ptr idata, magma_int_t ldi,
    magma_int_t m, magma_int_t n );

void magmablas_stranspose2s(
    magmaFloat_ptr       odata, magma_int_t ldo,
    magmaFloat_const_ptr idata, magma_int_t ldi,
    magma_int_t m, magma_int_t n,
    magma_queue_t stream );

void magmablas_sgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dAT,   magma_int_t ldda,
    float          *hA,    magma_int_t lda,
    magmaFloat_ptr       dwork, magma_int_t lddwork, magma_int_t nb );

void magmablas_ssetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const float *hA,    magma_int_t lda,
    magmaFloat_ptr    dAT,   magma_int_t ldda,
    magmaFloat_ptr    dwork, magma_int_t lddwork, magma_int_t nb );

  /*
   * Multi-GPU functions
   */
void magmablas_sgetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t stream[][2],
    magmaFloat_ptr  dAT[], magma_int_t ldda,
    float     *hA,    magma_int_t lda,
    magmaFloat_ptr  dB[],  magma_int_t lddb,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void magmablas_ssetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t stream[][2],
    const float *hA,    magma_int_t lda,
    magmaFloat_ptr    dAT[], magma_int_t ldda,
    magmaFloat_ptr    dB[],  magma_int_t lddb,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void magma_sgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA[], magma_int_t ldda,
    float    *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void magma_ssetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const float *hA,   magma_int_t lda,
    magmaFloat_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

void magma_sgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA[], magma_int_t ldda,
    float    *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void magma_ssetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const float *hA,   magma_int_t lda,
    magmaFloat_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

// in src/ssytrd_mgpu.cpp
magma_int_t magma_shtodhe(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    float *a, magma_int_t lda,
    float **dwork, magma_int_t ldda,
    magma_queue_t stream[][10], magma_int_t *info );

// in src/spotrf3_mgpu.cpp
magma_int_t magma_shtodpo(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    float  *h_A,   magma_int_t lda,
    float *d_lA[], magma_int_t ldda,
    magma_queue_t stream[][3], magma_int_t *info );

// in src/spotrf3_mgpu.cpp
magma_int_t magma_sdtohpo(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
    float  *a,     magma_int_t lda,
    float *work[], magma_int_t ldda,
    magma_queue_t stream[][3], magma_int_t *info );

magma_int_t magmablas_ssymv_mgpu_offset(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    float **A, magma_int_t lda,
    float **X, magma_int_t incx,
    float beta,
    float **Y, magma_int_t incy,
    float **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_queue_t stream[][10] );

magma_int_t magmablas_ssymv_mgpu_32_offset(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    float **A, magma_int_t lda,
    float **X, magma_int_t incx,
    float beta,
    float **Y, magma_int_t incy,
    float **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_queue_t stream[][10] );

magma_int_t magmablas_ssymv2_mgpu_offset(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    float **A, magma_int_t lda,
    float **x, magma_int_t incx,
    float beta,
    float **y, magma_int_t incy,
    float **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset);

magma_int_t magmablas_ssymv2_mgpu_32_offset(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    float **A, magma_int_t lda,
    float **x, magma_int_t incx,
    float beta,
    float **y, magma_int_t incy,
    float **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset);

magma_int_t magmablas_ssymv_mgpu(
    magma_int_t num_gpus, magma_int_t k, magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb,
    float alpha,
    float **da, magma_int_t ldda, magma_int_t offset,
    float **dx, magma_int_t incx,
    float beta,
    float **dy, magma_int_t incy,
    float **dwork, magma_int_t ldwork,
    float *work, float *w,
    magma_queue_t stream[][10] );

magma_int_t magmablas_ssymv_sync(
    magma_int_t num_gpus, magma_int_t k,
    magma_int_t n, float *work, float *w,
    magma_queue_t stream[][10] );

void magmablas_ssymm_1gpu_old(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dA[], magma_int_t ldda,  magma_int_t offset,
    magmaFloat_ptr dB[], magma_int_t lddb,
    float beta,
    magmaFloat_ptr dC[], magma_int_t lddc,
    float*    C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_ssymm_1gpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dA[], magma_int_t ldda,  magma_int_t offset,
    magmaFloat_ptr dB[], magma_int_t lddb,
    float beta,
    magmaFloat_ptr dC[], magma_int_t lddc,
    float*    C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_ssymm_mgpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloat_ptr dB[],    magma_int_t lddb,
    float beta,
    magmaFloat_ptr dC[],    magma_int_t lddc,
    magmaFloat_ptr dwork[], magma_int_t lddwork,
    float*    C,       magma_int_t ldc,
    float*    work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][20], magma_int_t nbevents );

void magmablas_ssymm_mgpu_com(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloat_ptr dB[],    magma_int_t lddb,
    float beta,
    magmaFloat_ptr dC[],    magma_int_t lddc,
    magmaFloat_ptr dwork[], magma_int_t lddwork,
    float*    C,       magma_int_t ldc,
    float*    work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void magmablas_ssymm_mgpu_spec(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloat_ptr dB[],    magma_int_t lddb,
    float beta,
    magmaFloat_ptr dC[],    magma_int_t lddc,
    magmaFloat_ptr dwork[], magma_int_t lddwork,
    float*    C,       magma_int_t ldc,
    float*    work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void magmablas_ssymm_mgpu_spec33(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaFloat_ptr dB[],    magma_int_t lddb,
    float beta,
    magmaFloat_ptr dC[],    magma_int_t lddc,
    magmaFloat_ptr dVIN[],  magma_int_t lddv, magma_int_t voffst,
    magmaFloat_ptr dwork[], magma_int_t lddwork,
    float *C,       magma_int_t ldc,
    float *work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

// Ichi's version, in src/ssytrd_mgpu.cpp
void magma_ssyr2k_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    float alpha,
    float **db, magma_int_t lddb, magma_int_t boffset,
    float beta,
    float **dc, magma_int_t lddc, magma_int_t offset,
    magma_int_t num_streams, magma_queue_t streams[][10] );

void magmablas_ssyr2k_mgpu2(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_ptr dA[], magma_int_t ldda, magma_int_t aoff,
    magmaFloat_ptr dB[], magma_int_t lddb, magma_int_t boff,
    float beta,
    magmaFloat_ptr dC[], magma_int_t lddc, magma_int_t offset,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_ssyr2k_mgpu_spec(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_ptr dA[], magma_int_t lda, magma_int_t aoff,
    magmaFloat_ptr dB[], magma_int_t ldb, magma_int_t boff,
    float beta,
    magmaFloat_ptr dC[], magma_int_t ldc,  magma_int_t offset,
    magma_int_t ngpu, magma_int_t nb, magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_ssyr2k_mgpu_spec324(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dVIN[], magma_int_t lddv, magma_int_t voff,
    magmaFloat_ptr dWIN[], magma_int_t lddw, magma_int_t woff,
    float beta,
    magmaFloat_ptr dC[], magma_int_t lddc,  magma_int_t offset,
    magmaFloat_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

void magmablas_ssyr2k_mgpu_spec325(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_ptr dVIN[], magma_int_t lddv, magma_int_t voff,
    magmaFloat_ptr dWIN[], magma_int_t lddw, magma_int_t woff,
    float beta,
    magmaFloat_ptr dC[], magma_int_t lddc,  magma_int_t offset,
    magmaFloat_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    float **harray[],
    magmaFloat_ptr *darray[],
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

  /*
   * LAPACK auxiliary functions
   */
void magmablas_sgeadd(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

void magmablas_sgeadd_batched(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr  const *dAarray, magma_int_t ldda,
    magmaFloat_ptr              *dBarray, magma_int_t lddb,
    magma_int_t batchCount );

void magmablas_slacpy(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

void magmablas_slacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr  const *dAarray, magma_int_t ldda,
    magmaFloat_ptr              *dBarray, magma_int_t lddb,
    magma_int_t batchCount );

float magmablas_slange(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork );

float magmablas_slansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork );

float magmablas_slansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork );

void magmablas_slascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *info );

void magmablas_slaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda );

void magmablas_slaset_identity(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda );

void magmablas_slaswp(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t i1,  magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci );

void magmablas_slaswpx(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldx, magma_int_t ldy,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci );

void magmablas_slaswp2(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *d_ipiv );

void magmablas_ssymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda );

void magmablas_ssymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride );

void magma_slarfgx_gpu(
    magma_int_t n, float *dx0, float *dx,
    float *dtau, float *dxnorm,
    float *ddx0, magma_int_t iter);

void magma_slarfx_gpu(
    magma_int_t m, magma_int_t n, float *v, float *tau,
    float *c, magma_int_t ldc, float *xnorm,
    float *dT, magma_int_t iter, float *work);

void magma_slarfbx_gpu(
    magma_int_t m, magma_int_t k, float *V, magma_int_t ldv,
    float *dT, magma_int_t ldt, float *c,
    float *dwork);

void magma_slarfgtx_gpu(
    magma_int_t n, float *dx0, float *dx,
    float *dtau, float *dxnorm,
    float *dA, magma_int_t it,
    float *V, magma_int_t ldv, float *T, magma_int_t ldt,
    float *dwork);

void magmablas_snrm2_adjust(
    magma_int_t k, float *xnorm, float *c);

void magmablas_snrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm);

void magmablas_snrm2_row_check_adjust(
    magma_int_t k, float tol, float *xnorm, float *xnorm2,
    float *c, magma_int_t ldc, float *lsticc);

void magmablas_snrm2_check(
    magma_int_t m, magma_int_t num, float *da, magma_int_t ldda,
    float *dxnorm, float *lsticc);

  /*
   * Level 1 BLAS
   */
void magmablas_sswap(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb );

void magmablas_sswapblk(
    magma_storev_t storev,
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset );

void magmablas_sswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloat_ptr dB, magma_int_t lddb, magma_int_t incb );

  /*
   * Level 2 BLAS
   */
void magmablas_sgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

magma_int_t magmablas_ssymv(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

magma_int_t magmablas_ssymv_work(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dX, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dY, magma_int_t incy,
    magmaFloat_ptr       dwork, magma_int_t lwork );

magma_int_t magmablas_ssymv(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

magma_int_t magmablas_ssymv_work(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy,
    magmaFloat_ptr       dwork, magma_int_t lwork );

  /*
   * Level 3 BLAS
   */
void magmablas_sgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magmablas_sgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    const float *dA, magma_int_t lda,
    const float *dB, magma_int_t ldb,
    float beta,
    float *dC, magma_int_t ldc );

void magmablas_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magmablas_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magmablas_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magmablas_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float  alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float  beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magmablas_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magmablas_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float  beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

#ifdef REAL
// only real [sd] precisions available
void magmablas_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

void magmablas_strsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       db, magma_int_t lddb,
    int flag, magmaFloat_ptr d_dinvA, magmaFloat_ptr dx );
#endif

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

#define magma_ssetvector(           n, hx_src, incx, dy_dst, incy ) \
        magma_ssetvector_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_sgetvector(           n, dx_src, incx, hy_dst, incy ) \
        magma_sgetvector_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_ssetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_ssetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_sgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_sgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_scopyvector_async(           n, dx_src, incx, dy_dst, incy, queue ) \
        magma_scopyvector_async_internal(  n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_scopyvector_async(           n, dx_src, incx, dy_dst, incy, queue ) \
        magma_scopyvector_async_internal(  n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

void magma_ssetvector_internal(
    magma_int_t n,
    float const*    hx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_sgetvector_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    float*          hy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

magma_err_t
magma_scopyvector_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_ssetvector_async_internal(
    magma_int_t n,
    float const*    hx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_sgetvector_async_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    float*          hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

magma_err_t
magma_scopyvector_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// copying sub-matrices (contiguous columns)
// set  copies host to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_ssetmatrix(           m, n, hA_src, lda, dB_dst, lddb ) \
        magma_ssetmatrix_internal(  m, n, hA_src, lda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_sgetmatrix(           m, n, dA_src, ldda, hB_dst, ldb ) \
        magma_sgetmatrix_internal(  m, n, dA_src, ldda, hB_dst, ldb, __func__, __FILE__, __LINE__ )

#define magma_scopymatrix(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_scopymatrix_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_ssetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        magma_ssetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_sgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        magma_sgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

#define magma_scopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_scopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

void magma_ssetmatrix_internal(
    magma_int_t m, magma_int_t n,
    float const*    hA_src, magma_int_t ldha,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_sgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    float*          hB_dst, magma_int_t ldhb,
    const char* func, const char* file, int line );

void magma_scopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_ssetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    float const*    hA_src, magma_int_t ldha,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_sgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    float*          hB_dst, magma_int_t ldhb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_scopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// Level 1 BLAS

// in cublas_v2, result returned through output argument
magma_int_t magma_isamax(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t magma_isamin(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
float magma_sasum(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

void magma_saxpy(
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy );

void magma_scopy(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float
magma_sdot(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float
magma_sdotu(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float magma_snrm2(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

void magma_srot(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    float dc, float ds );

void magma_srot(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    float dc, float ds );

#ifdef REAL
void magma_srotm(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magmaFloat_const_ptr param );

void magma_srotmg(
    magmaFloat_ptr d1, magmaFloat_ptr       d2,
    magmaFloat_ptr x1, magmaFloat_const_ptr y1,
    magmaFloat_ptr param );
#endif

void magma_sscal(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, magma_int_t incx );

void magma_sscal(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, magma_int_t incx );

void magma_sswap(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy );

// ========================================
// Level 2 BLAS

void magma_sgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

void magma_sger(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda );

void magma_sger(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda );

void magma_ssymv(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

void magma_ssyr(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dA, magma_int_t ldda );

void magma_ssyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda );

void magma_strmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dx, magma_int_t incx );

void magma_strsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dx, magma_int_t incx );

// ========================================
// Level 3 BLAS

void magma_sgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magma_ssymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magma_ssyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magma_ssyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void magma_strmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

void magma_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

#ifdef __cplusplus
}
#endif

#undef REAL

#endif  /* MAGMABLAS_S_H */
