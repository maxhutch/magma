/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:17 2013
*/

#ifndef MAGMABLAS_D_H
#define MAGMABLAS_D_H

#include "magma_types.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Interface to clean
   */
double cpu_gpu_ddiff(
    magma_int_t m, magma_int_t n,
    const double    *hA, magma_int_t lda,
    magmaDouble_const_ptr dA, magma_int_t ldda );

// see also dlaset
void dzero_32x32_block(
    magmaDouble_ptr dA, magma_int_t ldda );

void dzero_nbxnb_block(
    magma_int_t nb,
    magmaDouble_ptr dA, magma_int_t ldda );

// see also dlaswp
// ipiv gets updated
void magmablas_dpermute_long2(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t nb, magma_int_t ind );

// ipiv is not updated (unlike dpermute_long2)
void magmablas_dpermute_long3(
    /*magma_int_t n,*/
    magmaDouble_ptr dAT, magma_int_t ldda,
    const magma_int_t *ipiv, magma_int_t nb, magma_int_t ind );

  /*
   * Transpose functions
   */
void magmablas_dtranspose_inplace(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda );

void magmablas_dtranspose(
    magmaDouble_ptr       odata, magma_int_t ldo,
    magmaDouble_const_ptr idata, magma_int_t ldi,
    magma_int_t m, magma_int_t n );

void magmablas_dtranspose2(
    magmaDouble_ptr       odata, magma_int_t ldo,
    magmaDouble_const_ptr idata, magma_int_t ldi,
    magma_int_t m, magma_int_t n );

void magmablas_dtranspose2s(
    magmaDouble_ptr       odata, magma_int_t ldo,
    magmaDouble_const_ptr idata, magma_int_t ldi,
    magma_int_t m, magma_int_t n,
    magma_queue_t stream );

void magmablas_dgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dAT,   magma_int_t ldda,
    double          *hA,    magma_int_t lda,
    magmaDouble_ptr       dwork, magma_int_t lddwork, magma_int_t nb );

void magmablas_dsetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const double *hA,    magma_int_t lda,
    magmaDouble_ptr    dAT,   magma_int_t ldda,
    magmaDouble_ptr    dwork, magma_int_t lddwork, magma_int_t nb );

  /*
   * Multi-GPU functions
   */
void magmablas_dgetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t stream[][2],
    magmaDouble_ptr  dAT[], magma_int_t ldda,
    double     *hA,    magma_int_t lda,
    magmaDouble_ptr  dB[],  magma_int_t lddb,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void magmablas_dsetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t stream[][2],
    const double *hA,    magma_int_t lda,
    magmaDouble_ptr    dAT[], magma_int_t ldda,
    magmaDouble_ptr    dB[],  magma_int_t lddb,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void magma_dgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA[], magma_int_t ldda,
    double    *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void magma_dsetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const double *hA,   magma_int_t lda,
    magmaDouble_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

void magma_dgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA[], magma_int_t ldda,
    double    *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void magma_dsetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const double *hA,   magma_int_t lda,
    magmaDouble_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

// in src/dsytrd_mgpu.cpp
magma_int_t magma_dhtodhe(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    double *a, magma_int_t lda,
    double **dwork, magma_int_t ldda,
    magma_queue_t stream[][10], magma_int_t *info );

// in src/dpotrf3_mgpu.cpp
magma_int_t magma_dhtodpo(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    double  *h_A,   magma_int_t lda,
    double *d_lA[], magma_int_t ldda,
    magma_queue_t stream[][3], magma_int_t *info );

// in src/dpotrf3_mgpu.cpp
magma_int_t magma_ddtohpo(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
    double  *a,     magma_int_t lda,
    double *work[], magma_int_t ldda,
    magma_queue_t stream[][3], magma_int_t *info );

magma_int_t magmablas_dsymv_mgpu_offset(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    double **A, magma_int_t lda,
    double **X, magma_int_t incx,
    double beta,
    double **Y, magma_int_t incy,
    double **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_queue_t stream[][10] );

magma_int_t magmablas_dsymv_mgpu_32_offset(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    double **A, magma_int_t lda,
    double **X, magma_int_t incx,
    double beta,
    double **Y, magma_int_t incy,
    double **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_queue_t stream[][10] );

magma_int_t magmablas_dsymv2_mgpu_offset(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    double **A, magma_int_t lda,
    double **x, magma_int_t incx,
    double beta,
    double **y, magma_int_t incy,
    double **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset);

magma_int_t magmablas_dsymv2_mgpu_32_offset(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    double **A, magma_int_t lda,
    double **x, magma_int_t incx,
    double beta,
    double **y, magma_int_t incy,
    double **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset);

magma_int_t magmablas_dsymv_mgpu(
    magma_int_t num_gpus, magma_int_t k, magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb,
    double alpha,
    double **da, magma_int_t ldda, magma_int_t offset,
    double **dx, magma_int_t incx,
    double beta,
    double **dy, magma_int_t incy,
    double **dwork, magma_int_t ldwork,
    double *work, double *w,
    magma_queue_t stream[][10] );

magma_int_t magmablas_dsymv_sync(
    magma_int_t num_gpus, magma_int_t k,
    magma_int_t n, double *work, double *w,
    magma_queue_t stream[][10] );

void magmablas_dsymm_1gpu_old(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[], magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[], magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc,
    double*    C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_dsymm_1gpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[], magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[], magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc,
    double*    C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_dsymm_mgpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[],    magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[],    magma_int_t lddc,
    magmaDouble_ptr dwork[], magma_int_t lddwork,
    double*    C,       magma_int_t ldc,
    double*    work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][20], magma_int_t nbevents );

void magmablas_dsymm_mgpu_com(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[],    magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[],    magma_int_t lddc,
    magmaDouble_ptr dwork[], magma_int_t lddwork,
    double*    C,       magma_int_t ldc,
    double*    work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void magmablas_dsymm_mgpu_spec(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[],    magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[],    magma_int_t lddc,
    magmaDouble_ptr dwork[], magma_int_t lddwork,
    double*    C,       magma_int_t ldc,
    double*    work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void magmablas_dsymm_mgpu_spec33(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDouble_ptr dB[],    magma_int_t lddb,
    double beta,
    magmaDouble_ptr dC[],    magma_int_t lddc,
    magmaDouble_ptr dVIN[],  magma_int_t lddv, magma_int_t voffst,
    magmaDouble_ptr dwork[], magma_int_t lddwork,
    double *C,       magma_int_t ldc,
    double *work[],  magma_int_t ldwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

// Ichi's version, in src/dsytrd_mgpu.cpp
void magma_dsyr2k_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    double **db, magma_int_t lddb, magma_int_t boffset,
    double beta,
    double **dc, magma_int_t lddc, magma_int_t offset,
    magma_int_t num_streams, magma_queue_t streams[][10] );

void magmablas_dsyr2k_mgpu2(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_ptr dA[], magma_int_t ldda, magma_int_t aoff,
    magmaDouble_ptr dB[], magma_int_t lddb, magma_int_t boff,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc, magma_int_t offset,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_dsyr2k_mgpu_spec(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_ptr dA[], magma_int_t lda, magma_int_t aoff,
    magmaDouble_ptr dB[], magma_int_t ldb, magma_int_t boff,
    double beta,
    magmaDouble_ptr dC[], magma_int_t ldc,  magma_int_t offset,
    magma_int_t ngpu, magma_int_t nb, magma_queue_t streams[][20], magma_int_t nstream );

void magmablas_dsyr2k_mgpu_spec324(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dVIN[], magma_int_t lddv, magma_int_t voff,
    magmaDouble_ptr dWIN[], magma_int_t lddw, magma_int_t woff,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc,  magma_int_t offset,
    magmaDouble_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

void magmablas_dsyr2k_mgpu_spec325(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_ptr dVIN[], magma_int_t lddv, magma_int_t voff,
    magmaDouble_ptr dWIN[], magma_int_t lddw, magma_int_t woff,
    double beta,
    magmaDouble_ptr dC[], magma_int_t lddc,  magma_int_t offset,
    magmaDouble_ptr dwork[], magma_int_t lndwork,
    magma_int_t ngpu, magma_int_t nb,
    double **harray[],
    magmaDouble_ptr *darray[],
    magma_queue_t streams[][20], magma_int_t nstream,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents );

  /*
   * LAPACK auxiliary functions
   */
void magmablas_dgeadd(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb );

void magmablas_dgeadd_batched(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr  const *dAarray, magma_int_t ldda,
    magmaDouble_ptr              *dBarray, magma_int_t lddb,
    magma_int_t batchCount );

void magmablas_dlacpy(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb );

void magmablas_dlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr  const *dAarray, magma_int_t ldda,
    magmaDouble_ptr              *dBarray, magma_int_t lddb,
    magma_int_t batchCount );

double magmablas_dlange(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork );

double magmablas_dlansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork );

double magmablas_dlansy(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork );

void magmablas_dlascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *info );

void magmablas_dlaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda );

void magmablas_dlaset_identity(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda );

void magmablas_dlaswp(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldda,
    magma_int_t i1,  magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci );

void magmablas_dlaswpx(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldx, magma_int_t ldy,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci );

void magmablas_dlaswp2(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldda,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *d_ipiv );

void magmablas_dsymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaDouble_ptr dA, magma_int_t ldda );

void magmablas_dsymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride );

void magma_dlarfgx_gpu(
    magma_int_t n, double *dx0, double *dx,
    double *dtau, double *dxnorm,
    double *ddx0, magma_int_t iter);

void magma_dlarfx_gpu(
    magma_int_t m, magma_int_t n, double *v, double *tau,
    double *c, magma_int_t ldc, double *xnorm,
    double *dT, magma_int_t iter, double *work);

void magma_dlarfbx_gpu(
    magma_int_t m, magma_int_t k, double *V, magma_int_t ldv,
    double *dT, magma_int_t ldt, double *c,
    double *dwork);

void magma_dlarfgtx_gpu(
    magma_int_t n, double *dx0, double *dx,
    double *dtau, double *dxnorm,
    double *dA, magma_int_t it,
    double *V, magma_int_t ldv, double *T, magma_int_t ldt,
    double *dwork);

void magmablas_dnrm2_adjust(
    magma_int_t k, double *xnorm, double *c);

void magmablas_dnrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm);

void magmablas_dnrm2_row_check_adjust(
    magma_int_t k, double tol, double *xnorm, double *xnorm2,
    double *c, magma_int_t ldc, double *lsticc);

void magmablas_dnrm2_check(
    magma_int_t m, magma_int_t num, double *da, magma_int_t ldda,
    double *dxnorm, double *lsticc);

  /*
   * Level 1 BLAS
   */
void magmablas_dswap(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb );

void magmablas_dswapblk(
    magma_storev_t storev,
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset );

void magmablas_dswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaDouble_ptr dB, magma_int_t lddb, magma_int_t incb );

  /*
   * Level 2 BLAS
   */
void magmablas_dgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );

magma_int_t magmablas_dsymv(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );

magma_int_t magmablas_dsymv_work(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dX, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dY, magma_int_t incy,
    magmaDouble_ptr       dwork, magma_int_t lwork );

magma_int_t magmablas_dsymv(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );

magma_int_t magmablas_dsymv_work(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy,
    magmaDouble_ptr       dwork, magma_int_t lwork );

  /*
   * Level 3 BLAS
   */
void magmablas_dgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magmablas_dgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    const double *dA, magma_int_t lda,
    const double *dB, magma_int_t ldb,
    double beta,
    double *dC, magma_int_t ldc );

void magmablas_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magmablas_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magmablas_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magmablas_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double  alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double  beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magmablas_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magmablas_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double  beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

#ifdef REAL
// only real [sd] precisions available
void magmablas_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb );

void magmablas_dtrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       db, magma_int_t lddb,
    int flag, magmaDouble_ptr d_dinvA, magmaDouble_ptr dx );
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

#define magma_dsetvector(           n, hx_src, incx, dy_dst, incy ) \
        magma_dsetvector_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_dgetvector(           n, dx_src, incx, hy_dst, incy ) \
        magma_dgetvector_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_dsetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_dsetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_dgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_dgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_dcopyvector_async(           n, dx_src, incx, dy_dst, incy, queue ) \
        magma_dcopyvector_async_internal(  n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_dcopyvector_async(           n, dx_src, incx, dy_dst, incy, queue ) \
        magma_dcopyvector_async_internal(  n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

void magma_dsetvector_internal(
    magma_int_t n,
    double const*    hx_src, magma_int_t incx,
    magmaDouble_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_dgetvector_internal(
    magma_int_t n,
    magmaDouble_const_ptr dx_src, magma_int_t incx,
    double*          hy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

magma_err_t
magma_dcopyvector_internal(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dx_src, magma_int_t incx,
    magmaDouble_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line );

void magma_dsetvector_async_internal(
    magma_int_t n,
    double const*    hx_src, magma_int_t incx,
    magmaDouble_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_dgetvector_async_internal(
    magma_int_t n,
    magmaDouble_const_ptr dx_src, magma_int_t incx,
    double*          hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line );

magma_err_t
magma_dcopyvector_async_internal(
    magma_int_t m, magma_int_t n,
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

void magma_dsetmatrix_internal(
    magma_int_t m, magma_int_t n,
    double const*    hA_src, magma_int_t ldha,
    magmaDouble_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_dgetmatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, magma_int_t ldda,
    double*          hB_dst, magma_int_t ldhb,
    const char* func, const char* file, int line );

void magma_dcopymatrix_internal(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, magma_int_t ldda,
    magmaDouble_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line );

void magma_dsetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    double const*    hA_src, magma_int_t ldha,
    magmaDouble_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_dgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, magma_int_t ldda,
    double*          hB_dst, magma_int_t ldhb,
    magma_queue_t queue,
    const char* func, const char* file, int line );

void magma_dcopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA_src, magma_int_t ldda,
    magmaDouble_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line );


// ========================================
// Level 1 BLAS

// in cublas_v2, result returned through output argument
magma_int_t magma_idamax(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t magma_idamin(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
double magma_dasum(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

void magma_daxpy(
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dy, magma_int_t incy );

void magma_dcopy(
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
magma_ddotu(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
double magma_dnrm2(
    magma_int_t n,
    magmaDouble_const_ptr dx, magma_int_t incx );

void magma_drot(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    double dc, double ds );

void magma_drot(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    double dc, double ds );

#ifdef REAL
void magma_drotm(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    magmaDouble_const_ptr param );

void magma_drotmg(
    magmaDouble_ptr d1, magmaDouble_ptr       d2,
    magmaDouble_ptr x1, magmaDouble_const_ptr y1,
    magmaDouble_ptr param );
#endif

void magma_dscal(
    magma_int_t n,
    double alpha,
    magmaDouble_ptr dx, magma_int_t incx );

void magma_dscal(
    magma_int_t n,
    double alpha,
    magmaDouble_ptr dx, magma_int_t incx );

void magma_dswap(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy );

// ========================================
// Level 2 BLAS

void magma_dgemv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );

void magma_dger(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda );

void magma_dger(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda );

void magma_dsymv(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy );

void magma_dsyr(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_ptr       dA, magma_int_t ldda );

void magma_dsyr2(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dx, magma_int_t incx,
    magmaDouble_const_ptr dy, magma_int_t incy,
    magmaDouble_ptr       dA, magma_int_t ldda );

void magma_dtrmv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dx, magma_int_t incx );

void magma_dtrsv(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dx, magma_int_t incx );

// ========================================
// Level 3 BLAS

void magma_dgemm(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magma_dsymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magma_dsyrk(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magma_dsyr2k(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc );

void magma_dtrmm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb );

void magma_dtrsm(
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
