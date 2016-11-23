/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magmablas_z_v1.h, normal z -> s, Sun Nov 20 20:20:47 2016
*/

#ifndef MAGMABLAS_S_V1_H
#define MAGMABLAS_S_V1_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

#include "magma_types.h"
#include "magma_copy_v1.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
magmablas_stranspose_inplace_v1(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda );

void
magmablas_stranspose_inplace_v1(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda );

void
magmablas_stranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat );

void
magmablas_stranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat );

void
magmablas_sgetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dAT,   magma_int_t ldda,
    float          *hA,    magma_int_t lda,
    magmaFloat_ptr       dwork, magma_int_t lddwork, magma_int_t nb );

void
magmablas_ssetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    const float *hA,    magma_int_t lda,
    magmaFloat_ptr    dAT,   magma_int_t ldda,
    magmaFloat_ptr    dwork, magma_int_t lddwork, magma_int_t nb );

  /*
   * RBT-related functions
   */
void
magmablas_sprbt_v1(
    magma_int_t n, 
    magmaFloat_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr du,
    magmaFloat_ptr dv );

void
magmablas_sprbt_mv_v1(
    magma_int_t n, 
    magmaFloat_ptr dv,
    magmaFloat_ptr db );

void
magmablas_sprbt_mtv_v1(
    magma_int_t n, 
    magmaFloat_ptr du,
    magmaFloat_ptr db );

  /*
   * Multi-GPU copy functions
   */
void
magma_sgetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr const dA[], magma_int_t ldda,
    float                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_ssetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const float *hA,   magma_int_t lda,
    magmaFloat_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_sgetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr const dA[], magma_int_t ldda,
    float                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_ssetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const float *hA,   magma_int_t lda,
    magmaFloat_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
magmablas_sgeadd_v1(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

void
magmablas_sgeadd2_v1(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dB, magma_int_t lddb );

void
magmablas_slacpy_v1(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

void
magmablas_slacpy_conj_v1(
    magma_int_t n,
    magmaFloat_ptr dA1, magma_int_t lda1,
    magmaFloat_ptr dA2, magma_int_t lda2 );

void
magmablas_slacpy_sym_in_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

void
magmablas_slacpy_sym_out_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

float
magmablas_slange_v1(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork );

float
magmablas_slansy_v1(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork );

float
magmablas_slansy_v1(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork );

void
magmablas_slarfg_v1(
    magma_int_t n,
    magmaFloat_ptr dalpha,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dtau );

void
magmablas_slascl_v1(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_slascl_2x2_v1(
    magma_type_t type, magma_int_t m,
    magmaFloat_const_ptr dW, magma_int_t lddw,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_slascl2_v1(
    magma_type_t type,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_slascl_diag_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD, magma_int_t lddd,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_slaset_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda );

void
magmablas_slaset_band_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda );

void
magmablas_slaswp_v1(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_slaswp2_v1(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci );

void
magmablas_slaswp_sym_v1(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_slaswpx_v1(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_ssymmetrize_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda );

void
magmablas_ssymmetrize_tiles_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride );

void
magmablas_strtri_diag_v1(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr d_dinvA );

  /*
   * to cleanup (alphabetical order)
   */
void
magmablas_snrm2_adjust_v1(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dc );

void
magmablas_snrm2_check_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc );

void
magmablas_snrm2_cols_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm );

void
magmablas_snrm2_row_check_adjust_v1(
    magma_int_t k, float tol,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dxnorm2,
    magmaFloat_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dlsticc );

magma_int_t
magma_slarfb_gpu_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV, magma_int_t lddv,
    magmaFloat_const_ptr dT, magma_int_t lddt,
    magmaFloat_ptr dC,       magma_int_t lddc,
    magmaFloat_ptr dwork,    magma_int_t ldwork );

magma_int_t
magma_slarfb_gpu_gemm_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV, magma_int_t lddv,
    magmaFloat_const_ptr dT, magma_int_t lddt,
    magmaFloat_ptr dC,       magma_int_t lddc,
    magmaFloat_ptr dwork,    magma_int_t ldwork,
    magmaFloat_ptr dworkvt,  magma_int_t ldworkvt );

void
magma_slarfbx_gpu_v1(
    magma_int_t m, magma_int_t k,
    magmaFloat_ptr V,  magma_int_t ldv,
    magmaFloat_ptr dT, magma_int_t ldt,
    magmaFloat_ptr c,
    magmaFloat_ptr dwork );

void
magma_slarfg_gpu_v1(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dAkk );

void
magma_slarfgtx_gpu_v1(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dA, magma_int_t iter,
    magmaFloat_ptr V,  magma_int_t ldv,
    magmaFloat_ptr T,  magma_int_t ldt,
    magmaFloat_ptr dwork );

void
magma_slarfgx_gpu_v1(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dA, magma_int_t iter );

void
magma_slarfx_gpu_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr v,
    magmaFloat_ptr tau,
    magmaFloat_ptr C,  magma_int_t ldc,
    magmaFloat_ptr        xnorm,
    magmaFloat_ptr dT, magma_int_t iter,
    magmaFloat_ptr work );


  /*
   * Level 1 BLAS (alphabetical order)
   */
void
magmablas_saxpycp_v1(
    magma_int_t m,
    magmaFloat_ptr dr,
    magmaFloat_ptr dx,
    magmaFloat_const_ptr db );

void
magmablas_sswap_v1(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy );

void
magmablas_sswapblk_v1(
    magma_order_t order,
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset );

void
magmablas_sswapdblk_v1(
    magma_int_t n, magma_int_t nb,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloat_ptr dB, magma_int_t lddb, magma_int_t incb );

  /*
   * Level 2 BLAS (alphabetical order)
   */
void
magmablas_sgemv_v1(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

void
magmablas_sgemv_conj_v1(
    magma_int_t m, magma_int_t n, float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy );

magma_int_t
magmablas_ssymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

magma_int_t
magmablas_ssymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
magmablas_sgemm_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magmablas_sgemm_reduce_v1(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magmablas_ssymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magmablas_ssymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magmablas_ssyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magmablas_ssyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float  beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magmablas_ssyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magmablas_ssyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float  alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float  beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magmablas_strsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

void
magmablas_strsm_outofplace_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magmaFloat_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloat_ptr d_dinvA, magma_int_t dinvA_length );

void
magmablas_strsm_work_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magmaFloat_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloat_ptr d_dinvA, magma_int_t dinvA_length );


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

#define magma_ssetvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        magma_ssetvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_sgetvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        magma_sgetvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_scopyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        magma_scopyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
magma_ssetvector_v1_internal(
    magma_int_t n,
    float const    *hx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_setvector_v1_internal( n, sizeof(float),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
magma_sgetvector_v1_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    float          *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_getvector_v1_internal( n, sizeof(float),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
magma_scopyvector_v1_internal(
    magma_int_t n,
    magmaFloat_const_ptr dx_src, magma_int_t incx,
    magmaFloat_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_copyvector_v1_internal( n, sizeof(float),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

#define magma_ssetmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        magma_ssetmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_sgetmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb  ) \
        magma_sgetmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb,  __func__, __FILE__, __LINE__ )

#define magma_scopymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_scopymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
magma_ssetmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    float const    *hA_src, magma_int_t lda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_setmatrix_v1_internal( m, n, sizeof(float),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
magma_sgetmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    float          *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    magma_getmatrix_v1_internal( m, n, sizeof(float),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
magma_scopymatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA_src, magma_int_t ldda,
    magmaFloat_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_copymatrix_v1_internal( m, n, sizeof(float),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

// in cublas_v2, result returned through output argument
magma_int_t
magma_isamax_v1(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t
magma_isamin_v1(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
float
magma_sasum_v1(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

void
magma_saxpy_v1(
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy );

void
magma_scopy_v1(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float
magma_sdot_v1(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float
magma_sdot_v1(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float
magma_snrm2_v1(
    magma_int_t n,
    magmaFloat_const_ptr dx, magma_int_t incx );

void
magma_srot_v1(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    float dc, float ds );

void
magma_srot_v1(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    float dc, float ds );

#ifdef REAL
void
magma_srotm_v1(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magmaFloat_const_ptr param );

void
magma_srotmg_v1(
    magmaFloat_ptr d1, magmaFloat_ptr       d2,
    magmaFloat_ptr x1, magmaFloat_const_ptr y1,
    magmaFloat_ptr param );
#endif

void
magma_sscal_v1(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, magma_int_t incx );

void
magma_sscal_v1(
    magma_int_t n,
    float alpha,
    magmaFloat_ptr dx, magma_int_t incx );

void
magma_sswap_v1(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy );

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
magma_sgemv_v1(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

void
magma_sger_v1(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda );

void
magma_sger_v1(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda );

void
magma_ssymv_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy );

void
magma_ssyr_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_ptr       dA, magma_int_t ldda );

void
magma_ssyr2_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dx, magma_int_t incx,
    magmaFloat_const_ptr dy, magma_int_t incy,
    magmaFloat_ptr       dA, magma_int_t ldda );

void
magma_strmv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dx, magma_int_t incx );

void
magma_strsv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dx, magma_int_t incx );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
magma_sgemm_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magma_ssymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magma_ssymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magma_ssyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magma_ssyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magma_ssyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magma_ssyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc );

void
magma_strmm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );

void
magma_strsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb );


#ifdef __cplusplus
}
#endif

#undef REAL

#endif // MAGMABLAS_S_H
