/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magmablas_z_v1.h, normal z -> c, Sun Nov 20 20:20:47 2016
*/

#ifndef MAGMABLAS_C_V1_H
#define MAGMABLAS_C_V1_H

#ifdef MAGMA_NO_V1
#error "Since MAGMA_NO_V1 is defined, magma.h is invalid; use magma_v2.h"
#endif

#include "magma_types.h"
#include "magma_copy_v1.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
magmablas_ctranspose_inplace_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda );

void
magmablas_ctranspose_conj_inplace_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda );

void
magmablas_ctranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, magma_int_t lddat );

void
magmablas_ctranspose_conj_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, magma_int_t lddat );

void
magmablas_cgetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dAT,   magma_int_t ldda,
    magmaFloatComplex          *hA,    magma_int_t lda,
    magmaFloatComplex_ptr       dwork, magma_int_t lddwork, magma_int_t nb );

void
magmablas_csetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA,    magma_int_t lda,
    magmaFloatComplex_ptr    dAT,   magma_int_t ldda,
    magmaFloatComplex_ptr    dwork, magma_int_t lddwork, magma_int_t nb );

  /*
   * RBT-related functions
   */
void
magmablas_cprbt_v1(
    magma_int_t n, 
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    magmaFloatComplex_ptr du,
    magmaFloatComplex_ptr dv );

void
magmablas_cprbt_mv_v1(
    magma_int_t n, 
    magmaFloatComplex_ptr dv,
    magmaFloatComplex_ptr db );

void
magmablas_cprbt_mtv_v1(
    magma_int_t n, 
    magmaFloatComplex_ptr du,
    magmaFloatComplex_ptr db );

  /*
   * Multi-GPU copy functions
   */
void
magma_cgetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const dA[], magma_int_t ldda,
    magmaFloatComplex                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_csetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA,   magma_int_t lda,
    magmaFloatComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_cgetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const dA[], magma_int_t ldda,
    magmaFloatComplex                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb );

void
magma_csetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA,   magma_int_t lda,
    magmaFloatComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
magmablas_cgeadd_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magmablas_cgeadd2_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magmablas_clacpy_v1(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magmablas_clacpy_conj_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dA1, magma_int_t lda1,
    magmaFloatComplex_ptr dA2, magma_int_t lda2 );

void
magmablas_clacpy_sym_in_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magmablas_clacpy_sym_out_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

float
magmablas_clange_v1(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork );

float
magmablas_clanhe_v1(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork );

float
magmablas_clansy_v1(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork );

void
magmablas_clarfg_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dalpha,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dtau );

void
magmablas_clascl_v1(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_clascl_2x2_v1(
    magma_type_t type, magma_int_t m,
    magmaFloatComplex_const_ptr dW, magma_int_t lddw,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_clascl2_v1(
    magma_type_t type,
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_clascl_diag_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dD, magma_int_t lddd,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info );

void
magmablas_claset_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda );

void
magmablas_claset_band_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda );

void
magmablas_claswp_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_claswp2_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci );

void
magmablas_claswp_sym_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_claswpx_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci );

void
magmablas_csymmetrize_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda );

void
magmablas_csymmetrize_tiles_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride );

void
magmablas_ctrtri_diag_v1(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr d_dinvA );

  /*
   * to cleanup (alphabetical order)
   */
void
magmablas_scnrm2_adjust_v1(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloatComplex_ptr dc );

void
magmablas_scnrm2_check_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc );

void
magmablas_scnrm2_cols_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dxnorm );

void
magmablas_scnrm2_row_check_adjust_v1(
    magma_int_t k, float tol,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dxnorm2,
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dlsticc );

magma_int_t
magma_clarfb_gpu_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV, magma_int_t lddv,
    magmaFloatComplex_const_ptr dT, magma_int_t lddt,
    magmaFloatComplex_ptr dC,       magma_int_t lddc,
    magmaFloatComplex_ptr dwork,    magma_int_t ldwork );

magma_int_t
magma_clarfb_gpu_gemm_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV, magma_int_t lddv,
    magmaFloatComplex_const_ptr dT, magma_int_t lddt,
    magmaFloatComplex_ptr dC,       magma_int_t lddc,
    magmaFloatComplex_ptr dwork,    magma_int_t ldwork,
    magmaFloatComplex_ptr dworkvt,  magma_int_t ldworkvt );

void
magma_clarfbx_gpu_v1(
    magma_int_t m, magma_int_t k,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr dT, magma_int_t ldt,
    magmaFloatComplex_ptr c,
    magmaFloatComplex_ptr dwork );

void
magma_clarfg_gpu_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dAkk );

void
magma_clarfgtx_gpu_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr T,  magma_int_t ldt,
    magmaFloatComplex_ptr dwork );

void
magma_clarfgx_gpu_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter );

void
magma_clarfx_gpu_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr tau,
    magmaFloatComplex_ptr C,  magma_int_t ldc,
    magmaFloat_ptr        xnorm,
    magmaFloatComplex_ptr dT, magma_int_t iter,
    magmaFloatComplex_ptr work );


  /*
   * Level 1 BLAS (alphabetical order)
   */
void
magmablas_caxpycp_v1(
    magma_int_t m,
    magmaFloatComplex_ptr dr,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_const_ptr db );

void
magmablas_cswap_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy );

void
magmablas_cswapblk_v1(
    magma_order_t order,
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset );

void
magmablas_cswapdblk_v1(
    magma_int_t n, magma_int_t nb,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloatComplex_ptr dB, magma_int_t lddb, magma_int_t incb );

  /*
   * Level 2 BLAS (alphabetical order)
   */
void
magmablas_cgemv_v1(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

void
magmablas_cgemv_conj_v1(
    magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy, magma_int_t incy );

magma_int_t
magmablas_chemv_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

magma_int_t
magmablas_csymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
magmablas_cgemm_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_cgemm_reduce_v1(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_chemm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_csymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_csyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_cher2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    float  beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_csyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_cherk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float  alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    float  beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magmablas_ctrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magmablas_ctrsm_outofplace_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magmaFloatComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloatComplex_ptr d_dinvA, magma_int_t dinvA_length );

void
magmablas_ctrsm_work_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magmaFloatComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloatComplex_ptr d_dinvA, magma_int_t dinvA_length );


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

#define magma_csetvector_v1(           n, hx_src, incx, dy_dst, incy ) \
        magma_csetvector_v1_internal(  n, hx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_cgetvector_v1(           n, dx_src, incx, hy_dst, incy ) \
        magma_cgetvector_v1_internal(  n, dx_src, incx, hy_dst, incy, __func__, __FILE__, __LINE__ )

#define magma_ccopyvector_v1(          n, dx_src, incx, dy_dst, incy ) \
        magma_ccopyvector_v1_internal( n, dx_src, incx, dy_dst, incy, __func__, __FILE__, __LINE__ )

static inline void
magma_csetvector_v1_internal(
    magma_int_t n,
    magmaFloatComplex const    *hx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_setvector_v1_internal( n, sizeof(magmaFloatComplex),
                                 hx_src, incx,
                                 dy_dst, incy,
                                 func, file, line );
}

static inline void
magma_cgetvector_v1_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex          *hy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_getvector_v1_internal( n, sizeof(magmaFloatComplex),
                                 dx_src, incx,
                                 hy_dst, incy,
                                 func, file, line );
}

static inline void
magma_ccopyvector_v1_internal(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx_src, magma_int_t incx,
    magmaFloatComplex_ptr       dy_dst, magma_int_t incy,
    const char* func, const char* file, int line )
{
    magma_copyvector_v1_internal( n, sizeof(magmaFloatComplex),
                                  dx_src, incx,
                                  dy_dst, incy,
                                  func, file, line );
}


// =============================================================================
// copying sub-matrices (contiguous columns)

#define magma_csetmatrix_v1(           m, n, hA_src, lda,  dB_dst, lddb ) \
        magma_csetmatrix_v1_internal(  m, n, hA_src, lda,  dB_dst, lddb, __func__, __FILE__, __LINE__ )

#define magma_cgetmatrix_v1(           m, n, dA_src, ldda, hB_dst, ldb  ) \
        magma_cgetmatrix_v1_internal(  m, n, dA_src, ldda, hB_dst, ldb,  __func__, __FILE__, __LINE__ )

#define magma_ccopymatrix_v1(          m, n, dA_src, ldda, dB_dst, lddb ) \
        magma_ccopymatrix_v1_internal( m, n, dA_src, ldda, dB_dst, lddb, __func__, __FILE__, __LINE__ )

static inline void
magma_csetmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex const    *hA_src, magma_int_t lda,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_setmatrix_v1_internal( m, n, sizeof(magmaFloatComplex),
                                 hA_src, lda,
                                 dB_dst, lddb,
                                 func, file, line );
}

static inline void
magma_cgetmatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex          *hB_dst, magma_int_t ldb,
    const char* func, const char* file, int line )
{
    magma_getmatrix_v1_internal( m, n, sizeof(magmaFloatComplex),
                                 dA_src, ldda,
                                 hB_dst, ldb,
                                 func, file, line );
}

static inline void
magma_ccopymatrix_v1_internal(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA_src, magma_int_t ldda,
    magmaFloatComplex_ptr       dB_dst, magma_int_t lddb,
    const char* func, const char* file, int line )
{
    magma_copymatrix_v1_internal( m, n, sizeof(magmaFloatComplex),
                                  dA_src, ldda,
                                  dB_dst, lddb,
                                  func, file, line );
}


// =============================================================================
// Level 1 BLAS (alphabetical order)

// in cublas_v2, result returned through output argument
magma_int_t
magma_icamax_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
magma_int_t
magma_icamin_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

// in cublas_v2, result returned through output argument
float
magma_scasum_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

void
magma_caxpy_v1(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

void
magma_ccopy_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaFloatComplex
magma_cdotc_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
magmaFloatComplex
magma_cdotu_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy );

// in cublas_v2, result returned through output argument
float
magma_scnrm2_v1(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx );

void
magma_crot_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float dc, magmaFloatComplex ds );

void
magma_csrot_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float dc, float ds );

#ifdef REAL
void
magma_crotm_v1(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy,
    magmaFloat_const_ptr param );

void
magma_crotmg_v1(
    magmaFloat_ptr d1, magmaFloat_ptr       d2,
    magmaFloat_ptr x1, magmaFloat_const_ptr y1,
    magmaFloat_ptr param );
#endif

void
magma_cscal_v1(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx );

void
magma_csscal_v1(
    magma_int_t n,
    float alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx );

void
magma_cswap_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy );

// =============================================================================
// Level 2 BLAS (alphabetical order)

void
magma_cgemv_v1(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

void
magma_cgerc_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda );

void
magma_cgeru_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda );

void
magma_chemv_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy );

void
magma_cher_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dA, magma_int_t ldda );

void
magma_cher2_v1(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda );

void
magma_ctrmv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx );

void
magma_ctrsv_v1(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx );

// =============================================================================
// Level 3 BLAS (alphabetical order)

void
magma_cgemm_v1(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_csymm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_chemm_v1(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_csyr2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_cher2k_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_csyrk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_cherk_v1(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc );

void
magma_ctrmm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );

void
magma_ctrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb );


#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif // MAGMABLAS_C_H
