/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from magmablas/magmablas_z_v1.cpp, normal z -> s, Sun Nov 20 20:20:31 2016

       @author Mark Gates

       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "magma_internal.h"
#include "magmablas_v1.h"  // includes v1 prototypes; does NOT map routine names

#define REAL

// These MAGMA v1 routines are all deprecated.
// See corresponding v2 functions for documentation.

/******************************************************************************/
extern "C" void
magmablas_saxpycp_v1(
    magma_int_t m,
    magmaFloat_ptr r,
    magmaFloat_ptr x,
    magmaFloat_const_ptr b)
{
    magmablas_saxpycp( m, r, x, b, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sgeadd_v1(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_sgeadd( m, n, alpha, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sgeadd2_v1(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_sgeadd2( m, n, alpha, dA, ldda, beta, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sgemm_v1(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc )
{
    magmablas_sgemm( transA, transB, m, n, k,
                       alpha, dA, ldda,
                              dB, lddb,
                       beta,  dC, lddc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sgemv_v1(
    magma_trans_t trans, magma_int_t m, magma_int_t n, float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy)
{
    magmablas_sgemv( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sgemv_conj_v1(
    magma_int_t m, magma_int_t n, float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy)
{
    magmablas_sgemv_conj(
        m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sgemm_reduce_v1(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc )
{
    magmablas_sgemm_reduce(
        m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sgetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dAT, magma_int_t ldda,
    float          *hA,  magma_int_t lda,
    magmaFloat_ptr       dwork,  magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create_v1( &queues[0] );
    magma_queue_create_v1( &queues[1] );

    magmablas_sgetmatrix_transpose( m, n, nb, dAT, ldda, hA, lda, dwork, lddwork, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/******************************************************************************/
extern "C" magma_int_t
magmablas_ssymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy )
{
    return magmablas_ssymv( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


#ifdef COMPLEX
/******************************************************************************/
extern "C" magma_int_t
magmablas_ssymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy )
{
    return magmablas_ssymv( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}
#endif


/******************************************************************************/
extern "C" void
magmablas_sprbt_v1(
    magma_int_t n,
    float *dA, magma_int_t ldda,
    float *du, float *dv)
{
    magmablas_sprbt(n, dA, ldda, du, dv, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sprbt_mv_v1(
    magma_int_t n,
    float *dv, float *db)
{
    magmablas_sprbt_mv(n, dv, db, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sprbt_mtv_v1(
    magma_int_t n,
    float *du, float *db)
{
    magmablas_sprbt_mtv(n, du, db, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slacpy_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_slacpy( uplo, m, n, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slacpy_conj_v1(
    magma_int_t n,
    magmaFloat_ptr dA1, magma_int_t lda1,
    magmaFloat_ptr dA2, magma_int_t lda2)
{
    magmablas_slacpy_conj( n, dA1, lda1, dA2, lda2, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slacpy_sym_in_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_slacpy_sym_in( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slacpy_sym_out_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_slacpy_sym_out( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" float
magmablas_slange_v1(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork )
{
    return magmablas_slange( norm, m, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" float
magmablas_slansy_v1(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork )
{
    return magmablas_slansy( norm, uplo, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_slarfx_gpu_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr v,
    magmaFloat_ptr tau,
    magmaFloat_ptr C, magma_int_t ldc,
    magmaFloat_ptr        xnorm,
    magmaFloat_ptr dT, magma_int_t iter,
    magmaFloat_ptr work )
{
    magma_slarfx_gpu(m, n, v, tau, C, ldc, xnorm, dT, iter, work,
                       magmablasGetQueue());
}


/******************************************************************************/
extern "C" void
magma_slarfbx_gpu_v1(
    magma_int_t m, magma_int_t k,
    magmaFloat_ptr V,  magma_int_t ldv,
    magmaFloat_ptr dT, magma_int_t ldt,
    magmaFloat_ptr c,
    magmaFloat_ptr dwork)
{
    magma_slarfbx_gpu( m, k, V, ldv, dT, ldt, c, dwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slarfg_v1(
    magma_int_t n,
    magmaFloat_ptr dalpha,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dtau )
{
    magmablas_slarfg( n, dalpha, dx, incx, dtau, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_slarfg_gpu_v1(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dAkk )
{
    magma_slarfg_gpu( n, dx0, dx, dtau, dxnorm, dAkk, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_slarfgx_gpu_v1(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dA, magma_int_t iter)
{
    magma_slarfgx_gpu( n, dx0, dx, dtau, dxnorm, dA, iter, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_slarfgtx_gpu_v1(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dA, magma_int_t iter,
    magmaFloat_ptr V,  magma_int_t ldv,
    magmaFloat_ptr T,  magma_int_t ldt,
    magmaFloat_ptr dwork )
{
    magma_slarfgtx_gpu(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv,
                         T, ldt, dwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slascl_v1(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_slascl( type, kl, ku, cfrom, cto, m, n, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_slascl2_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *info )
{
    magmablas_slascl2( type, m, n, dD, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_slascl_2x2_v1(
    magma_type_t type, magma_int_t m,
    magmaFloat_const_ptr dW, magma_int_t lddw,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_slascl_2x2( type, m, dW, lddw, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_slascl_diag_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD, magma_int_t lddd,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_slascl_diag( type, m, n, dD, lddd, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_slaset_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda )
{
    magmablas_slaset( uplo, m, n, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slaset_band_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda)
{
    magmablas_slaset_band(uplo, m, n, k, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slaswp_v1(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_slaswp( n, dAT, ldda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slaswpx_v1(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_slaswpx( n, dA, ldx, ldy, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slaswp2_v1(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci )
{
    magmablas_slaswp2( n, dAT, ldda, k1, k2, d_ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_slaswp_sym_v1( magma_int_t n, float *dA, magma_int_t lda,
                      magma_int_t k1, magma_int_t k2,
                      const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_slaswp_sym( n, dA, lda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_snrm2_check_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc ) 
{
    magmablas_snrm2_check( m, n, dA, ldda, dxnorm, dlsticc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_snrm2_adjust_v1(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dc )
{
    magmablas_snrm2_adjust( k, dxnorm, dc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_snrm2_row_check_adjust_v1(
    magma_int_t k, float tol,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dxnorm2, 
    magmaFloat_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dlsticc )
{
    magmablas_snrm2_row_check_adjust( k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_snrm2_cols_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr dxnorm ) 
{
    magmablas_snrm2_cols( m, n, dA, ldda, dxnorm, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ssetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    const float     *hA, magma_int_t lda,
    magmaFloat_ptr       dAT, magma_int_t ldda,
    magmaFloat_ptr     dwork, magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create_v1( &queues[0] );
    magma_queue_create_v1( &queues[1] );

    magmablas_ssetmatrix_transpose( m, n, nb, hA, lda, dAT, ldda, dwork, lddwork, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/******************************************************************************/
extern "C" void
magmablas_sswap_v1(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy)
{
    magmablas_sswap( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sswapblk_v1(
    magma_order_t order, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    magmablas_sswapblk(
        order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_sswapdblk_v1(
    magma_int_t n, magma_int_t nb,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloat_ptr dB, magma_int_t lddb, magma_int_t incb )
{
    magmablas_sswapdblk( n, nb, dA, ldda, inca, dB, lddb, incb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ssymmetrize_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda )
{
    magmablas_ssymmetrize( uplo, m, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ssymmetrize_tiles_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride )
{
    magmablas_ssymmetrize_tiles( uplo, m, dA, ldda, ntile, mstride, nstride, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_stranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat )
{
    magmablas_stranspose( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


#ifdef COMPLEX
/******************************************************************************/
extern "C" void
magmablas_stranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat )
{
    magmablas_stranspose( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_stranspose_inplace_v1(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda )
{
    magmablas_stranspose_inplace( n, dA, ldda, magmablasGetQueue() );
}
#endif


/******************************************************************************/
extern "C" void
magmablas_stranspose_inplace_v1(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda )
{
    magmablas_stranspose_inplace( n, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_strsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_strsm( side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb,
                       magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_strsm_outofplace_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magmaFloat_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloat_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_strsm_outofplace( side, uplo, transA, diag, m, n, alpha,
                                  dA, ldda, dB, lddb, dX, lddx, flag,
                                  d_dinvA, dinvA_length, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_strsm_work_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magmaFloat_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloat_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_strsm_work( side, uplo, transA, diag, m, n, alpha,
                            dA, ldda, dB, lddb, dX, lddx, flag,
                            d_dinvA, dinvA_length, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_strtri_diag_v1(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr d_dinvA)
{
    magmablas_strtri_diag( uplo, diag, n, dA, ldda, d_dinvA, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_sgetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr const *dA, magma_int_t ldda,
    float                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_sgetmatrix_1D_row_bcyclic( ngpu, m, n, nb, dA, ldda, hA, lda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_sgetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr const *dA, magma_int_t ldda,
    float                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_sgetmatrix_1D_col_bcyclic( ngpu, m, n, nb, dA, ldda, hA, lda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_ssetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const float    *hA, magma_int_t lda,
    magmaFloat_ptr      *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_ssetmatrix_1D_row_bcyclic( ngpu, m, n, nb, hA, lda, dA, ldda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_ssetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const float *hA, magma_int_t lda,
    magmaFloat_ptr   *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_ssetmatrix_1D_col_bcyclic( ngpu, m, n, nb, hA, lda, dA, ldda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


// in src/slarfb_gpu.cpp
/******************************************************************************/
extern "C" magma_int_t
magma_slarfb_gpu_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV,    magma_int_t lddv,
    magmaFloat_const_ptr dT,    magma_int_t lddt,
    magmaFloat_ptr dC,          magma_int_t lddc,
    magmaFloat_ptr dwork,       magma_int_t ldwork )
{
    return magma_slarfb_gpu( side, trans, direct, storev,
                               m, n, k,
                               dV, lddv, dT, lddt, dC, lddc, dwork, ldwork,
                               magmablasGetQueue() );
}


// in src/slarfb_gpu_gemm.cpp
/******************************************************************************/
extern "C" magma_int_t
magma_slarfb_gpu_gemm_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV,    magma_int_t lddv,
    magmaFloat_const_ptr dT,    magma_int_t lddt,
    magmaFloat_ptr dC,          magma_int_t lddc,
    magmaFloat_ptr dwork,       magma_int_t ldwork,
    magmaFloat_ptr dworkvt,     magma_int_t ldworkvt )
{
    return magma_slarfb_gpu_gemm( side, trans, direct, storev,
                                    m, n, k,
                                    dV, lddv, dT, lddt, dC, lddc,
                                    dwork, ldwork, dworkvt, ldworkvt,
                                    magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
