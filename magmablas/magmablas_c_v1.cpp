/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from magmablas/magmablas_z_v1.cpp, normal z -> c, Sun Nov 20 20:20:31 2016

       @author Mark Gates

       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "magma_internal.h"
#include "magmablas_v1.h"  // includes v1 prototypes; does NOT map routine names

#define COMPLEX

// These MAGMA v1 routines are all deprecated.
// See corresponding v2 functions for documentation.

/******************************************************************************/
extern "C" void
magmablas_caxpycp_v1(
    magma_int_t m,
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr x,
    magmaFloatComplex_const_ptr b)
{
    magmablas_caxpycp( m, r, x, b, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cgeadd_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_cgeadd( m, n, alpha, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cgeadd2_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_cgeadd2( m, n, alpha, dA, ldda, beta, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cgemm_v1(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magmablas_cgemm( transA, transB, m, n, k,
                       alpha, dA, ldda,
                              dB, lddb,
                       beta,  dC, lddc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cgemv_v1(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy, magma_int_t incy)
{
    magmablas_cgemv( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cgemv_conj_v1(
    magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy, magma_int_t incy)
{
    magmablas_cgemv_conj(
        m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cgemm_reduce_v1(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magmablas_cgemm_reduce(
        m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cgetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dAT, magma_int_t ldda,
    magmaFloatComplex          *hA,  magma_int_t lda,
    magmaFloatComplex_ptr       dwork,  magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create_v1( &queues[0] );
    magma_queue_create_v1( &queues[1] );

    magmablas_cgetmatrix_transpose( m, n, nb, dAT, ldda, hA, lda, dwork, lddwork, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/******************************************************************************/
extern "C" magma_int_t
magmablas_chemv_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    return magmablas_chemv( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


#ifdef COMPLEX
/******************************************************************************/
extern "C" magma_int_t
magmablas_csymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    return magmablas_csymv( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}
#endif


/******************************************************************************/
extern "C" void
magmablas_cprbt_v1(
    magma_int_t n,
    magmaFloatComplex *dA, magma_int_t ldda,
    magmaFloatComplex *du, magmaFloatComplex *dv)
{
    magmablas_cprbt(n, dA, ldda, du, dv, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cprbt_mv_v1(
    magma_int_t n,
    magmaFloatComplex *dv, magmaFloatComplex *db)
{
    magmablas_cprbt_mv(n, dv, db, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cprbt_mtv_v1(
    magma_int_t n,
    magmaFloatComplex *du, magmaFloatComplex *db)
{
    magmablas_cprbt_mtv(n, du, db, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_clacpy_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_clacpy( uplo, m, n, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_clacpy_conj_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dA1, magma_int_t lda1,
    magmaFloatComplex_ptr dA2, magma_int_t lda2)
{
    magmablas_clacpy_conj( n, dA1, lda1, dA2, lda2, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_clacpy_sym_in_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_clacpy_sym_in( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_clacpy_sym_out_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_clacpy_sym_out( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" float
magmablas_clange_v1(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork )
{
    return magmablas_clange( norm, m, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" float
magmablas_clanhe_v1(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork )
{
    return magmablas_clanhe( norm, uplo, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_clarfx_gpu_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr tau,
    magmaFloatComplex_ptr C, magma_int_t ldc,
    magmaFloat_ptr        xnorm,
    magmaFloatComplex_ptr dT, magma_int_t iter,
    magmaFloatComplex_ptr work )
{
    magma_clarfx_gpu(m, n, v, tau, C, ldc, xnorm, dT, iter, work,
                       magmablasGetQueue());
}


/******************************************************************************/
extern "C" void
magma_clarfbx_gpu_v1(
    magma_int_t m, magma_int_t k,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr dT, magma_int_t ldt,
    magmaFloatComplex_ptr c,
    magmaFloatComplex_ptr dwork)
{
    magma_clarfbx_gpu( m, k, V, ldv, dT, ldt, c, dwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_clarfg_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dalpha,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dtau )
{
    magmablas_clarfg( n, dalpha, dx, incx, dtau, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_clarfg_gpu_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dAkk )
{
    magma_clarfg_gpu( n, dx0, dx, dtau, dxnorm, dAkk, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_clarfgx_gpu_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter)
{
    magma_clarfgx_gpu( n, dx0, dx, dtau, dxnorm, dA, iter, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_clarfgtx_gpu_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr T,  magma_int_t ldt,
    magmaFloatComplex_ptr dwork )
{
    magma_clarfgtx_gpu(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv,
                         T, ldt, dwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_clascl_v1(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_clascl( type, kl, ku, cfrom, cto, m, n, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_clascl2_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t *info )
{
    magmablas_clascl2( type, m, n, dD, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_clascl_2x2_v1(
    magma_type_t type, magma_int_t m,
    magmaFloatComplex_const_ptr dW, magma_int_t lddw,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_clascl_2x2( type, m, dW, lddw, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_clascl_diag_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dD, magma_int_t lddd,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_clascl_diag( type, m, n, dD, lddd, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_claset_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda )
{
    magmablas_claset( uplo, m, n, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_claset_band_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda)
{
    magmablas_claset_band(uplo, m, n, k, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_claswp_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_claswp( n, dAT, ldda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_claswpx_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_claswpx( n, dA, ldx, ldy, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_claswp2_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci )
{
    magmablas_claswp2( n, dAT, ldda, k1, k2, d_ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_claswp_sym_v1( magma_int_t n, magmaFloatComplex *dA, magma_int_t lda,
                      magma_int_t k1, magma_int_t k2,
                      const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_claswp_sym( n, dA, lda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_scnrm2_check_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc ) 
{
    magmablas_scnrm2_check( m, n, dA, ldda, dxnorm, dlsticc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_scnrm2_adjust_v1(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloatComplex_ptr dc )
{
    magmablas_scnrm2_adjust( k, dxnorm, dc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_scnrm2_row_check_adjust_v1(
    magma_int_t k, float tol,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dxnorm2, 
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dlsticc )
{
    magmablas_scnrm2_row_check_adjust( k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_scnrm2_cols_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr dxnorm ) 
{
    magmablas_scnrm2_cols( m, n, dA, ldda, dxnorm, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_csetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex     *hA, magma_int_t lda,
    magmaFloatComplex_ptr       dAT, magma_int_t ldda,
    magmaFloatComplex_ptr     dwork, magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create_v1( &queues[0] );
    magma_queue_create_v1( &queues[1] );

    magmablas_csetmatrix_transpose( m, n, nb, hA, lda, dAT, ldda, dwork, lddwork, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/******************************************************************************/
extern "C" void
magmablas_cswap_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy)
{
    magmablas_cswap( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cswapblk_v1(
    magma_order_t order, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    magmablas_cswapblk(
        order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_cswapdblk_v1(
    magma_int_t n, magma_int_t nb,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloatComplex_ptr dB, magma_int_t lddb, magma_int_t incb )
{
    magmablas_cswapdblk( n, nb, dA, ldda, inca, dB, lddb, incb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_csymmetrize_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda )
{
    magmablas_csymmetrize( uplo, m, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_csymmetrize_tiles_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride )
{
    magmablas_csymmetrize_tiles( uplo, m, dA, ldda, ntile, mstride, nstride, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ctranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, magma_int_t lddat )
{
    magmablas_ctranspose( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


#ifdef COMPLEX
/******************************************************************************/
extern "C" void
magmablas_ctranspose_conj_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, magma_int_t lddat )
{
    magmablas_ctranspose_conj( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ctranspose_conj_inplace_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda )
{
    magmablas_ctranspose_conj_inplace( n, dA, ldda, magmablasGetQueue() );
}
#endif


/******************************************************************************/
extern "C" void
magmablas_ctranspose_inplace_v1(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda )
{
    magmablas_ctranspose_inplace( n, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ctrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_ctrsm( side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb,
                       magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ctrsm_outofplace_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magmaFloatComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloatComplex_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_ctrsm_outofplace( side, uplo, transA, diag, m, n, alpha,
                                  dA, ldda, dB, lddb, dX, lddx, flag,
                                  d_dinvA, dinvA_length, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ctrsm_work_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magmaFloatComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloatComplex_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_ctrsm_work( side, uplo, transA, diag, m, n, alpha,
                            dA, ldda, dB, lddb, dX, lddx, flag,
                            d_dinvA, dinvA_length, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_ctrtri_diag_v1(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr d_dinvA)
{
    magmablas_ctrtri_diag( uplo, diag, n, dA, ldda, d_dinvA, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_cgetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const *dA, magma_int_t ldda,
    magmaFloatComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_cgetmatrix_1D_row_bcyclic( ngpu, m, n, nb, dA, ldda, hA, lda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_cgetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const *dA, magma_int_t ldda,
    magmaFloatComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_cgetmatrix_1D_col_bcyclic( ngpu, m, n, nb, dA, ldda, hA, lda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_csetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex    *hA, magma_int_t lda,
    magmaFloatComplex_ptr      *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_csetmatrix_1D_row_bcyclic( ngpu, m, n, nb, hA, lda, dA, ldda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_csetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA, magma_int_t lda,
    magmaFloatComplex_ptr   *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_csetmatrix_1D_col_bcyclic( ngpu, m, n, nb, hA, lda, dA, ldda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


// in src/clarfb_gpu.cpp
/******************************************************************************/
extern "C" magma_int_t
magma_clarfb_gpu_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV,    magma_int_t lddv,
    magmaFloatComplex_const_ptr dT,    magma_int_t lddt,
    magmaFloatComplex_ptr dC,          magma_int_t lddc,
    magmaFloatComplex_ptr dwork,       magma_int_t ldwork )
{
    return magma_clarfb_gpu( side, trans, direct, storev,
                               m, n, k,
                               dV, lddv, dT, lddt, dC, lddc, dwork, ldwork,
                               magmablasGetQueue() );
}


// in src/clarfb_gpu_gemm.cpp
/******************************************************************************/
extern "C" magma_int_t
magma_clarfb_gpu_gemm_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV,    magma_int_t lddv,
    magmaFloatComplex_const_ptr dT,    magma_int_t lddt,
    magmaFloatComplex_ptr dC,          magma_int_t lddc,
    magmaFloatComplex_ptr dwork,       magma_int_t ldwork,
    magmaFloatComplex_ptr dworkvt,     magma_int_t ldworkvt )
{
    return magma_clarfb_gpu_gemm( side, trans, direct, storev,
                                    m, n, k,
                                    dV, lddv, dT, lddt, dC, lddc,
                                    dwork, ldwork, dworkvt, ldworkvt,
                                    magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
