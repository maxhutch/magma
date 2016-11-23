/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from magmablas/magmablas_z_v1.cpp, normal z -> d, Sun Nov 20 20:20:31 2016

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
magmablas_daxpycp_v1(
    magma_int_t m,
    magmaDouble_ptr r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b)
{
    magmablas_daxpycp( m, r, x, b, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dgeadd_v1(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dgeadd( m, n, alpha, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dgeadd2_v1(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dgeadd2( m, n, alpha, dA, ldda, beta, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dgemm_v1(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magmablas_dgemm( transA, transB, m, n, k,
                       alpha, dA, ldda,
                              dB, lddb,
                       beta,  dC, lddc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dgemv_v1(
    magma_trans_t trans, magma_int_t m, magma_int_t n, double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr dy, magma_int_t incy)
{
    magmablas_dgemv( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dgemv_conj_v1(
    magma_int_t m, magma_int_t n, double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr dy, magma_int_t incy)
{
    magmablas_dgemv_conj(
        m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dgemm_reduce_v1(
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magmablas_dgemm_reduce(
        m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dgetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dAT, magma_int_t ldda,
    double          *hA,  magma_int_t lda,
    magmaDouble_ptr       dwork,  magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create_v1( &queues[0] );
    magma_queue_create_v1( &queues[1] );

    magmablas_dgetmatrix_transpose( m, n, nb, dAT, ldda, hA, lda, dwork, lddwork, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/******************************************************************************/
extern "C" magma_int_t
magmablas_dsymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    return magmablas_dsymv( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


#ifdef COMPLEX
/******************************************************************************/
extern "C" magma_int_t
magmablas_dsymv_v1(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    return magmablas_dsymv( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}
#endif


/******************************************************************************/
extern "C" void
magmablas_dprbt_v1(
    magma_int_t n,
    double *dA, magma_int_t ldda,
    double *du, double *dv)
{
    magmablas_dprbt(n, dA, ldda, du, dv, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dprbt_mv_v1(
    magma_int_t n,
    double *dv, double *db)
{
    magmablas_dprbt_mv(n, dv, db, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dprbt_mtv_v1(
    magma_int_t n,
    double *du, double *db)
{
    magmablas_dprbt_mtv(n, du, db, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlacpy_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dlacpy( uplo, m, n, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlacpy_conj_v1(
    magma_int_t n,
    magmaDouble_ptr dA1, magma_int_t lda1,
    magmaDouble_ptr dA2, magma_int_t lda2)
{
    magmablas_dlacpy_conj( n, dA1, lda1, dA2, lda2, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlacpy_sym_in_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dlacpy_sym_in( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlacpy_sym_out_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dlacpy_sym_out( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" double
magmablas_dlange_v1(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork )
{
    return magmablas_dlange( norm, m, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" double
magmablas_dlansy_v1(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork )
{
    return magmablas_dlansy( norm, uplo, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dlarfx_gpu_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr v,
    magmaDouble_ptr tau,
    magmaDouble_ptr C, magma_int_t ldc,
    magmaDouble_ptr        xnorm,
    magmaDouble_ptr dT, magma_int_t iter,
    magmaDouble_ptr work )
{
    magma_dlarfx_gpu(m, n, v, tau, C, ldc, xnorm, dT, iter, work,
                       magmablasGetQueue());
}


/******************************************************************************/
extern "C" void
magma_dlarfbx_gpu_v1(
    magma_int_t m, magma_int_t k,
    magmaDouble_ptr V,  magma_int_t ldv,
    magmaDouble_ptr dT, magma_int_t ldt,
    magmaDouble_ptr c,
    magmaDouble_ptr dwork)
{
    magma_dlarfbx_gpu( m, k, V, ldv, dT, ldt, c, dwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlarfg_v1(
    magma_int_t n,
    magmaDouble_ptr dalpha,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dtau )
{
    magmablas_dlarfg( n, dalpha, dx, incx, dtau, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dlarfg_gpu_v1(
    magma_int_t n,
    magmaDouble_ptr dx0,
    magmaDouble_ptr dx,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDouble_ptr dAkk )
{
    magma_dlarfg_gpu( n, dx0, dx, dtau, dxnorm, dAkk, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dlarfgx_gpu_v1(
    magma_int_t n,
    magmaDouble_ptr dx0,
    magmaDouble_ptr dx,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDouble_ptr dA, magma_int_t iter)
{
    magma_dlarfgx_gpu( n, dx0, dx, dtau, dxnorm, dA, iter, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dlarfgtx_gpu_v1(
    magma_int_t n,
    magmaDouble_ptr dx0,
    magmaDouble_ptr dx,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDouble_ptr dA, magma_int_t iter,
    magmaDouble_ptr V,  magma_int_t ldv,
    magmaDouble_ptr T,  magma_int_t ldt,
    magmaDouble_ptr dwork )
{
    magma_dlarfgtx_gpu(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv,
                         T, ldt, dwork, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlascl_v1(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_dlascl( type, kl, ku, cfrom, cto, m, n, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dlascl2_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *info )
{
    magmablas_dlascl2( type, m, n, dD, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dlascl_2x2_v1(
    magma_type_t type, magma_int_t m,
    magmaDouble_const_ptr dW, magma_int_t lddw,
    magmaDouble_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_dlascl_2x2( type, m, dW, lddw, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dlascl_diag_v1(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD, magma_int_t lddd,
    magmaDouble_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_dlascl_diag( type, m, n, dD, lddd, dA, ldda, magmablasGetQueue(), info );
}


/******************************************************************************/
extern "C" void
magmablas_dlaset_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    magmaDouble_ptr dA, magma_int_t ldda )
{
    magmablas_dlaset( uplo, m, n, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlaset_band_v1(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    double offdiag, double diag,
    magmaDouble_ptr dA, magma_int_t ldda)
{
    magmablas_dlaset_band(uplo, m, n, k, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlaswp_v1(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_dlaswp( n, dAT, ldda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlaswpx_v1(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_dlaswpx( n, dA, ldx, ldy, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlaswp2_v1(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci )
{
    magmablas_dlaswp2( n, dAT, ldda, k1, k2, d_ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dlaswp_sym_v1( magma_int_t n, double *dA, magma_int_t lda,
                      magma_int_t k1, magma_int_t k2,
                      const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_dlaswp_sym( n, dA, lda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dnrm2_check_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc ) 
{
    magmablas_dnrm2_check( m, n, dA, ldda, dxnorm, dlsticc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dnrm2_adjust_v1(
    magma_int_t k,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dc )
{
    magmablas_dnrm2_adjust( k, dxnorm, dc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dnrm2_row_check_adjust_v1(
    magma_int_t k, double tol,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dxnorm2, 
    magmaDouble_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dlsticc )
{
    magmablas_dnrm2_row_check_adjust( k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dnrm2_cols_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm ) 
{
    magmablas_dnrm2_cols( m, n, dA, ldda, dxnorm, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dsetmatrix_transpose_v1(
    magma_int_t m, magma_int_t n,
    const double     *hA, magma_int_t lda,
    magmaDouble_ptr       dAT, magma_int_t ldda,
    magmaDouble_ptr     dwork, magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create_v1( &queues[0] );
    magma_queue_create_v1( &queues[1] );

    magmablas_dsetmatrix_transpose( m, n, nb, hA, lda, dAT, ldda, dwork, lddwork, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/******************************************************************************/
extern "C" void
magmablas_dswap_v1(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy)
{
    magmablas_dswap( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dswapblk_v1(
    magma_order_t order, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    magmablas_dswapblk(
        order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dswapdblk_v1(
    magma_int_t n, magma_int_t nb,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaDouble_ptr dB, magma_int_t lddb, magma_int_t incb )
{
    magmablas_dswapdblk( n, nb, dA, ldda, inca, dB, lddb, incb, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dsymmetrize_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaDouble_ptr dA, magma_int_t ldda )
{
    magmablas_dsymmetrize( uplo, m, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dsymmetrize_tiles_v1(
    magma_uplo_t uplo, magma_int_t m,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride )
{
    magmablas_dsymmetrize_tiles( uplo, m, dA, ldda, ntile, mstride, nstride, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dtranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA,  magma_int_t ldda,
    magmaDouble_ptr       dAT, magma_int_t lddat )
{
    magmablas_dtranspose( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


#ifdef COMPLEX
/******************************************************************************/
extern "C" void
magmablas_dtranspose_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA,  magma_int_t ldda,
    magmaDouble_ptr       dAT, magma_int_t lddat )
{
    magmablas_dtranspose( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dtranspose_inplace_v1(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda )
{
    magmablas_dtranspose_inplace( n, dA, ldda, magmablasGetQueue() );
}
#endif


/******************************************************************************/
extern "C" void
magmablas_dtranspose_inplace_v1(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda )
{
    magmablas_dtranspose_inplace( n, dA, ldda, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dtrsm_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dtrsm( side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb,
                       magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dtrsm_outofplace_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb,
    magmaDouble_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDouble_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_dtrsm_outofplace( side, uplo, transA, diag, m, n, alpha,
                                  dA, ldda, dB, lddb, dX, lddx, flag,
                                  d_dinvA, dinvA_length, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dtrsm_work_v1(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb,
    magmaDouble_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDouble_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_dtrsm_work( side, uplo, transA, diag, m, n, alpha,
                            dA, ldda, dB, lddb, dX, lddx, flag,
                            d_dinvA, dinvA_length, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magmablas_dtrtri_diag_v1(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr d_dinvA)
{
    magmablas_dtrtri_diag( uplo, diag, n, dA, ldda, d_dinvA, magmablasGetQueue() );
}


/******************************************************************************/
extern "C" void
magma_dgetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const *dA, magma_int_t ldda,
    double                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_dgetmatrix_1D_row_bcyclic( ngpu, m, n, nb, dA, ldda, hA, lda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_dgetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const *dA, magma_int_t ldda,
    double                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_dgetmatrix_1D_col_bcyclic( ngpu, m, n, nb, dA, ldda, hA, lda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_dsetmatrix_1D_row_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const double    *hA, magma_int_t lda,
    magmaDouble_ptr      *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_dsetmatrix_1D_row_bcyclic( ngpu, m, n, nb, hA, lda, dA, ldda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/******************************************************************************/
extern "C" void
magma_dsetmatrix_1D_col_bcyclic_v1(
    magma_int_t m, magma_int_t n,
    const double *hA, magma_int_t lda,
    magmaDouble_ptr   *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &queues[dev] );
    }
    magma_dsetmatrix_1D_col_bcyclic( ngpu, m, n, nb, hA, lda, dA, ldda, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


// in src/dlarfb_gpu.cpp
/******************************************************************************/
extern "C" magma_int_t
magma_dlarfb_gpu_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_const_ptr dV,    magma_int_t lddv,
    magmaDouble_const_ptr dT,    magma_int_t lddt,
    magmaDouble_ptr dC,          magma_int_t lddc,
    magmaDouble_ptr dwork,       magma_int_t ldwork )
{
    return magma_dlarfb_gpu( side, trans, direct, storev,
                               m, n, k,
                               dV, lddv, dT, lddt, dC, lddc, dwork, ldwork,
                               magmablasGetQueue() );
}


// in src/dlarfb_gpu_gemm.cpp
/******************************************************************************/
extern "C" magma_int_t
magma_dlarfb_gpu_gemm_v1(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_const_ptr dV,    magma_int_t lddv,
    magmaDouble_const_ptr dT,    magma_int_t lddt,
    magmaDouble_ptr dC,          magma_int_t lddc,
    magmaDouble_ptr dwork,       magma_int_t ldwork,
    magmaDouble_ptr dworkvt,     magma_int_t ldworkvt )
{
    return magma_dlarfb_gpu_gemm( side, trans, direct, storev,
                                    m, n, k,
                                    dV, lddv, dT, lddt, dC, lddc,
                                    dwork, ldwork, dworkvt, ldworkvt,
                                    magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
