/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/magmablas_z_v1.cpp normal z -> d, Mon May  2 23:30:37 2016

       @author Mark Gates

       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "common_magma.h"

#define REAL


/**
    @see magmablas_daxpycp_q
    @ingroup magma_dblas1
    ********************************************************************/
extern "C" void
magmablas_daxpycp(
    magma_int_t m,
    magmaDouble_ptr r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b)
{
    magmablas_daxpycp_q( m, r, x, b, magmablasGetQueue() );
}


/**
    @see magmablas_dgeadd_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dgeadd(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dgeadd_q( m, n, alpha, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_dgeadd2_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dgeadd2(
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dgeadd2_q( m, n, alpha, dA, ldda, beta, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_dgemm_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C" void
magmablas_dgemm(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magmablas_dgemm_q( transA, transB, m, n, k,
                       alpha, dA, ldda,
                              dB, lddb,
                       beta,  dC, lddc, magmablasGetQueue() );
}


/**
    @see magmablas_dgemv_q
    @ingroup magma_dblas2
    ********************************************************************/
extern "C" void
magmablas_dgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n, double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr dy, magma_int_t incy)
{
    magmablas_dgemv_q( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_dgemv_conj_q
    @ingroup magma_dblas2
    ********************************************************************/
extern "C" void
magmablas_dgemv_conj(
    magma_int_t m, magma_int_t n, double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr dy, magma_int_t incy)
{
    magmablas_dgemv_conj_q(
        m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_dgemm_reduce_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C" void
magmablas_dgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDouble_ptr       dC, magma_int_t lddc )
{
    magmablas_dgemm_reduce_q(
        m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, magmablasGetQueue() );
}


// @see magmablas_dgetmatrix_transpose_q
extern "C" void
magmablas_dgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dAT, magma_int_t ldda,
    double          *hA,  magma_int_t lda,
    magmaDouble_ptr       dwork,  magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_dgetmatrix_transpose_q( m, n, dAT, ldda, hA, lda, dwork, lddwork, nb, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/**
    @see magmablas_dsymv_q
    @ingroup magma_dblas2
    ********************************************************************/
extern "C" magma_int_t
magmablas_dsymv(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    return magmablas_dsymv_q( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


#ifdef COMPLEX
/**
    @see magmablas_dsymv_q
    @ingroup magma_dblas2
    ********************************************************************/
extern "C" magma_int_t
magmablas_dsymv(
    magma_uplo_t uplo, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_const_ptr dx, magma_int_t incx,
    double beta,
    magmaDouble_ptr       dy, magma_int_t incy )
{
    return magmablas_dsymv_q( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}
#endif


/**
    @see magmablas_dprbt_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dprbt(
    magma_int_t n,
    double *dA, magma_int_t ldda,
    double *du, double *dv)
{
    magmablas_dprbt_q(n, dA, ldda, du, dv, magmablasGetQueue() );
}


/**
    @see magmablas_dprbt_mtv_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dprbt_mv(
    magma_int_t n,
    double *dv, double *db)
{
    magmablas_dprbt_mv_q(n, dv, db, magmablasGetQueue() );
}


/**
    @see magmablas_dprbt_mtv_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dprbt_mtv(
    magma_int_t n,
    double *du, double *db)
{
    magmablas_dprbt_mtv_q(n, du, db, magmablasGetQueue() );
}


/**
    @see magmablas_dlacpy_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dlacpy_q( uplo, m, n, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_dlacpy_conj_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlacpy_conj(
    magma_int_t n,
    magmaDouble_ptr dA1, magma_int_t lda1,
    magmaDouble_ptr dA2, magma_int_t lda2)
{
    magmablas_dlacpy_conj_q( n, dA1, lda1, dA2, lda2, magmablasGetQueue() );
}


/**
    @see magmablas_dlacpy_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlacpy_sym_in(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dlacpy_sym_in_q( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_dlacpy_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlacpy_sym_out(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dlacpy_sym_out_q( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_dlange_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" double
magmablas_dlange(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork )
{
    return magmablas_dlange_q( norm, m, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/**
    @see magmablas_dlansy_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" double
magmablas_dlansy(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork )
{
    return magmablas_dlansy_q( norm, uplo, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/**
    @see magma_dlarfx_gpu_q
    @ingroup magma_daux1
    ********************************************************************/
extern "C" void
magma_dlarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr v,
    magmaDouble_ptr tau,
    magmaDouble_ptr C, magma_int_t ldc,
    magmaDouble_ptr        xnorm,
    magmaDouble_ptr dT, magma_int_t iter,
    magmaDouble_ptr work )
{
    magma_dlarfx_gpu_q(m, n, v, tau, C, ldc, xnorm, dT, iter, work,
                       magmablasGetQueue());
}


/**
    @see magma_dlarfbx_gpu
    @ingroup magma_daux3
    ********************************************************************/
extern "C" void
magma_dlarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaDouble_ptr V,  magma_int_t ldv,
    magmaDouble_ptr dT, magma_int_t ldt,
    magmaDouble_ptr c,
    magmaDouble_ptr dwork)
{
    magma_dlarfbx_gpu_q( m, k, V, ldv, dT, ldt, c, dwork, magmablasGetQueue() );
}


/**
    @see magmablas_dlarfg_q
    @ingroup magma_daux1
    ********************************************************************/
extern "C"
void magmablas_dlarfg(
    magma_int_t n,
    magmaDouble_ptr dalpha,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dtau )
{
    magmablas_dlarfg_q( n, dalpha, dx, incx, dtau, magmablasGetQueue() );
}


/**
    @see magma_dlarfg_gpu_q
    @ingroup magma_daux1
    ********************************************************************/
extern "C" void
magma_dlarfg_gpu(
    magma_int_t n,
    magmaDouble_ptr dx0,
    magmaDouble_ptr dx,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDouble_ptr dAkk )
{
    magma_dlarfg_gpu_q( n, dx0, dx, dtau, dxnorm, dAkk, magmablasGetQueue() );
}


/**
    @see magma_dlarfgx_gpu_q
    @ingroup magma_daux1
    ********************************************************************/
extern "C" void
magma_dlarfgx_gpu(
    magma_int_t n,
    magmaDouble_ptr dx0,
    magmaDouble_ptr dx,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDouble_ptr dA, magma_int_t iter)
{
    magma_dlarfgx_gpu_q( n, dx0, dx, dtau, dxnorm, dA, iter, magmablasGetQueue() );
}


/**
    @see magma_dlarfgtx_gpu_q
    @ingroup magma_daux1
    ********************************************************************/
extern "C" void
magma_dlarfgtx_gpu(
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
    magma_dlarfgtx_gpu_q(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv,
                         T, ldt, dwork, magmablasGetQueue() );
}


/**
    @see magmablas_dlascl_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_dlascl_q( type, kl, ku, cfrom, cto, m, n, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_dlascl2_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlascl2(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *info )
{
    magmablas_dlascl2_q( type, m, n, dD, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_dlascl2_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlascl_2x2(
    magma_type_t type, magma_int_t m,
    magmaDouble_const_ptr dW, magma_int_t lddw,
    magmaDouble_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_dlascl_2x2_q( type, m, dW, lddw, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_dlascl_diag_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD, magma_int_t lddd,
    magmaDouble_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_dlascl_diag_q( type, m, n, dD, lddd, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_dlaset_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C"
void magmablas_dlaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    magmaDouble_ptr dA, magma_int_t ldda )
{
    magmablas_dlaset_q( uplo, m, n, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_dlaset_band_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    double offdiag, double diag,
    magmaDouble_ptr dA, magma_int_t ldda)
{
    magmablas_dlaset_band_q(uplo, m, n, k, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_dlaswp_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaswp(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_dlaswp_q( n, dAT, ldda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_dlaswpx_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaswpx(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_dlaswpx_q( n, dA, ldx, ldy, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_dlaswp2_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaswp2(
    magma_int_t n,
    magmaDouble_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci )
{
    magmablas_dlaswp2_q( n, dAT, ldda, k1, k2, d_ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_dlaswpx_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaswp_sym( magma_int_t n, double *dA, magma_int_t lda,
                      magma_int_t k1, magma_int_t k2,
                      const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_dlaswp_sym_q( n, dA, lda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_dnrm2_check_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dnrm2_check(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc ) 
{
    magmablas_dnrm2_check_q( m, n, dA, ldda, dxnorm, dlsticc, magmablasGetQueue() );
}


/**
    @see magmablas_dnrm2_adjust_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dnrm2_adjust(
    magma_int_t k,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dc )
{
    magmablas_dnrm2_adjust_q( k, dxnorm, dc, magmablasGetQueue() );
}


/**
    @see magmablas_dnrm2_row_check_adjust_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dnrm2_row_check_adjust(
    magma_int_t k, double tol,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dxnorm2, 
    magmaDouble_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dlsticc )
{
    magmablas_dnrm2_row_check_adjust_q( k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc, magmablasGetQueue() );
}


/**
    @see magmablas_dnrm2_cols_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dnrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm ) 
{
    magmablas_dnrm2_cols_q( m, n, dA, ldda, dxnorm, magmablasGetQueue() );
}


/**
    @see magmablas_dsetmatrix_transpose_q
    @ingroup magma_dblas1
    ********************************************************************/
extern "C" void
magmablas_dsetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const double     *hA, magma_int_t lda,
    magmaDouble_ptr       dAT, magma_int_t ldda,
    magmaDouble_ptr     dwork, magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_dsetmatrix_transpose_q( m, n, hA, lda, dAT, ldda, dwork, lddwork, nb, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/**
    @see magmablas_dswap_q
    @ingroup magma_dblas1
    ********************************************************************/
extern "C" void
magmablas_dswap(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy)
{
    magmablas_dswap_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_dswapblk_q
    @ingroup magma_dblas2
    ********************************************************************/
extern "C" void
magmablas_dswapblk(
    magma_order_t order, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    magmablas_dswapblk_q(
        order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset, magmablasGetQueue() );
}


/**
    @see magmablas_dswapdblk_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaDouble_ptr dB, magma_int_t lddb, magma_int_t incb )
{
    magmablas_dswapdblk_q( n, nb, dA, ldda, inca, dB, lddb, incb, magmablasGetQueue() );
}


/**
    @see magmablas_dsymmetrize_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dsymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaDouble_ptr dA, magma_int_t ldda )
{
    magmablas_dsymmetrize_q( uplo, m, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_dsymmetrize_tiles_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dsymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride )
{
    magmablas_dsymmetrize_tiles_q( uplo, m, dA, ldda, ntile, mstride, nstride, magmablasGetQueue() );
}


/**
    @see magmablas_dtranspose_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dtranspose(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA,  magma_int_t ldda,
    magmaDouble_ptr       dAT, magma_int_t lddat )
{
    magmablas_dtranspose_q( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


#ifdef COMPLEX
/**
    @see magmablas_dtranspose_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dtranspose(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA,  magma_int_t ldda,
    magmaDouble_ptr       dAT, magma_int_t lddat )
{
    magmablas_dtranspose_q( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


/**
    @see magmablas_dtranspose_inplace_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dtranspose_inplace(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda )
{
    magmablas_dtranspose_inplace_q( n, dA, ldda, magmablasGetQueue() );
}
#endif


/**
    @see magmablas_dtranspose_inplace_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dtranspose_inplace(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda )
{
    magmablas_dtranspose_inplace_q( n, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_dtrsm_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C"
void magmablas_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dtrsm_q( side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb,
                       magmablasGetQueue() );
}


/**
    @see magmablas_dtrsm_outofplace_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C"
void magmablas_dtrsm_outofplace(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb,
    magmaDouble_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDouble_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_dtrsm_outofplace_q( side, uplo, transA, diag, m, n, alpha,
                                  dA, ldda, dB, lddb, dX, lddx, flag,
                                  d_dinvA, dinvA_length, magmablasGetQueue() );
}


/**
    @see magmablas_dtrsm_work_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C"
void magmablas_dtrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    double alpha,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb,
    magmaDouble_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDouble_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_dtrsm_work_q( side, uplo, transA, diag, m, n, alpha,
                            dA, ldda, dB, lddb, dX, lddx, flag,
                            d_dinvA, dinvA_length, magmablasGetQueue() );
}


/**
    @see magmablas_dtrtri_diag_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C" void
magmablas_dtrtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr d_dinvA)
{
    magmablas_dtrtri_diag_q( uplo, diag, n, dA, ldda, d_dinvA, magmablasGetQueue() );
}


/**
    @see magma_dgetmatrix_1D_row_bcyclic_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C" void
magma_dgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const *dA, magma_int_t ldda,
    double                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_dgetmatrix_1D_row_bcyclic_q( m, n, dA, ldda, hA, lda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_dgetmatrix_1D_col_bcyclic_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C" void
magma_dgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const *dA, magma_int_t ldda,
    double                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_dgetmatrix_1D_col_bcyclic_q( m, n, dA, ldda, hA, lda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_dsetmatrix_1D_row_bcyclic_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C" void
magma_dsetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const double    *hA, magma_int_t lda,
    magmaDouble_ptr      *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_dsetmatrix_1D_row_bcyclic_q( m, n, hA, lda, dA, ldda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_dsetmatrix_1D_col_bcyclic_q
    @ingroup magma_dblas3
    ********************************************************************/
extern "C" void
magma_dsetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const double *hA, magma_int_t lda,
    magmaDouble_ptr   *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_dsetmatrix_1D_col_bcyclic_q( m, n, hA, lda, dA, ldda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


// in src/dlarfb_gpu.cpp
/**
    @see magma_dlarfb_gpu_q
    @ingroup magma_daux3
    ********************************************************************/
extern "C" magma_int_t
magma_dlarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_const_ptr dV,    magma_int_t lddv,
    magmaDouble_const_ptr dT,    magma_int_t lddt,
    magmaDouble_ptr dC,          magma_int_t lddc,
    magmaDouble_ptr dwork,       magma_int_t ldwork )
{
    return magma_dlarfb_gpu_q( side, trans, direct, storev,
                               m, n, k,
                               dV, lddv, dT, lddt, dC, lddc, dwork, ldwork,
                               magmablasGetQueue() );
}


// in src/dlarfb_gpu_gemm.cpp
/**
    @see magma_dlarfb_gpu_gemm_q
    @ingroup magma_daux3
    ********************************************************************/
extern "C" magma_int_t
magma_dlarfb_gpu_gemm(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_const_ptr dV,    magma_int_t lddv,
    magmaDouble_const_ptr dT,    magma_int_t lddt,
    magmaDouble_ptr dC,          magma_int_t lddc,
    magmaDouble_ptr dwork,       magma_int_t ldwork,
    magmaDouble_ptr dworkvt,     magma_int_t ldworkvt )
{
    return magma_dlarfb_gpu_gemm_q( side, trans, direct, storev,
                                    m, n, k,
                                    dV, lddv, dT, lddt, dC, lddc,
                                    dwork, ldwork, dworkvt, ldworkvt,
                                    magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
