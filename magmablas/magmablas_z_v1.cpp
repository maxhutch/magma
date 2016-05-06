/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> c d s

       @author Mark Gates

       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "common_magma.h"

#define COMPLEX


/**
    @see magmablas_zaxpycp_q
    @ingroup magma_zblas1
    ********************************************************************/
extern "C" void
magmablas_zaxpycp(
    magma_int_t m,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_const_ptr b)
{
    magmablas_zaxpycp_q( m, r, x, b, magmablasGetQueue() );
}


/**
    @see magmablas_zgeadd_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zgeadd(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zgeadd_q( m, n, alpha, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_zgeadd2_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zgeadd2(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zgeadd2_q( m, n, alpha, dA, ldda, beta, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_zgemm_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C" void
magmablas_zgemm(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magmablas_zgemm_q( transA, transB, m, n, k,
                       alpha, dA, ldda,
                              dB, lddb,
                       beta,  dC, lddc, magmablasGetQueue() );
}


/**
    @see magmablas_zgemv_q
    @ingroup magma_zblas2
    ********************************************************************/
extern "C" void
magmablas_zgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy)
{
    magmablas_zgemv_q( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_zgemv_conj_q
    @ingroup magma_zblas2
    ********************************************************************/
extern "C" void
magmablas_zgemv_conj(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy)
{
    magmablas_zgemv_conj_q(
        m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_zgemm_reduce_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C" void
magmablas_zgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc )
{
    magmablas_zgemm_reduce_q(
        m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, magmablasGetQueue() );
}


// @see magmablas_zgetmatrix_transpose_q
extern "C" void
magmablas_zgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dAT, magma_int_t ldda,
    magmaDoubleComplex          *hA,  magma_int_t lda,
    magmaDoubleComplex_ptr       dwork,  magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_zgetmatrix_transpose_q( m, n, dAT, ldda, hA, lda, dwork, lddwork, nb, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/**
    @see magmablas_zhemv_q
    @ingroup magma_zblas2
    ********************************************************************/
extern "C" magma_int_t
magmablas_zhemv(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy )
{
    return magmablas_zhemv_q( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


#ifdef COMPLEX
/**
    @see magmablas_zsymv_q
    @ingroup magma_zblas2
    ********************************************************************/
extern "C" magma_int_t
magmablas_zsymv(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy )
{
    return magmablas_zsymv_q( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}
#endif


/**
    @see magmablas_zprbt_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zprbt(
    magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex *du, magmaDoubleComplex *dv)
{
    magmablas_zprbt_q(n, dA, ldda, du, dv, magmablasGetQueue() );
}


/**
    @see magmablas_zprbt_mtv_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zprbt_mv(
    magma_int_t n,
    magmaDoubleComplex *dv, magmaDoubleComplex *db)
{
    magmablas_zprbt_mv_q(n, dv, db, magmablasGetQueue() );
}


/**
    @see magmablas_zprbt_mtv_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zprbt_mtv(
    magma_int_t n,
    magmaDoubleComplex *du, magmaDoubleComplex *db)
{
    magmablas_zprbt_mtv_q(n, du, db, magmablasGetQueue() );
}


/**
    @see magmablas_zlacpy_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zlacpy_q( uplo, m, n, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_zlacpy_conj_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlacpy_conj(
    magma_int_t n,
    magmaDoubleComplex_ptr dA1, magma_int_t lda1,
    magmaDoubleComplex_ptr dA2, magma_int_t lda2)
{
    magmablas_zlacpy_conj_q( n, dA1, lda1, dA2, lda2, magmablasGetQueue() );
}


/**
    @see magmablas_zlacpy_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlacpy_sym_in(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zlacpy_sym_in_q( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_zlacpy_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlacpy_sym_out(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_zlacpy_sym_out_q( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_zlange_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" double
magmablas_zlange(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork )
{
    return magmablas_zlange_q( norm, m, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/**
    @see magmablas_zlanhe_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" double
magmablas_zlanhe(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork )
{
    return magmablas_zlanhe_q( norm, uplo, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/**
    @see magma_zlarfx_gpu_q
    @ingroup magma_zaux1
    ********************************************************************/
extern "C" void
magma_zlarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr tau,
    magmaDoubleComplex_ptr C, magma_int_t ldc,
    magmaDouble_ptr        xnorm,
    magmaDoubleComplex_ptr dT, magma_int_t iter,
    magmaDoubleComplex_ptr work )
{
    magma_zlarfx_gpu_q(m, n, v, tau, C, ldc, xnorm, dT, iter, work,
                       magmablasGetQueue());
}


/**
    @see magma_zlarfbx_gpu
    @ingroup magma_zaux3
    ********************************************************************/
extern "C" void
magma_zlarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr dT, magma_int_t ldt,
    magmaDoubleComplex_ptr c,
    magmaDoubleComplex_ptr dwork)
{
    magma_zlarfbx_gpu_q( m, k, V, ldv, dT, ldt, c, dwork, magmablasGetQueue() );
}


/**
    @see magmablas_zlarfg_q
    @ingroup magma_zaux1
    ********************************************************************/
extern "C"
void magmablas_zlarfg(
    magma_int_t n,
    magmaDoubleComplex_ptr dalpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dtau )
{
    magmablas_zlarfg_q( n, dalpha, dx, incx, dtau, magmablasGetQueue() );
}


/**
    @see magma_zlarfg_gpu_q
    @ingroup magma_zaux1
    ********************************************************************/
extern "C" void
magma_zlarfg_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dAkk )
{
    magma_zlarfg_gpu_q( n, dx0, dx, dtau, dxnorm, dAkk, magmablasGetQueue() );
}


/**
    @see magma_zlarfgx_gpu_q
    @ingroup magma_zaux1
    ********************************************************************/
extern "C" void
magma_zlarfgx_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter)
{
    magma_zlarfgx_gpu_q( n, dx0, dx, dtau, dxnorm, dA, iter, magmablasGetQueue() );
}


/**
    @see magma_zlarfgtx_gpu_q
    @ingroup magma_zaux1
    ********************************************************************/
extern "C" void
magma_zlarfgtx_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr T,  magma_int_t ldt,
    magmaDoubleComplex_ptr dwork )
{
    magma_zlarfgtx_gpu_q(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv,
                         T, ldt, dwork, magmablasGetQueue() );
}


/**
    @see magmablas_zlascl_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_zlascl_q( type, kl, ku, cfrom, cto, m, n, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_zlascl2_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlascl2(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t *info )
{
    magmablas_zlascl2_q( type, m, n, dD, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_zlascl2_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlascl_2x2(
    magma_type_t type, magma_int_t m,
    magmaDoubleComplex_const_ptr dW, magma_int_t lddw,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_zlascl_2x2_q( type, m, dW, lddw, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_zlascl_diag_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dD, magma_int_t lddd,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_zlascl_diag_q( type, m, n, dD, lddd, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_zlaset_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C"
void magmablas_zlaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda )
{
    magmablas_zlaset_q( uplo, m, n, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_zlaset_band_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda)
{
    magmablas_zlaset_band_q(uplo, m, n, k, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_zlaswp_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswp(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_zlaswp_q( n, dAT, ldda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_zlaswpx_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswpx(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_zlaswpx_q( n, dA, ldx, ldy, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_zlaswp2_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswp2(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci )
{
    magmablas_zlaswp2_q( n, dAT, ldda, k1, k2, d_ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_zlaswpx_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswp_sym( magma_int_t n, magmaDoubleComplex *dA, magma_int_t lda,
                      magma_int_t k1, magma_int_t k2,
                      const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_zlaswp_sym_q( n, dA, lda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_dznrm2_check_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_dznrm2_check(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc ) 
{
    magmablas_dznrm2_check_q( m, n, dA, ldda, dxnorm, dlsticc, magmablasGetQueue() );
}


/**
    @see magmablas_dznrm2_adjust_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_dznrm2_adjust(
    magma_int_t k,
    magmaDouble_ptr dxnorm,
    magmaDoubleComplex_ptr dc )
{
    magmablas_dznrm2_adjust_q( k, dxnorm, dc, magmablasGetQueue() );
}


/**
    @see magmablas_dznrm2_row_check_adjust_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_dznrm2_row_check_adjust(
    magma_int_t k, double tol,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dxnorm2, 
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dlsticc )
{
    magmablas_dznrm2_row_check_adjust_q( k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc, magmablasGetQueue() );
}


/**
    @see magmablas_dznrm2_cols_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_dznrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm ) 
{
    magmablas_dznrm2_cols_q( m, n, dA, ldda, dxnorm, magmablasGetQueue() );
}


/**
    @see magmablas_zsetmatrix_transpose_q
    @ingroup magma_zblas1
    ********************************************************************/
extern "C" void
magmablas_zsetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex     *hA, magma_int_t lda,
    magmaDoubleComplex_ptr       dAT, magma_int_t ldda,
    magmaDoubleComplex_ptr     dwork, magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_zsetmatrix_transpose_q( m, n, hA, lda, dAT, ldda, dwork, lddwork, nb, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/**
    @see magmablas_zswap_q
    @ingroup magma_zblas1
    ********************************************************************/
extern "C" void
magmablas_zswap(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy)
{
    magmablas_zswap_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_zswapblk_q
    @ingroup magma_zblas2
    ********************************************************************/
extern "C" void
magmablas_zswapblk(
    magma_order_t order, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    magmablas_zswapblk_q(
        order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset, magmablasGetQueue() );
}


/**
    @see magmablas_zswapdblk_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t incb )
{
    magmablas_zswapdblk_q( n, nb, dA, ldda, inca, dB, lddb, incb, magmablasGetQueue() );
}


/**
    @see magmablas_zsymmetrize_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zsymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda )
{
    magmablas_zsymmetrize_q( uplo, m, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_zsymmetrize_tiles_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zsymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride )
{
    magmablas_zsymmetrize_tiles_q( uplo, m, dA, ldda, ntile, mstride, nstride, magmablasGetQueue() );
}


/**
    @see magmablas_ztranspose_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_ztranspose(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA,  magma_int_t ldda,
    magmaDoubleComplex_ptr       dAT, magma_int_t lddat )
{
    magmablas_ztranspose_q( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


#ifdef COMPLEX
/**
    @see magmablas_ztranspose_conj_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_ztranspose_conj(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA,  magma_int_t ldda,
    magmaDoubleComplex_ptr       dAT, magma_int_t lddat )
{
    magmablas_ztranspose_conj_q( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


/**
    @see magmablas_ztranspose_conj_inplace_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_ztranspose_conj_inplace(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda )
{
    magmablas_ztranspose_conj_inplace_q( n, dA, ldda, magmablasGetQueue() );
}
#endif


/**
    @see magmablas_ztranspose_inplace_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_ztranspose_inplace(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda )
{
    magmablas_ztranspose_inplace_q( n, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_ztrsm_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C"
void magmablas_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_ztrsm_q( side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb,
                       magmablasGetQueue() );
}


/**
    @see magmablas_ztrsm_outofplace_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C"
void magmablas_ztrsm_outofplace(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_ztrsm_outofplace_q( side, uplo, transA, diag, m, n, alpha,
                                  dA, ldda, dB, lddb, dX, lddx, flag,
                                  d_dinvA, dinvA_length, magmablasGetQueue() );
}


/**
    @see magmablas_ztrsm_work_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C"
void magmablas_ztrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_ztrsm_work_q( side, uplo, transA, diag, m, n, alpha,
                            dA, ldda, dB, lddb, dX, lddx, flag,
                            d_dinvA, dinvA_length, magmablasGetQueue() );
}


/**
    @see magmablas_ztrtri_diag_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C" void
magmablas_ztrtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr d_dinvA)
{
    magmablas_ztrtri_diag_q( uplo, diag, n, dA, ldda, d_dinvA, magmablasGetQueue() );
}


/**
    @see magma_zgetmatrix_1D_row_bcyclic_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C" void
magma_zgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const *dA, magma_int_t ldda,
    magmaDoubleComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_zgetmatrix_1D_row_bcyclic_q( m, n, dA, ldda, hA, lda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_zgetmatrix_1D_col_bcyclic_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C" void
magma_zgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const *dA, magma_int_t ldda,
    magmaDoubleComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_zgetmatrix_1D_col_bcyclic_q( m, n, dA, ldda, hA, lda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_zsetmatrix_1D_row_bcyclic_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C" void
magma_zsetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex    *hA, magma_int_t lda,
    magmaDoubleComplex_ptr      *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_zsetmatrix_1D_row_bcyclic_q( m, n, hA, lda, dA, ldda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_zsetmatrix_1D_col_bcyclic_q
    @ingroup magma_zblas3
    ********************************************************************/
extern "C" void
magma_zsetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA, magma_int_t lda,
    magmaDoubleComplex_ptr   *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_zsetmatrix_1D_col_bcyclic_q( m, n, hA, lda, dA, ldda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


// in src/zlarfb_gpu.cpp
/**
    @see magma_zlarfb_gpu_q
    @ingroup magma_zaux3
    ********************************************************************/
extern "C" magma_int_t
magma_zlarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV,    magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT,    magma_int_t lddt,
    magmaDoubleComplex_ptr dC,          magma_int_t lddc,
    magmaDoubleComplex_ptr dwork,       magma_int_t ldwork )
{
    return magma_zlarfb_gpu_q( side, trans, direct, storev,
                               m, n, k,
                               dV, lddv, dT, lddt, dC, lddc, dwork, ldwork,
                               magmablasGetQueue() );
}


// in src/zlarfb_gpu_gemm.cpp
/**
    @see magma_zlarfb_gpu_gemm_q
    @ingroup magma_zaux3
    ********************************************************************/
extern "C" magma_int_t
magma_zlarfb_gpu_gemm(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV,    magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT,    magma_int_t lddt,
    magmaDoubleComplex_ptr dC,          magma_int_t lddc,
    magmaDoubleComplex_ptr dwork,       magma_int_t ldwork,
    magmaDoubleComplex_ptr dworkvt,     magma_int_t ldworkvt )
{
    return magma_zlarfb_gpu_gemm_q( side, trans, direct, storev,
                                    m, n, k,
                                    dV, lddv, dT, lddt, dC, lddc,
                                    dwork, ldwork, dworkvt, ldworkvt,
                                    magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
