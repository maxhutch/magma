/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/magmablas_z_v1.cpp normal z -> c, Mon May  2 23:30:37 2016

       @author Mark Gates

       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "common_magma.h"

#define COMPLEX


/**
    @see magmablas_caxpycp_q
    @ingroup magma_cblas1
    ********************************************************************/
extern "C" void
magmablas_caxpycp(
    magma_int_t m,
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr x,
    magmaFloatComplex_const_ptr b)
{
    magmablas_caxpycp_q( m, r, x, b, magmablasGetQueue() );
}


/**
    @see magmablas_cgeadd_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_cgeadd(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_cgeadd_q( m, n, alpha, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_cgeadd2_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_cgeadd2(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_cgeadd2_q( m, n, alpha, dA, ldda, beta, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_cgemm_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C" void
magmablas_cgemm(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magmablas_cgemm_q( transA, transB, m, n, k,
                       alpha, dA, ldda,
                              dB, lddb,
                       beta,  dC, lddc, magmablasGetQueue() );
}


/**
    @see magmablas_cgemv_q
    @ingroup magma_cblas2
    ********************************************************************/
extern "C" void
magmablas_cgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy, magma_int_t incy)
{
    magmablas_cgemv_q( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_cgemv_conj_q
    @ingroup magma_cblas2
    ********************************************************************/
extern "C" void
magmablas_cgemv_conj(
    magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy, magma_int_t incy)
{
    magmablas_cgemv_conj_q(
        m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_cgemm_reduce_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C" void
magmablas_cgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc )
{
    magmablas_cgemm_reduce_q(
        m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, magmablasGetQueue() );
}


// @see magmablas_cgetmatrix_transpose_q
extern "C" void
magmablas_cgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dAT, magma_int_t ldda,
    magmaFloatComplex          *hA,  magma_int_t lda,
    magmaFloatComplex_ptr       dwork,  magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_cgetmatrix_transpose_q( m, n, dAT, ldda, hA, lda, dwork, lddwork, nb, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/**
    @see magmablas_chemv_q
    @ingroup magma_cblas2
    ********************************************************************/
extern "C" magma_int_t
magmablas_chemv(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    return magmablas_chemv_q( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


#ifdef COMPLEX
/**
    @see magmablas_csymv_q
    @ingroup magma_cblas2
    ********************************************************************/
extern "C" magma_int_t
magmablas_csymv(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy )
{
    return magmablas_csymv_q( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}
#endif


/**
    @see magmablas_cprbt_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_cprbt(
    magma_int_t n,
    magmaFloatComplex *dA, magma_int_t ldda,
    magmaFloatComplex *du, magmaFloatComplex *dv)
{
    magmablas_cprbt_q(n, dA, ldda, du, dv, magmablasGetQueue() );
}


/**
    @see magmablas_cprbt_mtv_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_cprbt_mv(
    magma_int_t n,
    magmaFloatComplex *dv, magmaFloatComplex *db)
{
    magmablas_cprbt_mv_q(n, dv, db, magmablasGetQueue() );
}


/**
    @see magmablas_cprbt_mtv_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_cprbt_mtv(
    magma_int_t n,
    magmaFloatComplex *du, magmaFloatComplex *db)
{
    magmablas_cprbt_mtv_q(n, du, db, magmablasGetQueue() );
}


/**
    @see magmablas_clacpy_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_clacpy_q( uplo, m, n, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_clacpy_conj_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clacpy_conj(
    magma_int_t n,
    magmaFloatComplex_ptr dA1, magma_int_t lda1,
    magmaFloatComplex_ptr dA2, magma_int_t lda2)
{
    magmablas_clacpy_conj_q( n, dA1, lda1, dA2, lda2, magmablasGetQueue() );
}


/**
    @see magmablas_clacpy_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clacpy_sym_in(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_clacpy_sym_in_q( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_clacpy_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clacpy_sym_out(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_clacpy_sym_out_q( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_clange_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" float
magmablas_clange(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork )
{
    return magmablas_clange_q( norm, m, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/**
    @see magmablas_clanhe_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" float
magmablas_clanhe(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork )
{
    return magmablas_clanhe_q( norm, uplo, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/**
    @see magma_clarfx_gpu_q
    @ingroup magma_caux1
    ********************************************************************/
extern "C" void
magma_clarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr tau,
    magmaFloatComplex_ptr C, magma_int_t ldc,
    magmaFloat_ptr        xnorm,
    magmaFloatComplex_ptr dT, magma_int_t iter,
    magmaFloatComplex_ptr work )
{
    magma_clarfx_gpu_q(m, n, v, tau, C, ldc, xnorm, dT, iter, work,
                       magmablasGetQueue());
}


/**
    @see magma_clarfbx_gpu
    @ingroup magma_caux3
    ********************************************************************/
extern "C" void
magma_clarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr dT, magma_int_t ldt,
    magmaFloatComplex_ptr c,
    magmaFloatComplex_ptr dwork)
{
    magma_clarfbx_gpu_q( m, k, V, ldv, dT, ldt, c, dwork, magmablasGetQueue() );
}


/**
    @see magmablas_clarfg_q
    @ingroup magma_caux1
    ********************************************************************/
extern "C"
void magmablas_clarfg(
    magma_int_t n,
    magmaFloatComplex_ptr dalpha,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dtau )
{
    magmablas_clarfg_q( n, dalpha, dx, incx, dtau, magmablasGetQueue() );
}


/**
    @see magma_clarfg_gpu_q
    @ingroup magma_caux1
    ********************************************************************/
extern "C" void
magma_clarfg_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dAkk )
{
    magma_clarfg_gpu_q( n, dx0, dx, dtau, dxnorm, dAkk, magmablasGetQueue() );
}


/**
    @see magma_clarfgx_gpu_q
    @ingroup magma_caux1
    ********************************************************************/
extern "C" void
magma_clarfgx_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter)
{
    magma_clarfgx_gpu_q( n, dx0, dx, dtau, dxnorm, dA, iter, magmablasGetQueue() );
}


/**
    @see magma_clarfgtx_gpu_q
    @ingroup magma_caux1
    ********************************************************************/
extern "C" void
magma_clarfgtx_gpu(
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
    magma_clarfgtx_gpu_q(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv,
                         T, ldt, dwork, magmablasGetQueue() );
}


/**
    @see magmablas_clascl_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_clascl_q( type, kl, ku, cfrom, cto, m, n, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_clascl2_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clascl2(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t *info )
{
    magmablas_clascl2_q( type, m, n, dD, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_clascl2_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clascl_2x2(
    magma_type_t type, magma_int_t m,
    magmaFloatComplex_const_ptr dW, magma_int_t lddw,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_clascl_2x2_q( type, m, dW, lddw, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_clascl_diag_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dD, magma_int_t lddd,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_clascl_diag_q( type, m, n, dD, lddd, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_claset_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C"
void magmablas_claset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda )
{
    magmablas_claset_q( uplo, m, n, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_claset_band_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_claset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex offdiag, magmaFloatComplex diag,
    magmaFloatComplex_ptr dA, magma_int_t ldda)
{
    magmablas_claset_band_q(uplo, m, n, k, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_claswp_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_claswp(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_claswp_q( n, dAT, ldda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_claswpx_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_claswpx(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_claswpx_q( n, dA, ldx, ldy, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_claswp2_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_claswp2(
    magma_int_t n,
    magmaFloatComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci )
{
    magmablas_claswp2_q( n, dAT, ldda, k1, k2, d_ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_claswpx_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_claswp_sym( magma_int_t n, magmaFloatComplex *dA, magma_int_t lda,
                      magma_int_t k1, magma_int_t k2,
                      const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_claswp_sym_q( n, dA, lda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_scnrm2_check_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_scnrm2_check(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc ) 
{
    magmablas_scnrm2_check_q( m, n, dA, ldda, dxnorm, dlsticc, magmablasGetQueue() );
}


/**
    @see magmablas_scnrm2_adjust_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_scnrm2_adjust(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloatComplex_ptr dc )
{
    magmablas_scnrm2_adjust_q( k, dxnorm, dc, magmablasGetQueue() );
}


/**
    @see magmablas_scnrm2_row_check_adjust_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_scnrm2_row_check_adjust(
    magma_int_t k, float tol,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dxnorm2, 
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dlsticc )
{
    magmablas_scnrm2_row_check_adjust_q( k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc, magmablasGetQueue() );
}


/**
    @see magmablas_scnrm2_cols_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_scnrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr dxnorm ) 
{
    magmablas_scnrm2_cols_q( m, n, dA, ldda, dxnorm, magmablasGetQueue() );
}


/**
    @see magmablas_csetmatrix_transpose_q
    @ingroup magma_cblas1
    ********************************************************************/
extern "C" void
magmablas_csetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex     *hA, magma_int_t lda,
    magmaFloatComplex_ptr       dAT, magma_int_t ldda,
    magmaFloatComplex_ptr     dwork, magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_csetmatrix_transpose_q( m, n, hA, lda, dAT, ldda, dwork, lddwork, nb, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/**
    @see magmablas_cswap_q
    @ingroup magma_cblas1
    ********************************************************************/
extern "C" void
magmablas_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy)
{
    magmablas_cswap_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_cswapblk_q
    @ingroup magma_cblas2
    ********************************************************************/
extern "C" void
magmablas_cswapblk(
    magma_order_t order, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    magmablas_cswapblk_q(
        order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset, magmablasGetQueue() );
}


/**
    @see magmablas_cswapdblk_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_cswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloatComplex_ptr dB, magma_int_t lddb, magma_int_t incb )
{
    magmablas_cswapdblk_q( n, nb, dA, ldda, inca, dB, lddb, incb, magmablasGetQueue() );
}


/**
    @see magmablas_csymmetrize_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_csymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda )
{
    magmablas_csymmetrize_q( uplo, m, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_csymmetrize_tiles_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_csymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride )
{
    magmablas_csymmetrize_tiles_q( uplo, m, dA, ldda, ntile, mstride, nstride, magmablasGetQueue() );
}


/**
    @see magmablas_ctranspose_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_ctranspose(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, magma_int_t lddat )
{
    magmablas_ctranspose_q( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


#ifdef COMPLEX
/**
    @see magmablas_ctranspose_conj_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_ctranspose_conj(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA,  magma_int_t ldda,
    magmaFloatComplex_ptr       dAT, magma_int_t lddat )
{
    magmablas_ctranspose_conj_q( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


/**
    @see magmablas_ctranspose_conj_inplace_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_ctranspose_conj_inplace(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda )
{
    magmablas_ctranspose_conj_inplace_q( n, dA, ldda, magmablasGetQueue() );
}
#endif


/**
    @see magmablas_ctranspose_inplace_q
    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_ctranspose_inplace(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda )
{
    magmablas_ctranspose_inplace_q( n, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_ctrsm_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C"
void magmablas_ctrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb )
{
    magmablas_ctrsm_q( side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb,
                       magmablasGetQueue() );
}


/**
    @see magmablas_ctrsm_outofplace_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C"
void magmablas_ctrsm_outofplace(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magmaFloatComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloatComplex_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_ctrsm_outofplace_q( side, uplo, transA, diag, m, n, alpha,
                                  dA, ldda, dB, lddb, dX, lddx, flag,
                                  d_dinvA, dinvA_length, magmablasGetQueue() );
}


/**
    @see magmablas_ctrsm_work_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C"
void magmablas_ctrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magmaFloatComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloatComplex_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_ctrsm_work_q( side, uplo, transA, diag, m, n, alpha,
                            dA, ldda, dB, lddb, dX, lddx, flag,
                            d_dinvA, dinvA_length, magmablasGetQueue() );
}


/**
    @see magmablas_ctrtri_diag_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C" void
magmablas_ctrtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr d_dinvA)
{
    magmablas_ctrtri_diag_q( uplo, diag, n, dA, ldda, d_dinvA, magmablasGetQueue() );
}


/**
    @see magma_cgetmatrix_1D_row_bcyclic_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C" void
magma_cgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const *dA, magma_int_t ldda,
    magmaFloatComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_cgetmatrix_1D_row_bcyclic_q( m, n, dA, ldda, hA, lda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_cgetmatrix_1D_col_bcyclic_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C" void
magma_cgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr const *dA, magma_int_t ldda,
    magmaFloatComplex                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_cgetmatrix_1D_col_bcyclic_q( m, n, dA, ldda, hA, lda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_csetmatrix_1D_row_bcyclic_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C" void
magma_csetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex    *hA, magma_int_t lda,
    magmaFloatComplex_ptr      *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_csetmatrix_1D_row_bcyclic_q( m, n, hA, lda, dA, ldda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_csetmatrix_1D_col_bcyclic_q
    @ingroup magma_cblas3
    ********************************************************************/
extern "C" void
magma_csetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *hA, magma_int_t lda,
    magmaFloatComplex_ptr   *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_csetmatrix_1D_col_bcyclic_q( m, n, hA, lda, dA, ldda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


// in src/clarfb_gpu.cpp
/**
    @see magma_clarfb_gpu_q
    @ingroup magma_caux3
    ********************************************************************/
extern "C" magma_int_t
magma_clarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV,    magma_int_t lddv,
    magmaFloatComplex_const_ptr dT,    magma_int_t lddt,
    magmaFloatComplex_ptr dC,          magma_int_t lddc,
    magmaFloatComplex_ptr dwork,       magma_int_t ldwork )
{
    return magma_clarfb_gpu_q( side, trans, direct, storev,
                               m, n, k,
                               dV, lddv, dT, lddt, dC, lddc, dwork, ldwork,
                               magmablasGetQueue() );
}


// in src/clarfb_gpu_gemm.cpp
/**
    @see magma_clarfb_gpu_gemm_q
    @ingroup magma_caux3
    ********************************************************************/
extern "C" magma_int_t
magma_clarfb_gpu_gemm(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV,    magma_int_t lddv,
    magmaFloatComplex_const_ptr dT,    magma_int_t lddt,
    magmaFloatComplex_ptr dC,          magma_int_t lddc,
    magmaFloatComplex_ptr dwork,       magma_int_t ldwork,
    magmaFloatComplex_ptr dworkvt,     magma_int_t ldworkvt )
{
    return magma_clarfb_gpu_gemm_q( side, trans, direct, storev,
                                    m, n, k,
                                    dV, lddv, dT, lddt, dC, lddc,
                                    dwork, ldwork, dworkvt, ldworkvt,
                                    magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
