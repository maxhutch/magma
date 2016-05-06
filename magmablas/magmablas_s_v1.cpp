/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/magmablas_z_v1.cpp normal z -> s, Mon May  2 23:30:37 2016

       @author Mark Gates

       Implements all the wrappers for v1 backwards compatability.
       Separating the wrappers allows the new functions to use magma_internal.h
*/
#ifndef MAGMA_NO_V1

#include "common_magma.h"

#define REAL


/**
    @see magmablas_saxpycp_q
    @ingroup magma_sblas1
    ********************************************************************/
extern "C" void
magmablas_saxpycp(
    magma_int_t m,
    magmaFloat_ptr r,
    magmaFloat_ptr x,
    magmaFloat_const_ptr b)
{
    magmablas_saxpycp_q( m, r, x, b, magmablasGetQueue() );
}


/**
    @see magmablas_sgeadd_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_sgeadd(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_sgeadd_q( m, n, alpha, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_sgeadd2_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_sgeadd2(
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_sgeadd2_q( m, n, alpha, dA, ldda, beta, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_sgemm_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C" void
magmablas_sgemm(
    magma_trans_t transA, magma_trans_t transB, magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc )
{
    magmablas_sgemm_q( transA, transB, m, n, k,
                       alpha, dA, ldda,
                              dB, lddb,
                       beta,  dC, lddc, magmablasGetQueue() );
}


/**
    @see magmablas_sgemv_q
    @ingroup magma_sblas2
    ********************************************************************/
extern "C" void
magmablas_sgemv(
    magma_trans_t trans, magma_int_t m, magma_int_t n, float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy)
{
    magmablas_sgemv_q( trans, m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_sgemv_conj_q
    @ingroup magma_sblas2
    ********************************************************************/
extern "C" void
magmablas_sgemv_conj(
    magma_int_t m, magma_int_t n, float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy)
{
    magmablas_sgemv_conj_q(
        m, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_sgemm_reduce_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C" void
magmablas_sgemm_reduce(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloat_ptr       dC, magma_int_t lddc )
{
    magmablas_sgemm_reduce_q(
        m, n, k, alpha, dA, ldda, dB, lddb, beta, dC, lddc, magmablasGetQueue() );
}


// @see magmablas_sgetmatrix_transpose_q
extern "C" void
magmablas_sgetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dAT, magma_int_t ldda,
    float          *hA,  magma_int_t lda,
    magmaFloat_ptr       dwork,  magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_sgetmatrix_transpose_q( m, n, dAT, ldda, hA, lda, dwork, lddwork, nb, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/**
    @see magmablas_ssymv_q
    @ingroup magma_sblas2
    ********************************************************************/
extern "C" magma_int_t
magmablas_ssymv(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy )
{
    return magmablas_ssymv_q( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}


#ifdef COMPLEX
/**
    @see magmablas_ssymv_q
    @ingroup magma_sblas2
    ********************************************************************/
extern "C" magma_int_t
magmablas_ssymv(
    magma_uplo_t uplo, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr       dy, magma_int_t incy )
{
    return magmablas_ssymv_q( uplo, n, alpha, dA, ldda, dx, incx, beta, dy, incy, magmablasGetQueue() );
}
#endif


/**
    @see magmablas_sprbt_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_sprbt(
    magma_int_t n,
    float *dA, magma_int_t ldda,
    float *du, float *dv)
{
    magmablas_sprbt_q(n, dA, ldda, du, dv, magmablasGetQueue() );
}


/**
    @see magmablas_sprbt_mtv_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_sprbt_mv(
    magma_int_t n,
    float *dv, float *db)
{
    magmablas_sprbt_mv_q(n, dv, db, magmablasGetQueue() );
}


/**
    @see magmablas_sprbt_mtv_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_sprbt_mtv(
    magma_int_t n,
    float *du, float *db)
{
    magmablas_sprbt_mtv_q(n, du, db, magmablasGetQueue() );
}


/**
    @see magmablas_slacpy_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_slacpy_q( uplo, m, n, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_slacpy_conj_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slacpy_conj(
    magma_int_t n,
    magmaFloat_ptr dA1, magma_int_t lda1,
    magmaFloat_ptr dA2, magma_int_t lda2)
{
    magmablas_slacpy_conj_q( n, dA1, lda1, dA2, lda2, magmablasGetQueue() );
}


/**
    @see magmablas_slacpy_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slacpy_sym_in(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_slacpy_sym_in_q( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_slacpy_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slacpy_sym_out(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_slacpy_sym_out_q( uplo, m, n, rows, perm, dA, ldda, dB, lddb, magmablasGetQueue() );
}


/**
    @see magmablas_slange_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" float
magmablas_slange(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork )
{
    return magmablas_slange_q( norm, m, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/**
    @see magmablas_slansy_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" float
magmablas_slansy(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, magma_int_t lwork )
{
    return magmablas_slansy_q( norm, uplo, n, dA, ldda, dwork, lwork, magmablasGetQueue() );
}


/**
    @see magma_slarfx_gpu_q
    @ingroup magma_saux1
    ********************************************************************/
extern "C" void
magma_slarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr v,
    magmaFloat_ptr tau,
    magmaFloat_ptr C, magma_int_t ldc,
    magmaFloat_ptr        xnorm,
    magmaFloat_ptr dT, magma_int_t iter,
    magmaFloat_ptr work )
{
    magma_slarfx_gpu_q(m, n, v, tau, C, ldc, xnorm, dT, iter, work,
                       magmablasGetQueue());
}


/**
    @see magma_slarfbx_gpu
    @ingroup magma_saux3
    ********************************************************************/
extern "C" void
magma_slarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaFloat_ptr V,  magma_int_t ldv,
    magmaFloat_ptr dT, magma_int_t ldt,
    magmaFloat_ptr c,
    magmaFloat_ptr dwork)
{
    magma_slarfbx_gpu_q( m, k, V, ldv, dT, ldt, c, dwork, magmablasGetQueue() );
}


/**
    @see magmablas_slarfg_q
    @ingroup magma_saux1
    ********************************************************************/
extern "C"
void magmablas_slarfg(
    magma_int_t n,
    magmaFloat_ptr dalpha,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dtau )
{
    magmablas_slarfg_q( n, dalpha, dx, incx, dtau, magmablasGetQueue() );
}


/**
    @see magma_slarfg_gpu_q
    @ingroup magma_saux1
    ********************************************************************/
extern "C" void
magma_slarfg_gpu(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dAkk )
{
    magma_slarfg_gpu_q( n, dx0, dx, dtau, dxnorm, dAkk, magmablasGetQueue() );
}


/**
    @see magma_slarfgx_gpu_q
    @ingroup magma_saux1
    ********************************************************************/
extern "C" void
magma_slarfgx_gpu(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dA, magma_int_t iter)
{
    magma_slarfgx_gpu_q( n, dx0, dx, dtau, dxnorm, dA, iter, magmablasGetQueue() );
}


/**
    @see magma_slarfgtx_gpu_q
    @ingroup magma_saux1
    ********************************************************************/
extern "C" void
magma_slarfgtx_gpu(
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
    magma_slarfgtx_gpu_q(n, dx0, dx, dtau, dxnorm, dA, iter, V, ldv,
                         T, ldt, dwork, magmablasGetQueue() );
}


/**
    @see magmablas_slascl_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    float cfrom, float cto,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_slascl_q( type, kl, ku, cfrom, cto, m, n, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_slascl2_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slascl2(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *info )
{
    magmablas_slascl2_q( type, m, n, dD, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_slascl2_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slascl_2x2(
    magma_type_t type, magma_int_t m,
    magmaFloat_const_ptr dW, magma_int_t lddw,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_slascl_2x2_q( type, m, dW, lddw, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_slascl_diag_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slascl_diag(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD, magma_int_t lddd,
    magmaFloat_ptr       dA, magma_int_t ldda,
    magma_int_t *info )
{
    magmablas_slascl_diag_q( type, m, n, dD, lddd, dA, ldda, magmablasGetQueue(), info );
}


/**
    @see magmablas_slaset_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C"
void magmablas_slaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda )
{
    magmablas_slaset_q( uplo, m, n, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_slaset_band_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slaset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    float offdiag, float diag,
    magmaFloat_ptr dA, magma_int_t ldda)
{
    magmablas_slaset_band_q(uplo, m, n, k, offdiag, diag, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_slaswp_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slaswp(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_slaswp_q( n, dAT, ldda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_slaswpx_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slaswpx(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_slaswpx_q( n, dA, ldx, ldy, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_slaswp2_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slaswp2(
    magma_int_t n,
    magmaFloat_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci )
{
    magmablas_slaswp2_q( n, dAT, ldda, k1, k2, d_ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_slaswpx_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slaswp_sym( magma_int_t n, float *dA, magma_int_t lda,
                      magma_int_t k1, magma_int_t k2,
                      const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_slaswp_sym_q( n, dA, lda, k1, k2, ipiv, inci, magmablasGetQueue() );
}


/**
    @see magmablas_snrm2_check_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_snrm2_check(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dlsticc ) 
{
    magmablas_snrm2_check_q( m, n, dA, ldda, dxnorm, dlsticc, magmablasGetQueue() );
}


/**
    @see magmablas_snrm2_adjust_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_snrm2_adjust(
    magma_int_t k,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dc )
{
    magmablas_snrm2_adjust_q( k, dxnorm, dc, magmablasGetQueue() );
}


/**
    @see magmablas_snrm2_row_check_adjust_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_snrm2_row_check_adjust(
    magma_int_t k, float tol,
    magmaFloat_ptr dxnorm,
    magmaFloat_ptr dxnorm2, 
    magmaFloat_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dlsticc )
{
    magmablas_snrm2_row_check_adjust_q( k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc, magmablasGetQueue() );
}


/**
    @see magmablas_snrm2_cols_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_snrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr dxnorm ) 
{
    magmablas_snrm2_cols_q( m, n, dA, ldda, dxnorm, magmablasGetQueue() );
}


/**
    @see magmablas_ssetmatrix_transpose_q
    @ingroup magma_sblas1
    ********************************************************************/
extern "C" void
magmablas_ssetmatrix_transpose(
    magma_int_t m, magma_int_t n,
    const float     *hA, magma_int_t lda,
    magmaFloat_ptr       dAT, magma_int_t ldda,
    magmaFloat_ptr     dwork, magma_int_t lddwork, magma_int_t nb )
{
    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    magmablas_ssetmatrix_transpose_q( m, n, hA, lda, dAT, ldda, dwork, lddwork, nb, queues );

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
}


/**
    @see magmablas_sswap_q
    @ingroup magma_sblas1
    ********************************************************************/
extern "C" void
magmablas_sswap(
    magma_int_t n,
    magmaFloat_ptr dx, magma_int_t incx,
    magmaFloat_ptr dy, magma_int_t incy)
{
    magmablas_sswap_q( n, dx, incx, dy, incy, magmablasGetQueue() );
}


/**
    @see magmablas_sswapblk_q
    @ingroup magma_sblas2
    ********************************************************************/
extern "C" void
magmablas_sswapblk(
    magma_order_t order, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci, magma_int_t offset )
{
    magmablas_sswapblk_q(
        order, n, dA, ldda, dB, lddb, i1, i2, ipiv, inci, offset, magmablasGetQueue() );
}


/**
    @see magmablas_sswapdblk_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_sswapdblk(
    magma_int_t n, magma_int_t nb,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaFloat_ptr dB, magma_int_t lddb, magma_int_t incb )
{
    magmablas_sswapdblk_q( n, nb, dA, ldda, inca, dB, lddb, incb, magmablasGetQueue() );
}


/**
    @see magmablas_ssymmetrize_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_ssymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda )
{
    magmablas_ssymmetrize_q( uplo, m, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_ssymmetrize_tiles_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_ssymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride )
{
    magmablas_ssymmetrize_tiles_q( uplo, m, dA, ldda, ntile, mstride, nstride, magmablasGetQueue() );
}


/**
    @see magmablas_stranspose_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_stranspose(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat )
{
    magmablas_stranspose_q( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


#ifdef COMPLEX
/**
    @see magmablas_stranspose_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_stranspose(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA,  magma_int_t ldda,
    magmaFloat_ptr       dAT, magma_int_t lddat )
{
    magmablas_stranspose_q( m, n, dA, ldda, dAT, lddat, magmablasGetQueue() );
}


/**
    @see magmablas_stranspose_inplace_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_stranspose_inplace(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda )
{
    magmablas_stranspose_inplace_q( n, dA, ldda, magmablasGetQueue() );
}
#endif


/**
    @see magmablas_stranspose_inplace_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_stranspose_inplace(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda )
{
    magmablas_stranspose_inplace_q( n, dA, ldda, magmablasGetQueue() );
}


/**
    @see magmablas_strsm_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C"
void magmablas_strsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb )
{
    magmablas_strsm_q( side, uplo, transA, diag, m, n, alpha, dA, ldda, dB, lddb,
                       magmablasGetQueue() );
}


/**
    @see magmablas_strsm_outofplace_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C"
void magmablas_strsm_outofplace(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magmaFloat_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloat_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_strsm_outofplace_q( side, uplo, transA, diag, m, n, alpha,
                                  dA, ldda, dB, lddb, dX, lddx, flag,
                                  d_dinvA, dinvA_length, magmablasGetQueue() );
}


/**
    @see magmablas_strsm_work_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C"
void magmablas_strsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr       dB, magma_int_t lddb,
    magmaFloat_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaFloat_ptr d_dinvA, magma_int_t dinvA_length )
{
    magmablas_strsm_work_q( side, uplo, transA, diag, m, n, alpha,
                            dA, ldda, dB, lddb, dX, lddx, flag,
                            d_dinvA, dinvA_length, magmablasGetQueue() );
}


/**
    @see magmablas_strtri_diag_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C" void
magmablas_strtri_diag(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_ptr d_dinvA)
{
    magmablas_strtri_diag_q( uplo, diag, n, dA, ldda, d_dinvA, magmablasGetQueue() );
}


/**
    @see magma_sgetmatrix_1D_row_bcyclic_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C" void
magma_sgetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr const *dA, magma_int_t ldda,
    float                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_sgetmatrix_1D_row_bcyclic_q( m, n, dA, ldda, hA, lda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_sgetmatrix_1D_col_bcyclic_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C" void
magma_sgetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr const *dA, magma_int_t ldda,
    float                 *hA, magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_sgetmatrix_1D_col_bcyclic_q( m, n, dA, ldda, hA, lda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_ssetmatrix_1D_row_bcyclic_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C" void
magma_ssetmatrix_1D_row_bcyclic(
    magma_int_t m, magma_int_t n,
    const float    *hA, magma_int_t lda,
    magmaFloat_ptr      *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_ssetmatrix_1D_row_bcyclic_q( m, n, hA, lda, dA, ldda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


/**
    @see magma_ssetmatrix_1D_col_bcyclic_q
    @ingroup magma_sblas3
    ********************************************************************/
extern "C" void
magma_ssetmatrix_1D_col_bcyclic(
    magma_int_t m, magma_int_t n,
    const float *hA, magma_int_t lda,
    magmaFloat_ptr   *dA, magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb )
{
    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    magma_ssetmatrix_1D_col_bcyclic_q( m, n, hA, lda, dA, ldda, ngpu, nb, queues );
    for( int dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        magma_queue_destroy( queues[dev] );
    }
}


// in src/slarfb_gpu.cpp
/**
    @see magma_slarfb_gpu_q
    @ingroup magma_saux3
    ********************************************************************/
extern "C" magma_int_t
magma_slarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV,    magma_int_t lddv,
    magmaFloat_const_ptr dT,    magma_int_t lddt,
    magmaFloat_ptr dC,          magma_int_t lddc,
    magmaFloat_ptr dwork,       magma_int_t ldwork )
{
    return magma_slarfb_gpu_q( side, trans, direct, storev,
                               m, n, k,
                               dV, lddv, dT, lddt, dC, lddc, dwork, ldwork,
                               magmablasGetQueue() );
}


// in src/slarfb_gpu_gemm.cpp
/**
    @see magma_slarfb_gpu_gemm_q
    @ingroup magma_saux3
    ********************************************************************/
extern "C" magma_int_t
magma_slarfb_gpu_gemm(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV,    magma_int_t lddv,
    magmaFloat_const_ptr dT,    magma_int_t lddt,
    magmaFloat_ptr dC,          magma_int_t lddc,
    magmaFloat_ptr dwork,       magma_int_t ldwork,
    magmaFloat_ptr dworkvt,     magma_int_t ldworkvt )
{
    return magma_slarfb_gpu_gemm_q( side, trans, direct, storev,
                                    m, n, k,
                                    dV, lddv, dT, lddt, dC, lddc,
                                    dwork, ldwork, dworkvt, ldworkvt,
                                    magmablasGetQueue() );
}

#endif // MAGMA_NO_V1
