/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_zbulge.h normal z -> s, Fri Jan 30 19:00:05 2015
*/

#ifndef MAGMA_SBULGE_H
#define MAGMA_SBULGE_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

magma_int_t
magma_sbulge_applyQ_v2(
    magma_side_t side, 
    magma_int_t NE, magma_int_t N, 
    magma_int_t NB, magma_int_t Vblksiz, 
    magmaFloat_ptr dE, magma_int_t ldde, 
    float *V, magma_int_t ldv, 
    float *T, magma_int_t ldt, 
    magma_int_t *info);

magma_int_t
magma_sbulge_applyQ_v2_m(
    magma_int_t ngpu, magma_side_t side, 
    magma_int_t NE, magma_int_t N, 
    magma_int_t NB, magma_int_t Vblksiz, 
    float *E, magma_int_t lde, 
    float *V, magma_int_t ldv, 
    float *T, magma_int_t ldt, 
    magma_int_t *info);

magma_int_t
magma_sbulge_back(
    magma_uplo_t uplo, 
    magma_int_t n, magma_int_t nb, 
    magma_int_t ne, magma_int_t Vblksiz,
    float *Z, magma_int_t ldz,
    magmaFloat_ptr dZ, magma_int_t lddz,
    float *V, magma_int_t ldv,
    float *TAU,
    float *T, magma_int_t ldt,
    magma_int_t* info);

magma_int_t
magma_sbulge_back_m(
    magma_int_t ngpu, magma_uplo_t uplo, 
    magma_int_t n, magma_int_t nb, 
    magma_int_t ne, magma_int_t Vblksiz,
    float *Z, magma_int_t ldz,
    float *V, magma_int_t ldv, 
    float *TAU, 
    float *T, magma_int_t ldt, 
    magma_int_t* info);

void
magma_strdtype1cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    float *A, magma_int_t lda, 
    float *V, magma_int_t ldv, 
    float *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    float *work);

void
magma_strdtype2cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    float *A, magma_int_t lda, 
    float *V, magma_int_t ldv, 
    float *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    float *work);

void
magma_strdtype3cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    float *A, magma_int_t lda, 
    float *V, magma_int_t ldv, 
    float *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    float *work);

magma_int_t
magma_sormqr_gpu_2stages(
    magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dT, magma_int_t nb,
    magma_int_t *info);

// used only for old version and internal
magma_int_t
magma_ssytrd_bsy2trc_v5(
    magma_int_t threads, magma_int_t wantz, magma_uplo_t uplo, 
    magma_int_t ne, magma_int_t n, magma_int_t nb,
    float *A, magma_int_t lda, 
    float *D, float *E,
    magmaFloat_ptr dT1, magma_int_t ldt1);

magma_int_t
magma_sorgqr_2stage_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dT,
    magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_sbulge_get_lq2(
    magma_int_t n, magma_int_t threads);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_SBULGE_H */
