/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magma_zbulge.h, normal z -> s, Sun Nov 20 20:20:47 2016
*/

#ifndef MAGMA_SBULGE_H
#define MAGMA_SBULGE_H

#include "magma_types.h"
#define REAL

#ifdef __cplusplus
extern "C" {
#endif

magma_int_t
magma_sbulge_applyQ_v2(
    magma_side_t side, 
    magma_int_t NE, magma_int_t n, 
    magma_int_t nb, magma_int_t Vblksiz, 
    magmaFloat_ptr dE, magma_int_t ldde, 
    float *V, magma_int_t ldv, 
    float *T, magma_int_t ldt, 
    magma_int_t *info);

magma_int_t
magma_sbulge_applyQ_v2_m(
    magma_int_t ngpu, magma_side_t side, 
    magma_int_t NE, magma_int_t n, 
    magma_int_t nb, magma_int_t Vblksiz, 
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

void 
magma_slarfy(
    magma_int_t n,
    float *A, magma_int_t lda,
    const float *V, const float *TAU,
    float *work);

void
magma_ssbtype1cb(magma_int_t n, magma_int_t nb,
                float *A, magma_int_t lda,
                float *V, magma_int_t LDV, 
                float *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep, 
                magma_int_t Vblksiz, magma_int_t wantz,
                float *work);

void
magma_ssbtype2cb(magma_int_t n, magma_int_t nb,
                float *A, magma_int_t lda,
                float *V, magma_int_t ldv,
                float *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep,
                magma_int_t Vblksiz, magma_int_t wantz,
                float *work);
void
magma_ssbtype3cb(magma_int_t n, magma_int_t nb,
                float *A, magma_int_t lda,
                float *V, magma_int_t ldv, 
                float *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep, 
                magma_int_t Vblksiz, magma_int_t wantz,
                float *work);


magma_int_t
magma_sormqr_2stage_gpu(
    magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dC, magma_int_t lddc,
    magmaFloat_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_get_sbulge_lq2( magma_int_t n, magma_int_t threads, magma_int_t wantz);

magma_int_t 
magma_sbulge_getstg2size(magma_int_t n, magma_int_t nb, magma_int_t wantz, 
                         magma_int_t Vblksiz, magma_int_t ldv, magma_int_t ldt, 
                         magma_int_t *blkcnt, magma_int_t *sizTAU2, 
                         magma_int_t *sizT2, magma_int_t *sizV2);


magma_int_t 
magma_sbulge_getlwstg2(magma_int_t n, magma_int_t threads, magma_int_t wantz, 
                       magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt, 
                       magma_int_t *blkcnt, magma_int_t *sizTAU2, 
                       magma_int_t *sizT2, magma_int_t *sizV2);


void 
magma_bulge_get_VTsiz(magma_int_t n, magma_int_t nb, magma_int_t threads, 
        magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt);
void 
magma_ssyevdx_getworksize(magma_int_t n, magma_int_t threads,
        magma_int_t wantz, 
        magma_int_t *lwmin, 
        #ifdef COMPLEX
        magma_int_t *lrwmin, 
        #endif
        magma_int_t *liwmin);


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


#ifdef __cplusplus
}
#endif
#undef REAL
#endif /* MAGMA_SBULGE_H */
