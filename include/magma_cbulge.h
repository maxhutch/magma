/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magma_zbulge.h, normal z -> c, Sun Nov 20 20:20:46 2016
*/

#ifndef MAGMA_CBULGE_H
#define MAGMA_CBULGE_H

#include "magma_types.h"
#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

magma_int_t
magma_cbulge_applyQ_v2(
    magma_side_t side, 
    magma_int_t NE, magma_int_t n, 
    magma_int_t nb, magma_int_t Vblksiz, 
    magmaFloatComplex_ptr dE, magma_int_t ldde, 
    magmaFloatComplex *V, magma_int_t ldv, 
    magmaFloatComplex *T, magma_int_t ldt, 
    magma_int_t *info);

magma_int_t
magma_cbulge_applyQ_v2_m(
    magma_int_t ngpu, magma_side_t side, 
    magma_int_t NE, magma_int_t n, 
    magma_int_t nb, magma_int_t Vblksiz, 
    magmaFloatComplex *E, magma_int_t lde, 
    magmaFloatComplex *V, magma_int_t ldv, 
    magmaFloatComplex *T, magma_int_t ldt, 
    magma_int_t *info);

magma_int_t
magma_cbulge_back(
    magma_uplo_t uplo, 
    magma_int_t n, magma_int_t nb, 
    magma_int_t ne, magma_int_t Vblksiz,
    magmaFloatComplex *Z, magma_int_t ldz,
    magmaFloatComplex_ptr dZ, magma_int_t lddz,
    magmaFloatComplex *V, magma_int_t ldv,
    magmaFloatComplex *TAU,
    magmaFloatComplex *T, magma_int_t ldt,
    magma_int_t* info);

magma_int_t
magma_cbulge_back_m(
    magma_int_t ngpu, magma_uplo_t uplo, 
    magma_int_t n, magma_int_t nb, 
    magma_int_t ne, magma_int_t Vblksiz,
    magmaFloatComplex *Z, magma_int_t ldz,
    magmaFloatComplex *V, magma_int_t ldv, 
    magmaFloatComplex *TAU, 
    magmaFloatComplex *T, magma_int_t ldt, 
    magma_int_t* info);

void
magma_ctrdtype1cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    magmaFloatComplex *A, magma_int_t lda, 
    magmaFloatComplex *V, magma_int_t ldv, 
    magmaFloatComplex *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    magmaFloatComplex *work);


void
magma_ctrdtype2cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    magmaFloatComplex *A, magma_int_t lda, 
    magmaFloatComplex *V, magma_int_t ldv, 
    magmaFloatComplex *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    magmaFloatComplex *work);


void
magma_ctrdtype3cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    magmaFloatComplex *A, magma_int_t lda, 
    magmaFloatComplex *V, magma_int_t ldv, 
    magmaFloatComplex *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    magmaFloatComplex *work);

void 
magma_clarfy(
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    const magmaFloatComplex *V, const magmaFloatComplex *TAU,
    magmaFloatComplex *work);

void
magma_chbtype1cb(magma_int_t n, magma_int_t nb,
                magmaFloatComplex *A, magma_int_t lda,
                magmaFloatComplex *V, magma_int_t LDV, 
                magmaFloatComplex *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep, 
                magma_int_t Vblksiz, magma_int_t wantz,
                magmaFloatComplex *work);

void
magma_chbtype2cb(magma_int_t n, magma_int_t nb,
                magmaFloatComplex *A, magma_int_t lda,
                magmaFloatComplex *V, magma_int_t ldv,
                magmaFloatComplex *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep,
                magma_int_t Vblksiz, magma_int_t wantz,
                magmaFloatComplex *work);
void
magma_chbtype3cb(magma_int_t n, magma_int_t nb,
                magmaFloatComplex *A, magma_int_t lda,
                magmaFloatComplex *V, magma_int_t ldv, 
                magmaFloatComplex *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep, 
                magma_int_t Vblksiz, magma_int_t wantz,
                magmaFloatComplex *work);


magma_int_t
magma_cunmqr_2stage_gpu(
    magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloatComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_get_cbulge_lq2( magma_int_t n, magma_int_t threads, magma_int_t wantz);

magma_int_t 
magma_cbulge_getstg2size(magma_int_t n, magma_int_t nb, magma_int_t wantz, 
                         magma_int_t Vblksiz, magma_int_t ldv, magma_int_t ldt, 
                         magma_int_t *blkcnt, magma_int_t *sizTAU2, 
                         magma_int_t *sizT2, magma_int_t *sizV2);


magma_int_t 
magma_cbulge_getlwstg2(magma_int_t n, magma_int_t threads, magma_int_t wantz, 
                       magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt, 
                       magma_int_t *blkcnt, magma_int_t *sizTAU2, 
                       magma_int_t *sizT2, magma_int_t *sizV2);


void 
magma_bulge_get_VTsiz(magma_int_t n, magma_int_t nb, magma_int_t threads, 
        magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt);
void 
magma_cheevdx_getworksize(magma_int_t n, magma_int_t threads,
        magma_int_t wantz, 
        magma_int_t *lwmin, 
        #ifdef COMPLEX
        magma_int_t *lrwmin, 
        #endif
        magma_int_t *liwmin);


// used only for old version and internal
magma_int_t
magma_chetrd_bhe2trc_v5(
    magma_int_t threads, magma_int_t wantz, magma_uplo_t uplo, 
    magma_int_t ne, magma_int_t n, magma_int_t nb,
    magmaFloatComplex *A, magma_int_t lda, 
    float *D, float *E,
    magmaFloatComplex_ptr dT1, magma_int_t ldt1);

magma_int_t
magma_cungqr_2stage_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT,
    magma_int_t nb,
    magma_int_t *info);


#ifdef __cplusplus
}
#endif
#undef COMPLEX
#endif /* MAGMA_CBULGE_H */
