/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from include/magma_zbulge.h, normal z -> d, Sun Nov 20 20:20:46 2016
*/

#ifndef MAGMA_DBULGE_H
#define MAGMA_DBULGE_H

#include "magma_types.h"
#define REAL

#ifdef __cplusplus
extern "C" {
#endif

magma_int_t
magma_dbulge_applyQ_v2(
    magma_side_t side, 
    magma_int_t NE, magma_int_t n, 
    magma_int_t nb, magma_int_t Vblksiz, 
    magmaDouble_ptr dE, magma_int_t ldde, 
    double *V, magma_int_t ldv, 
    double *T, magma_int_t ldt, 
    magma_int_t *info);

magma_int_t
magma_dbulge_applyQ_v2_m(
    magma_int_t ngpu, magma_side_t side, 
    magma_int_t NE, magma_int_t n, 
    magma_int_t nb, magma_int_t Vblksiz, 
    double *E, magma_int_t lde, 
    double *V, magma_int_t ldv, 
    double *T, magma_int_t ldt, 
    magma_int_t *info);

magma_int_t
magma_dbulge_back(
    magma_uplo_t uplo, 
    magma_int_t n, magma_int_t nb, 
    magma_int_t ne, magma_int_t Vblksiz,
    double *Z, magma_int_t ldz,
    magmaDouble_ptr dZ, magma_int_t lddz,
    double *V, magma_int_t ldv,
    double *TAU,
    double *T, magma_int_t ldt,
    magma_int_t* info);

magma_int_t
magma_dbulge_back_m(
    magma_int_t ngpu, magma_uplo_t uplo, 
    magma_int_t n, magma_int_t nb, 
    magma_int_t ne, magma_int_t Vblksiz,
    double *Z, magma_int_t ldz,
    double *V, magma_int_t ldv, 
    double *TAU, 
    double *T, magma_int_t ldt, 
    magma_int_t* info);

void
magma_dtrdtype1cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    double *A, magma_int_t lda, 
    double *V, magma_int_t ldv, 
    double *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    double *work);


void
magma_dtrdtype2cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    double *A, magma_int_t lda, 
    double *V, magma_int_t ldv, 
    double *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    double *work);


void
magma_dtrdtype3cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    double *A, magma_int_t lda, 
    double *V, magma_int_t ldv, 
    double *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    double *work);

void 
magma_dlarfy(
    magma_int_t n,
    double *A, magma_int_t lda,
    const double *V, const double *TAU,
    double *work);

void
magma_dsbtype1cb(magma_int_t n, magma_int_t nb,
                double *A, magma_int_t lda,
                double *V, magma_int_t LDV, 
                double *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep, 
                magma_int_t Vblksiz, magma_int_t wantz,
                double *work);

void
magma_dsbtype2cb(magma_int_t n, magma_int_t nb,
                double *A, magma_int_t lda,
                double *V, magma_int_t ldv,
                double *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep,
                magma_int_t Vblksiz, magma_int_t wantz,
                double *work);
void
magma_dsbtype3cb(magma_int_t n, magma_int_t nb,
                double *A, magma_int_t lda,
                double *V, magma_int_t ldv, 
                double *TAU,
                magma_int_t st, magma_int_t ed, magma_int_t sweep, 
                magma_int_t Vblksiz, magma_int_t wantz,
                double *work);


magma_int_t
magma_dormqr_2stage_gpu(
    magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_get_dbulge_lq2( magma_int_t n, magma_int_t threads, magma_int_t wantz);

magma_int_t 
magma_dbulge_getstg2size(magma_int_t n, magma_int_t nb, magma_int_t wantz, 
                         magma_int_t Vblksiz, magma_int_t ldv, magma_int_t ldt, 
                         magma_int_t *blkcnt, magma_int_t *sizTAU2, 
                         magma_int_t *sizT2, magma_int_t *sizV2);


magma_int_t 
magma_dbulge_getlwstg2(magma_int_t n, magma_int_t threads, magma_int_t wantz, 
                       magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt, 
                       magma_int_t *blkcnt, magma_int_t *sizTAU2, 
                       magma_int_t *sizT2, magma_int_t *sizV2);


void 
magma_bulge_get_VTsiz(magma_int_t n, magma_int_t nb, magma_int_t threads, 
        magma_int_t *Vblksiz, magma_int_t *ldv, magma_int_t *ldt);
void 
magma_dsyevdx_getworksize(magma_int_t n, magma_int_t threads,
        magma_int_t wantz, 
        magma_int_t *lwmin, 
        #ifdef COMPLEX
        magma_int_t *lrwmin, 
        #endif
        magma_int_t *liwmin);


// used only for old version and internal
magma_int_t
magma_dsytrd_bsy2trc_v5(
    magma_int_t threads, magma_int_t wantz, magma_uplo_t uplo, 
    magma_int_t ne, magma_int_t n, magma_int_t nb,
    double *A, magma_int_t lda, 
    double *D, double *E,
    magmaDouble_ptr dT1, magma_int_t ldt1);

magma_int_t
magma_dorgqr_2stage_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT,
    magma_int_t nb,
    magma_int_t *info);


#ifdef __cplusplus
}
#endif
#undef REAL
#endif /* MAGMA_DBULGE_H */
