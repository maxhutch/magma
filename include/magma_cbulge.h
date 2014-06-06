/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:17 2013
*/

#ifndef MAGMA_CBULGE_H
#define MAGMA_CBULGE_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif


magma_int_t magma_cbulge_applyQ_v2(char side, 
                              magma_int_t NE, magma_int_t N, 
                              magma_int_t NB, magma_int_t Vblksiz, 
                              magmaFloatComplex *dE, magma_int_t ldde, 
                              magmaFloatComplex *V, magma_int_t ldv, 
                              magmaFloatComplex *T, magma_int_t ldt, 
                              magma_int_t *info);
magma_int_t magma_cbulge_applyQ_v2_m(magma_int_t ngpu, char side, 
                              magma_int_t NE, magma_int_t N, 
                              magma_int_t NB, magma_int_t Vblksiz, 
                              magmaFloatComplex *E, magma_int_t lde, 
                              magmaFloatComplex *V, magma_int_t ldv, 
                              magmaFloatComplex *T, magma_int_t ldt, 
                              magma_int_t *info);

magma_int_t magma_cbulge_back( magma_int_t threads, char uplo, 
                              magma_int_t n, magma_int_t nb, 
                              magma_int_t ne, magma_int_t Vblksiz,
                              magmaFloatComplex *Z, magma_int_t ldz,
                              magmaFloatComplex *dZ, magma_int_t lddz,
                              magmaFloatComplex *V, magma_int_t ldv,
                              magmaFloatComplex *TAU,
                              magmaFloatComplex *T, magma_int_t ldt,
                              magma_int_t* info);
magma_int_t magma_cbulge_back_m(magma_int_t nrgpu, magma_int_t threads, char uplo, 
                              magma_int_t n, magma_int_t nb, 
                              magma_int_t ne, magma_int_t Vblksiz,
                              magmaFloatComplex *Z, magma_int_t ldz,
                              magmaFloatComplex *V, magma_int_t ldv, 
                              magmaFloatComplex *TAU, 
                              magmaFloatComplex *T, magma_int_t ldt, 
                              magma_int_t* info);

void magma_ctrdtype1cbHLsym_withQ_v2(magma_int_t n, magma_int_t nb, 
                              magmaFloatComplex *A, magma_int_t lda, 
                              magmaFloatComplex *V, magma_int_t ldv, 
                              magmaFloatComplex *TAU,
                              magma_int_t st, magma_int_t ed, 
                              magma_int_t sweep, magma_int_t Vblksiz, 
                              magmaFloatComplex *work);
void magma_ctrdtype2cbHLsym_withQ_v2(magma_int_t n, magma_int_t nb, 
                              magmaFloatComplex *A, magma_int_t lda, 
                              magmaFloatComplex *V, magma_int_t ldv, 
                              magmaFloatComplex *TAU,
                              magma_int_t st, magma_int_t ed, 
                              magma_int_t sweep, magma_int_t Vblksiz, 
                              magmaFloatComplex *work);
void magma_ctrdtype3cbHLsym_withQ_v2(magma_int_t n, magma_int_t nb, 
                              magmaFloatComplex *A, magma_int_t lda, 
                              magmaFloatComplex *V, magma_int_t ldv, 
                              magmaFloatComplex *TAU,
                              magma_int_t st, magma_int_t ed, 
                              magma_int_t sweep, magma_int_t Vblksiz, 
                              magmaFloatComplex *work);

magma_int_t magma_cunmqr_gpu_2stages(char side, char trans, magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaFloatComplex *dA, magma_int_t ldda,
                              magmaFloatComplex *dC, magma_int_t lddc,
                              magmaFloatComplex *dT, magma_int_t nb,
                              magma_int_t *info);

// used only for old version and internal
magma_int_t magma_chetrd_bhe2trc_v5(magma_int_t threads, magma_int_t wantz, char uplo, 
                              magma_int_t ne, magma_int_t n, magma_int_t nb,
                              magmaFloatComplex *A, magma_int_t lda, 
                              float *D, float *E,
                              magmaFloatComplex *dT1, magma_int_t ldt1);
magma_int_t magma_cungqr_2stage_gpu(magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaFloatComplex *da, magma_int_t ldda,
                              magmaFloatComplex *tau, magmaFloatComplex *dT,
                              magma_int_t nb, magma_int_t *info);





magma_int_t magma_cbulge_get_lq2(magma_int_t n, magma_int_t threads);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_CBULGE_H */
