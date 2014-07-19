/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from magma_zbulge.h normal z -> d, Fri Jul 18 17:34:10 2014
*/

#ifndef MAGMA_DBULGE_H
#define MAGMA_DBULGE_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif


magma_int_t magma_dbulge_applyQ_v2(magma_side_t side, 
                              magma_int_t NE, magma_int_t N, 
                              magma_int_t NB, magma_int_t Vblksiz, 
                              double *dE, magma_int_t ldde, 
                              double *V, magma_int_t ldv, 
                              double *T, magma_int_t ldt, 
                              magma_int_t *info);
magma_int_t magma_dbulge_applyQ_v2_m(magma_int_t ngpu, magma_side_t side, 
                              magma_int_t NE, magma_int_t N, 
                              magma_int_t NB, magma_int_t Vblksiz, 
                              double *E, magma_int_t lde, 
                              double *V, magma_int_t ldv, 
                              double *T, magma_int_t ldt, 
                              magma_int_t *info);

magma_int_t magma_dbulge_back( magma_uplo_t uplo, 
                              magma_int_t n, magma_int_t nb, 
                              magma_int_t ne, magma_int_t Vblksiz,
                              double *Z, magma_int_t ldz,
                              double *dZ, magma_int_t lddz,
                              double *V, magma_int_t ldv,
                              double *TAU,
                              double *T, magma_int_t ldt,
                              magma_int_t* info);
magma_int_t magma_dbulge_back_m(magma_int_t nrgpu, magma_uplo_t uplo, 
                              magma_int_t n, magma_int_t nb, 
                              magma_int_t ne, magma_int_t Vblksiz,
                              double *Z, magma_int_t ldz,
                              double *V, magma_int_t ldv, 
                              double *TAU, 
                              double *T, magma_int_t ldt, 
                              magma_int_t* info);

void magma_dtrdtype1cbHLsym_withQ_v2(magma_int_t n, magma_int_t nb, 
                              double *A, magma_int_t lda, 
                              double *V, magma_int_t ldv, 
                              double *TAU,
                              magma_int_t st, magma_int_t ed, 
                              magma_int_t sweep, magma_int_t Vblksiz, 
                              double *work);
void magma_dtrdtype2cbHLsym_withQ_v2(magma_int_t n, magma_int_t nb, 
                              double *A, magma_int_t lda, 
                              double *V, magma_int_t ldv, 
                              double *TAU,
                              magma_int_t st, magma_int_t ed, 
                              magma_int_t sweep, magma_int_t Vblksiz, 
                              double *work);
void magma_dtrdtype3cbHLsym_withQ_v2(magma_int_t n, magma_int_t nb, 
                              double *A, magma_int_t lda, 
                              double *V, magma_int_t ldv, 
                              double *TAU,
                              magma_int_t st, magma_int_t ed, 
                              magma_int_t sweep, magma_int_t Vblksiz, 
                              double *work);

magma_int_t magma_dormqr_gpu_2stages(magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k,
                              double *dA, magma_int_t ldda,
                              double *dC, magma_int_t lddc,
                              double *dT, magma_int_t nb,
                              magma_int_t *info);

// used only for old version and internal
magma_int_t magma_dsytrd_bsy2trc_v5(magma_int_t threads, magma_int_t wantz, magma_uplo_t uplo, 
                              magma_int_t ne, magma_int_t n, magma_int_t nb,
                              double *A, magma_int_t lda, 
                              double *D, double *E,
                              double *dT1, magma_int_t ldt1);
magma_int_t magma_dorgqr_2stage_gpu(magma_int_t m, magma_int_t n, magma_int_t k,
                              double *da, magma_int_t ldda,
                              double *tau, double *dT,
                              magma_int_t nb, magma_int_t *info);





magma_int_t magma_dbulge_get_lq2(magma_int_t n, magma_int_t threads);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_DBULGE_H */
