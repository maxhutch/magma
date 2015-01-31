/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c
*/

#ifndef MAGMA_ZBULGE_H
#define MAGMA_ZBULGE_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

magma_int_t
magma_zbulge_applyQ_v2(
    magma_side_t side, 
    magma_int_t NE, magma_int_t N, 
    magma_int_t NB, magma_int_t Vblksiz, 
    magmaDoubleComplex_ptr dE, magma_int_t ldde, 
    magmaDoubleComplex *V, magma_int_t ldv, 
    magmaDoubleComplex *T, magma_int_t ldt, 
    magma_int_t *info);

magma_int_t
magma_zbulge_applyQ_v2_m(
    magma_int_t ngpu, magma_side_t side, 
    magma_int_t NE, magma_int_t N, 
    magma_int_t NB, magma_int_t Vblksiz, 
    magmaDoubleComplex *E, magma_int_t lde, 
    magmaDoubleComplex *V, magma_int_t ldv, 
    magmaDoubleComplex *T, magma_int_t ldt, 
    magma_int_t *info);

magma_int_t
magma_zbulge_back(
    magma_uplo_t uplo, 
    magma_int_t n, magma_int_t nb, 
    magma_int_t ne, magma_int_t Vblksiz,
    magmaDoubleComplex *Z, magma_int_t ldz,
    magmaDoubleComplex_ptr dZ, magma_int_t lddz,
    magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *TAU,
    magmaDoubleComplex *T, magma_int_t ldt,
    magma_int_t* info);

magma_int_t
magma_zbulge_back_m(
    magma_int_t ngpu, magma_uplo_t uplo, 
    magma_int_t n, magma_int_t nb, 
    magma_int_t ne, magma_int_t Vblksiz,
    magmaDoubleComplex *Z, magma_int_t ldz,
    magmaDoubleComplex *V, magma_int_t ldv, 
    magmaDoubleComplex *TAU, 
    magmaDoubleComplex *T, magma_int_t ldt, 
    magma_int_t* info);

void
magma_ztrdtype1cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    magmaDoubleComplex *A, magma_int_t lda, 
    magmaDoubleComplex *V, magma_int_t ldv, 
    magmaDoubleComplex *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    magmaDoubleComplex *work);

void
magma_ztrdtype2cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    magmaDoubleComplex *A, magma_int_t lda, 
    magmaDoubleComplex *V, magma_int_t ldv, 
    magmaDoubleComplex *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    magmaDoubleComplex *work);

void
magma_ztrdtype3cbHLsym_withQ_v2(
    magma_int_t n, magma_int_t nb, 
    magmaDoubleComplex *A, magma_int_t lda, 
    magmaDoubleComplex *V, magma_int_t ldv, 
    magmaDoubleComplex *TAU,
    magma_int_t st, magma_int_t ed, 
    magma_int_t sweep, magma_int_t Vblksiz, 
    magmaDoubleComplex *work);

magma_int_t
magma_zunmqr_gpu_2stages(
    magma_side_t side, magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    magmaDoubleComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

// used only for old version and internal
magma_int_t
magma_zhetrd_bhe2trc_v5(
    magma_int_t threads, magma_int_t wantz, magma_uplo_t uplo, 
    magma_int_t ne, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex *A, magma_int_t lda, 
    double *D, double *E,
    magmaDoubleComplex_ptr dT1, magma_int_t ldt1);

magma_int_t
magma_zungqr_2stage_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex_ptr dT,
    magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_zbulge_get_lq2(
    magma_int_t n, magma_int_t threads);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA_ZBULGE_H */
