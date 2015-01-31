/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_z.h normal z -> c, Fri Jan 30 19:00:06 2015
*/

#ifndef MAGMA_C_H
#define MAGMA_C_H

#include "magma_types.h"
#include "magma_cgehrd_m.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
#ifdef REAL
magma_int_t magma_get_slaex3_m_nb();       // defined in slaex3_m.cpp
#endif

magma_int_t magma_get_cpotrf_nb( magma_int_t m );
magma_int_t magma_get_cgetrf_nb( magma_int_t m );
magma_int_t magma_get_cgetri_nb( magma_int_t m );
magma_int_t magma_get_cgeqp3_nb( magma_int_t m );
magma_int_t magma_get_cgeqrf_nb( magma_int_t m );
magma_int_t magma_get_cgeqlf_nb( magma_int_t m );
magma_int_t magma_get_cgehrd_nb( magma_int_t m );
magma_int_t magma_get_chetrd_nb( magma_int_t m );
magma_int_t magma_get_chetrf_nb( magma_int_t m );
magma_int_t magma_get_chetrf_nopiv_nb( magma_int_t m );
magma_int_t magma_get_cgelqf_nb( magma_int_t m );
magma_int_t magma_get_cgebrd_nb( magma_int_t m );
magma_int_t magma_get_chegst_nb( magma_int_t m );
magma_int_t magma_get_cgesvd_nb( magma_int_t m );
magma_int_t magma_get_chegst_nb_m( magma_int_t m );
magma_int_t magma_get_cbulge_nb( magma_int_t m, magma_int_t nbthreads );
magma_int_t magma_get_cbulge_nb_mgpu( magma_int_t m );
magma_int_t magma_cbulge_get_Vblksiz( magma_int_t m, magma_int_t nb, magma_int_t nbthreads );
magma_int_t magma_get_cbulge_gcperf();


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU (alphabetical order)
*/

#ifdef REAL
// only applicable to real [sd] precisions
void
magma_smove_eig(
    magma_range_t range, magma_int_t n, float *w,
    magma_int_t *il, magma_int_t *iu, float vl, float vu, magma_int_t *m);

// defined in slaex3.cpp
void
magma_cvrange(
    magma_int_t k, float *d, magma_int_t *il, magma_int_t *iu, float vl, float vu);

void
magma_cirange(
    magma_int_t k, magma_int_t *indxq, magma_int_t *iil, magma_int_t *iiu, magma_int_t il, magma_int_t iu);
#endif

magma_int_t
magma_cgebrd(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float *d, float *e,
    magmaFloatComplex *tauq, magmaFloatComplex *taup,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    #ifdef COMPLEX
    magmaFloatComplex *w,
    #else
    float *wr, float *wi,
    #endif
    magmaFloatComplex *VL, magma_int_t ldvl,
    magmaFloatComplex *VR, magma_int_t ldvr,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_cgehrd(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magmaFloatComplex_ptr dT,
    magma_int_t *info);

magma_int_t
magma_cgehrd2(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgelqf(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A,    magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgeqlf(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A,    magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgeqp3(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *jpvt, magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_cgeqrf(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgeqrf_ooc(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgeqrf4(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A,    magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgesdd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float *s,
    magmaFloatComplex *U, magma_int_t ldu,
    magmaFloatComplex *VT, magma_int_t ldvt,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *iwork,
    magma_int_t *info);

magma_int_t
magma_cgesv(
    magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magmaFloatComplex *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_cgesv_rbt(
    magma_bool_t ref, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex *A, magma_int_t lda, 
    magmaFloatComplex *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_cgesvd(
    magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
    magmaFloatComplex *A,    magma_int_t lda, float *s,
    magmaFloatComplex *U,    magma_int_t ldu,
    magmaFloatComplex *VT,   magma_int_t ldvt,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_cgetf2_nopiv(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_cgetrf(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_cgetrf_nopiv(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_cgetrf_piv(
    magma_int_t m, magma_int_t n, magma_int_t NB,
    magmaFloatComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_cgetrf2(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_cheevd(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_cheevdx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_cheevdx_2stage(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t
magma_cheevr(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float vl, float vu,
    magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
    float *w,
    magmaFloatComplex *Z, magma_int_t ldz,
    magma_int_t *isuppz,
    magmaFloatComplex *work, magma_int_t lwork,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// no real [sd] precisions available
magma_int_t
magma_cheevx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float vl, float vu,
    magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
    float *w,
    magmaFloatComplex *Z, magma_int_t ldz,
    magmaFloatComplex *work, magma_int_t lwork,
    float *rwork, magma_int_t *iwork,
    magma_int_t *ifail,
    magma_int_t *info);
#endif  // COMPLEX

magma_int_t
magma_chegst(
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_chegvd(
    magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    float *w, magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_chegvdx(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n, magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_chegvdx_2stage(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_chegvr(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    float abstol, magma_int_t *m, float *w,
    magmaFloatComplex *Z, magma_int_t ldz,
    magma_int_t *isuppz, magmaFloatComplex *work, magma_int_t lwork,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_chegvx(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n, magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    float abstol, magma_int_t *m, float *w,
    magmaFloatComplex *Z, magma_int_t ldz,
    magmaFloatComplex *work, magma_int_t lwork, float *rwork,
    magma_int_t *iwork, magma_int_t *ifail,
    magma_int_t *info);

magma_int_t
magma_chesv(magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
            magmaFloatComplex *A, magma_int_t lda, magma_int_t *ipiv,
            magmaFloatComplex *B, magma_int_t ldb,
            magma_int_t *info );

magma_int_t
magma_chetrd(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float *d, float *e, magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_chetrf(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_chetrf_nopiv(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_chetrd_hb2st(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
    magmaFloatComplex *A, magma_int_t lda,
    float *d, float *e,
    magmaFloatComplex *V, magma_int_t ldv,
    magmaFloatComplex *TAU, magma_int_t compT,
    magmaFloatComplex *T, magma_int_t ldt);

magma_int_t
magma_chetrd_he2hb(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magmaFloatComplex_ptr dT,
    magma_int_t *info);

#ifdef REAL
// only applicable to real [sd] precisions
magma_int_t
magma_slaex0(
    magma_int_t n, float *d, float *e,
    float *Q, magma_int_t ldq,
    float *work, magma_int_t *iwork,
    magmaFloat_ptr dwork,
    magma_range_t range, float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);

magma_int_t
magma_slaex1(
    magma_int_t n, float *d,
    float *Q, magma_int_t ldq,
    magma_int_t *indxq, float rho, magma_int_t cutpnt,
    float *work, magma_int_t *iwork,
    magmaFloat_ptr dwork,
    magma_range_t range, float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);

magma_int_t
magma_slaex3(
    magma_int_t k, magma_int_t n, magma_int_t n1, float *d,
    float *Q, magma_int_t ldq,
    float rho,
    float *dlamda, float *Q2, magma_int_t *indx,
    magma_int_t *ctot, float *w, float *s, magma_int_t *indxq,
    magmaFloat_ptr dwork,
    magma_range_t range, float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);
#endif  // REAL

magma_int_t
magma_clahef_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t *kb,
    magmaFloatComplex    *hA, magma_int_t lda,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaFloatComplex_ptr dW, magma_int_t lddw,
    magma_queue_t queues[], magma_event_t event[],
    magma_int_t *info);

magma_int_t
chetrf_nopiv_cpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t ib,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_chetrs_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_chesv_nopiv_gpu(
    magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs, 
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb, 
    magma_int_t *info);

magma_int_t
magma_clahr2(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dV, magma_int_t lddv,
    magmaFloatComplex *A,  magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *T,  magma_int_t ldt,
    magmaFloatComplex *Y,  magma_int_t ldy);

magma_int_t
magma_clahru(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    magmaFloatComplex     *A, magma_int_t lda,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dY, magma_int_t lddy,
    magmaFloatComplex_ptr dV, magma_int_t lddv,
    magmaFloatComplex_ptr dT,
    magmaFloatComplex_ptr dwork);

#ifdef REAL
magma_int_t
magma_slaln2(
    magma_int_t trans, magma_int_t na, magma_int_t nw,
    float smin, float ca, const float *A, magma_int_t lda,
    float d1, float d2,   const float *B, magma_int_t ldb,
    float wr, float wi, float *X, magma_int_t ldx,
    float *scale, float *xnorm,
    magma_int_t *info);
#endif

magma_int_t
magma_claqps(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloatComplex *A,  magma_int_t lda,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, magmaFloatComplex *tau, float *vn1, float *vn2,
    magmaFloatComplex *auxv,
    magmaFloatComplex *F,  magma_int_t ldf,
    magmaFloatComplex_ptr dF, magma_int_t lddf );

#ifdef REAL
magma_int_t
magma_claqtrsd(
    magma_trans_t trans, magma_int_t n,
    const float *T, magma_int_t ldt,
    float *x,       magma_int_t ldx,
    const float *cnorm,
    magma_int_t *info);
#endif

magma_int_t
magma_clatrd(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaFloatComplex *A, magma_int_t lda,
    float *e, magmaFloatComplex *tau,
    magmaFloatComplex *W, magma_int_t ldw,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dW, magma_int_t lddw);

magma_int_t
magma_clatrd2(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaFloatComplex *A,  magma_int_t lda,
    float *e, magmaFloatComplex *tau,
    magmaFloatComplex *W,  magma_int_t ldw,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dW, magma_int_t lddw,
    magmaFloatComplex_ptr dwork, magma_int_t ldwork);

#ifdef COMPLEX
magma_int_t
magma_clatrsd(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_diag_t diag, magma_bool_t normin,
    magma_int_t n, const magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex lambda,
    magmaFloatComplex *x,
    float *scale, float *cnorm,
    magma_int_t *info);
#endif

magma_int_t
magma_clauum(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_cposv(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_cpotrf(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_cpotri(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_cstedx(
    magma_range_t range, magma_int_t n, float vl, float vu,
    magma_int_t il, magma_int_t iu, float *d, float *e,
    magmaFloatComplex *Z, magma_int_t ldz,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magmaFloat_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_ctrevc3(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select, magma_int_t n,
    magmaFloatComplex *T,  magma_int_t ldt,
    magmaFloatComplex *VL, magma_int_t ldvl,
    magmaFloatComplex *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_ctrevc3_mt(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select, magma_int_t n,
    magmaFloatComplex *T,  magma_int_t ldt,
    magmaFloatComplex *VL, magma_int_t ldvl,
    magmaFloatComplex *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_ctrtri(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_cunghr(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_cungqr(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_cungqr2(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magma_int_t *info);

magma_int_t
magma_cunmbr(
    magma_vect_t vect, magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *C, magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cunmlq(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *C, magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

// not yet implemented
//magma_int_t magma_cunmrq( magma_side_t side, magma_trans_t trans,
//                          magma_int_t m, magma_int_t n, magma_int_t k,
//                          magmaFloatComplex *A, magma_int_t lda,
//                          magmaFloatComplex *tau,
//                          magmaFloatComplex *C, magma_int_t ldc,
//                          magmaFloatComplex *work, magma_int_t lwork,
//                          magma_int_t *info);

magma_int_t
magma_cunmql(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *C, magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cunmqr(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *C, magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cunmtr(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A,    magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *C,    magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU / Multi-GPU (alphabetical order)
*/
magma_int_t
magma_cgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    #ifdef COMPLEX
    magmaFloatComplex *w,
    #else
    float *wr, float *wi,
    #endif
    magmaFloatComplex *VL, magma_int_t ldvl,
    magmaFloatComplex *VR, magma_int_t ldvr,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_cgehrd_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magmaFloatComplex *T,
    magma_int_t *info);

magma_int_t
magma_cgetrf_m(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_cheevd_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_cheevdx_2stage_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_cheevdx_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_chegst_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_chegvd_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_chegvdx_2stage_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_chegvdx_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef REAL
magma_int_t
magma_slaex0_m(
    magma_int_t ngpu,
    magma_int_t n, float *d, float *e,
    float *Q, magma_int_t ldq,
    float *work, magma_int_t *iwork,
    magma_range_t range, float vl, float vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *info);

magma_int_t
magma_slaex1_m(
    magma_int_t ngpu,
    magma_int_t n, float *d,
    float *Q, magma_int_t ldq,
    magma_int_t *indxq, float rho, magma_int_t cutpnt,
    float *work, magma_int_t *iwork,
    magmaFloat_ptr dwork[],
    magma_queue_t queues[MagmaMaxGPUs][2],
    magma_range_t range, float vl, float vu,
    magma_int_t il, magma_int_t iu, magma_int_t *info);

magma_int_t
magma_slaex3_m(
    magma_int_t ngpu,
    magma_int_t k, magma_int_t n, magma_int_t n1, float *d,
    float *Q, magma_int_t ldq, float rho,
    float *dlamda, float *Q2, magma_int_t *indx,
    magma_int_t *ctot, float *w, float *s, magma_int_t *indxq,
    magmaFloat_ptr dwork[],
    magma_queue_t queues[MagmaMaxGPUs][2],
    magma_range_t range, float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);
#endif

magma_int_t
magma_clahr2_m(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *T, magma_int_t ldt,
    magmaFloatComplex *Y, magma_int_t ldy,
    struct cgehrd_data *data );

magma_int_t
magma_clahru_m(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    magmaFloatComplex *A, magma_int_t lda,
    struct cgehrd_data *data );

magma_int_t
magma_cpotrf_m(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_cstedx_m(
    magma_int_t ngpu,
    magma_range_t range, magma_int_t n, float vl, float vu,
    magma_int_t il, magma_int_t iu, float *d, float *e,
    magmaFloatComplex *Z, magma_int_t ldz,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ctrsm_m(
    magma_int_t ngpu,
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transa, magma_diag_t diag,
    magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb);

magma_int_t
magma_cunghr_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *T, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_cungqr_m(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *T, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_cunmqr_m(
    magma_int_t ngpu,
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *A,    magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *C,    magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cunmtr_m(
    magma_int_t ngpu,
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A,    magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *C,    magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on GPU (alphabetical order)
*/
magma_int_t
magma_cgegqr_gpu(
    magma_int_t ikind, magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dwork, magmaFloatComplex *work,
    magma_int_t *info);

magma_int_t
magma_cgelqf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgels_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgels3_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgeqp3_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, magmaFloatComplex *tau,
    magmaFloatComplex_ptr dwork, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_cgeqr2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dwork,
    magma_int_t *info);

magma_int_t
magma_cgeqr2x_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dtau,
    magmaFloatComplex_ptr dT, magmaFloatComplex_ptr ddA,
    magmaFloat_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_cgeqr2x2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dtau,
    magmaFloatComplex_ptr dT, magmaFloatComplex_ptr ddA,
    magmaFloat_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_cgeqr2x3_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dtau,
    magmaFloatComplex_ptr dT,
    magmaFloatComplex_ptr ddA,
    magmaFloat_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_cgeqr2x4_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dtau,
    magmaFloatComplex_ptr dT, magmaFloatComplex_ptr ddA,
    magmaFloat_ptr dwork,
    magma_queue_t queue,
    magma_int_t *info);

magma_int_t
magma_cgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT,
    magma_int_t *info);

magma_int_t
magma_cgeqrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magma_int_t *info);

magma_int_t
magma_cgeqrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dlA[], magma_int_t ldda,
    magmaFloatComplex *tau,
    magma_int_t *info);

magma_int_t
magma_cgeqrf3_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT,
    magma_int_t *info);

magma_int_t
magma_cgeqrs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgeqrs3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgerbt_gpu(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs, 
    magmaFloatComplex_ptr dA, magma_int_t ldda, 
    magmaFloatComplex_ptr dB, magma_int_t lddb, 
    magmaFloatComplex *U, magmaFloatComplex *V,
    magma_int_t *info);

magma_int_t
magma_cgessm_gpu(
    magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib,
    magma_int_t *ipiv,
    magmaFloatComplex_ptr dL1, magma_int_t lddl1,
    magmaFloatComplex_ptr dL,  magma_int_t lddl,
    magmaFloatComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_cgesv_gpu(
    magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_cgesv_nopiv_gpu( 
    magma_int_t n, magma_int_t nrhs, 
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb, 
                 magma_int_t *info);

magma_int_t
magma_cgetf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_cgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_cgetrf_incpiv_gpu(
    magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t ib,
    magmaFloatComplex    *hA, magma_int_t ldha,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex    *hL, magma_int_t ldhl,
    magmaFloatComplex_ptr dL, magma_int_t lddl,
    magma_int_t *ipiv,
    magmaFloatComplex_ptr dwork, magma_int_t lddwork,
    magma_int_t *info);

magma_int_t
magma_cgetrf_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr d_lA[], magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_cgetrf_nopiv_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_cgetrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
    magmaFloatComplex_ptr d_lAT[], magma_int_t lddat, magma_int_t *ipiv,
    magmaFloatComplex_ptr d_lAP[],
    magmaFloatComplex *W, magma_int_t ldw,
    magma_queue_t queues[][2],
    magma_int_t *info);

magma_int_t
magma_cgetri_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaFloatComplex_ptr dwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_cgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_cgetrs_nopiv_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_cheevd_gpu(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    float *w,
    magmaFloatComplex *wA,  magma_int_t ldwa,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_cheevdx_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    float vl, float vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    magmaFloatComplex *wA,  magma_int_t ldwa,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t
magma_cheevr_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    float vl, float vu,
    magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
    float *w,
    magmaFloatComplex_ptr dZ, magma_int_t lddz,
    magma_int_t *isuppz,
    magmaFloatComplex *wA, magma_int_t ldwa,
    magmaFloatComplex *wZ, magma_int_t ldwz,
    magmaFloatComplex *work, magma_int_t lwork,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_cheevx_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    float abstol, magma_int_t *m,
    float *w,
    magmaFloatComplex_ptr dZ, magma_int_t lddz,
    magmaFloatComplex *wA, magma_int_t ldwa,
    magmaFloatComplex *wZ, magma_int_t ldwz,
    magmaFloatComplex *work, magma_int_t lwork,
    float *rwork, magma_int_t *iwork,
    magma_int_t *ifail,
    magma_int_t *info);
#endif  // COMPLEX

magma_int_t
magma_chegst_gpu(
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_chetrd_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    float *d, float *e, magmaFloatComplex *tau,
    magmaFloatComplex *wA,  magma_int_t ldwa,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_chetrd_he2hb_mgpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magmaFloatComplex_ptr dAmgpu[], magma_int_t ldda,
    magmaFloatComplex_ptr dTmgpu[], magma_int_t lddt,
    magma_int_t ngpu, magma_int_t distblk,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_int_t *info);

magma_int_t
magma_chetrd_he2hb_mgpu_spec(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magmaFloatComplex_ptr dAmgpu[], magma_int_t ldda,
    magmaFloatComplex_ptr dTmgpu[], magma_int_t lddt,
    magma_int_t ngpu, magma_int_t distblk,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_int_t *info);

magma_int_t
magma_chetrd_mgpu(
    magma_int_t ngpu, magma_int_t nqueue,
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float *d, float *e, magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_chetrd2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    float *d, float *e, magmaFloatComplex *tau,
    magmaFloatComplex *wA,  magma_int_t ldwa,
    magmaFloatComplex *work, magma_int_t lwork,
    magmaFloatComplex_ptr dwork, magma_int_t ldwork,
    magma_int_t *info);

magma_int_t
magma_chetrf_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_clabrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    magmaFloatComplex     *A, magma_int_t lda,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    float *d, float *e, magmaFloatComplex *tauq, magmaFloatComplex *taup,
    magmaFloatComplex     *X, magma_int_t ldx,
    magmaFloatComplex_ptr dX, magma_int_t lddx,
    magmaFloatComplex     *Y, magma_int_t ldy,
    magmaFloatComplex_ptr dY, magma_int_t lddy );

magma_int_t
magma_claqps_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloatComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt, magmaFloatComplex *tau,
    float *vn1, float *vn2,
    magmaFloatComplex_ptr dauxv,
    magmaFloatComplex_ptr dF, magma_int_t lddf);

magma_int_t
magma_claqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloatComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr dvn1, magmaFloat_ptr dvn2,
    magmaFloatComplex_ptr dauxv,
    magmaFloatComplex_ptr dF, magma_int_t lddf);

magma_int_t
magma_claqps3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloatComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr dvn1, magmaFloat_ptr dvn2,
    magmaFloatComplex_ptr dauxv,
    magmaFloatComplex_ptr dF, magma_int_t lddf);

magma_int_t
magma_clarf_gpu(
    magma_int_t m,  magma_int_t n,
    magmaFloatComplex_const_ptr dv, magmaFloatComplex_const_ptr dtau,
    magmaFloatComplex_ptr dC,  magma_int_t lddc);

magma_int_t
magma_clarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV, magma_int_t lddv,
    magmaFloatComplex_const_ptr dT, magma_int_t lddt,
    magmaFloatComplex_ptr dC,       magma_int_t lddc,
    magmaFloatComplex_ptr dwork,    magma_int_t ldwork );

magma_int_t
magma_clarfb_gpu_gemm(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV, magma_int_t lddv,
    magmaFloatComplex_const_ptr dT, magma_int_t lddt,
    magmaFloatComplex_ptr dC,       magma_int_t lddc,
    magmaFloatComplex_ptr dwork,    magma_int_t ldwork,
    magmaFloatComplex_ptr dworkvt,  magma_int_t ldworkvt);

magma_int_t
magma_clarfb2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_const_ptr dV, magma_int_t lddv,
    magmaFloatComplex_const_ptr dT, magma_int_t lddt,
    magmaFloatComplex_ptr dC,       magma_int_t lddc,
    magmaFloatComplex_ptr dwork,    magma_int_t ldwork );

magma_int_t
magma_clatrd_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb, magma_int_t nb0,
    magmaFloatComplex *A,  magma_int_t lda,
    float *e, magmaFloatComplex *tau,
    magmaFloatComplex    *W,       magma_int_t ldw,
    magmaFloatComplex_ptr dA[],    magma_int_t ldda, magma_int_t offset,
    magmaFloatComplex_ptr dW[],    magma_int_t lddw,
    magmaFloatComplex    *hwork,   magma_int_t lhwork,
    magmaFloatComplex_ptr dwork[], magma_int_t ldwork,
    magma_queue_t queues[] );

magma_int_t
magma_clauum_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_cposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_cpotf2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_cpotrf_gpu(
    magma_uplo_t uplo,  magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_cpotrf_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_cpotrf_mgpu_right(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_cpotrf3_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaFloatComplex_ptr d_lA[],  magma_int_t ldda,
    magmaFloatComplex_ptr d_lP[],  magma_int_t lddp,
    magmaFloatComplex *A, magma_int_t lda, magma_int_t h,
    magma_queue_t queues[][3], magma_event_t events[][5],
    magma_int_t *info);

magma_int_t
magma_cpotri_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_cpotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_cssssm_gpu(
    magma_order_t order, magma_int_t m1, magma_int_t n1,
    magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib,
    magmaFloatComplex_ptr dA1, magma_int_t ldda1,
    magmaFloatComplex_ptr dA2, magma_int_t ldda2,
    magmaFloatComplex_ptr dL1, magma_int_t lddl1,
    magmaFloatComplex_ptr dL2, magma_int_t lddl2,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_ctrtri_gpu(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_ctsqrt_gpu(
    magma_int_t *m, magma_int_t *n,
    magmaFloatComplex *A1, magmaFloatComplex *A2, magma_int_t *lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t *lwork,
    magmaFloatComplex_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_ctstrf_gpu(
    magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
    magmaFloatComplex    *hU, magma_int_t ldhu,
    magmaFloatComplex_ptr dU, magma_int_t lddu,
    magmaFloatComplex    *hA, magma_int_t ldha,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex    *hL, magma_int_t ldhl,
    magmaFloatComplex_ptr dL, magma_int_t lddl,
    magma_int_t *ipiv,
    magmaFloatComplex *hwork, magma_int_t ldhwork,
    magmaFloatComplex_ptr dwork, magma_int_t lddwork,
    magma_int_t *info);

magma_int_t
magma_cungqr_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_cunmql2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloatComplex *wA, magma_int_t ldwa,
    magma_int_t *info);

magma_int_t
magma_cunmqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magmaFloatComplex_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_cunmqr2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloatComplex    *wA, magma_int_t ldwa,
    magma_int_t *info);

magma_int_t
magma_cunmtr_gpu(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex *tau,
    magmaFloatComplex_ptr dC, magma_int_t lddc,
    magmaFloatComplex    *wA, magma_int_t ldwa,
    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA utility function definitions
*/

extern const magmaFloatComplex MAGMA_C_NAN;
extern const magmaFloatComplex MAGMA_C_INF;

magma_int_t
magma_cnan_inf(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf );

magma_int_t
magma_cnan_inf_gpu(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf );

void magma_cprint(
    magma_int_t m, magma_int_t n,
    const magmaFloatComplex *A, magma_int_t lda );

void magma_cprint_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda );

void cpanel_to_q(
    magma_uplo_t uplo, magma_int_t ib,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *work );

void cq_to_panel(
    magma_uplo_t uplo, magma_int_t ib,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *work );

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif /* MAGMA_C_H */
