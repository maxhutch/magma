/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_z.h normal z -> s, Fri Jan 30 19:00:06 2015
*/

#ifndef MAGMA_S_H
#define MAGMA_S_H

#include "magma_types.h"
#include "magma_sgehrd_m.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
#ifdef REAL
magma_int_t magma_get_slaex3_m_nb();       // defined in slaex3_m.cpp
#endif

magma_int_t magma_get_spotrf_nb( magma_int_t m );
magma_int_t magma_get_sgetrf_nb( magma_int_t m );
magma_int_t magma_get_sgetri_nb( magma_int_t m );
magma_int_t magma_get_sgeqp3_nb( magma_int_t m );
magma_int_t magma_get_sgeqrf_nb( magma_int_t m );
magma_int_t magma_get_sgeqlf_nb( magma_int_t m );
magma_int_t magma_get_sgehrd_nb( magma_int_t m );
magma_int_t magma_get_ssytrd_nb( magma_int_t m );
magma_int_t magma_get_ssytrf_nb( magma_int_t m );
magma_int_t magma_get_ssytrf_nopiv_nb( magma_int_t m );
magma_int_t magma_get_sgelqf_nb( magma_int_t m );
magma_int_t magma_get_sgebrd_nb( magma_int_t m );
magma_int_t magma_get_ssygst_nb( magma_int_t m );
magma_int_t magma_get_sgesvd_nb( magma_int_t m );
magma_int_t magma_get_ssygst_nb_m( magma_int_t m );
magma_int_t magma_get_sbulge_nb( magma_int_t m, magma_int_t nbthreads );
magma_int_t magma_get_sbulge_nb_mgpu( magma_int_t m );
magma_int_t magma_sbulge_get_Vblksiz( magma_int_t m, magma_int_t nb, magma_int_t nbthreads );
magma_int_t magma_get_sbulge_gcperf();


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
magma_svrange(
    magma_int_t k, float *d, magma_int_t *il, magma_int_t *iu, float vl, float vu);

void
magma_sirange(
    magma_int_t k, magma_int_t *indxq, magma_int_t *iil, magma_int_t *iiu, magma_int_t il, magma_int_t iu);
#endif

magma_int_t
magma_sgebrd(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda,
    float *d, float *e,
    float *tauq, float *taup,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    float *A, magma_int_t lda,
    #ifdef COMPLEX
    float *w,
    #else
    float *wr, float *wi,
    #endif
    float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_sgehrd(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magmaFloat_ptr dT,
    magma_int_t *info);

magma_int_t
magma_sgehrd2(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgelqf(
    magma_int_t m, magma_int_t n,
    float *A,    magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgeqlf(
    magma_int_t m, magma_int_t n,
    float *A,    magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgeqp3(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *jpvt, float *tau,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_sgeqrf(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgeqrf_ooc(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgeqrf4(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    float *A,    magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgesdd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda,
    float *s,
    float *U, magma_int_t ldu,
    float *VT, magma_int_t ldvt,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *iwork,
    magma_int_t *info);

magma_int_t
magma_sgesv(
    magma_int_t n, magma_int_t nrhs,
    float *A, magma_int_t lda, magma_int_t *ipiv,
    float *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_sgesv_rbt(
    magma_bool_t ref, magma_int_t n, magma_int_t nrhs,
    float *A, magma_int_t lda, 
    float *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_sgesvd(
    magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
    float *A,    magma_int_t lda, float *s,
    float *U,    magma_int_t ldu,
    float *VT,   magma_int_t ldvt,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_sgetf2_nopiv(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_sgetrf(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_sgetrf_nopiv(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_sgetrf_piv(
    magma_int_t m, magma_int_t n, magma_int_t NB,
    float *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_sgetrf2(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_ssyevd(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssyevdx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssyevdx_2stage(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    float *A, magma_int_t lda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t
magma_ssyevr(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float vl, float vu,
    magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
    float *w,
    float *Z, magma_int_t ldz,
    magma_int_t *isuppz,
    float *work, magma_int_t lwork,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// no real [sd] precisions available
magma_int_t
magma_ssyevx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float vl, float vu,
    magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
    float *w,
    float *Z, magma_int_t ldz,
    float *work, magma_int_t lwork,
    float *rwork, magma_int_t *iwork,
    magma_int_t *ifail,
    magma_int_t *info);
#endif  // COMPLEX

magma_int_t
magma_ssygst(
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_ssygvd(
    magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    float *w, float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssygvdx(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n, float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssygvdx_2stage(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssygvr(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    float abstol, magma_int_t *m, float *w,
    float *Z, magma_int_t ldz,
    magma_int_t *isuppz, float *work, magma_int_t lwork,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssygvx(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n, float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    float abstol, magma_int_t *m, float *w,
    float *Z, magma_int_t ldz,
    float *work, magma_int_t lwork, float *rwork,
    magma_int_t *iwork, magma_int_t *ifail,
    magma_int_t *info);

magma_int_t
magma_ssysv(magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
            float *A, magma_int_t lda, magma_int_t *ipiv,
            float *B, magma_int_t ldb,
            magma_int_t *info );

magma_int_t
magma_ssytrd(
    magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float *d, float *e, float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_ssytrf(
    magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_ssytrf_nopiv(
    magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_ssytrd_sb2st(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
    float *A, magma_int_t lda,
    float *d, float *e,
    float *V, magma_int_t ldv,
    float *TAU, magma_int_t compT,
    float *T, magma_int_t ldt);

magma_int_t
magma_ssytrd_sy2sb(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magmaFloat_ptr dT,
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
magma_slasyf_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t *kb,
    float    *hA, magma_int_t lda,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaFloat_ptr dW, magma_int_t lddw,
    magma_queue_t queues[], magma_event_t event[],
    magma_int_t *info);

magma_int_t
ssytrf_nopiv_cpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t ib,
    float *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_ssytrs_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_ssysv_nopiv_gpu(
    magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs, 
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb, 
    magma_int_t *info);

magma_int_t
magma_slahr2(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dV, magma_int_t lddv,
    float *A,  magma_int_t lda,
    float *tau,
    float *T,  magma_int_t ldt,
    float *Y,  magma_int_t ldy);

magma_int_t
magma_slahru(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    float     *A, magma_int_t lda,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dY, magma_int_t lddy,
    magmaFloat_ptr dV, magma_int_t lddv,
    magmaFloat_ptr dT,
    magmaFloat_ptr dwork);

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
magma_slaqps(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    float *A,  magma_int_t lda,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, float *tau, float *vn1, float *vn2,
    float *auxv,
    float *F,  magma_int_t ldf,
    magmaFloat_ptr dF, magma_int_t lddf );

#ifdef REAL
magma_int_t
magma_slaqtrsd(
    magma_trans_t trans, magma_int_t n,
    const float *T, magma_int_t ldt,
    float *x,       magma_int_t ldx,
    const float *cnorm,
    magma_int_t *info);
#endif

magma_int_t
magma_slatrd(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    float *A, magma_int_t lda,
    float *e, float *tau,
    float *W, magma_int_t ldw,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dW, magma_int_t lddw);

magma_int_t
magma_slatrd2(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    float *A,  magma_int_t lda,
    float *e, float *tau,
    float *W,  magma_int_t ldw,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dW, magma_int_t lddw,
    magmaFloat_ptr dwork, magma_int_t ldwork);

#ifdef COMPLEX
magma_int_t
magma_slatrsd(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_diag_t diag, magma_bool_t normin,
    magma_int_t n, const float *A, magma_int_t lda,
    float lambda,
    float *x,
    float *scale, float *cnorm,
    magma_int_t *info);
#endif

magma_int_t
magma_slauum(
    magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_sposv(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_spotrf(
    magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_spotri(
    magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_sstedx(
    magma_range_t range, magma_int_t n, float vl, float vu,
    magma_int_t il, magma_int_t iu, float *d, float *e,
    float *Z, magma_int_t ldz,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magmaFloat_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_strevc3(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select, magma_int_t n,
    float *T,  magma_int_t ldt,
    float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_strevc3_mt(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select, magma_int_t n,
    float *T,  magma_int_t ldt,
    float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_strtri(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_sorghr(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    float *A, magma_int_t lda,
    float *tau,
    magmaFloat_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_sorgqr(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *A, magma_int_t lda,
    float *tau,
    magmaFloat_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_sorgqr2(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *A, magma_int_t lda,
    float *tau,
    magma_int_t *info);

magma_int_t
magma_sormbr(
    magma_vect_t vect, magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *A, magma_int_t lda,
    float *tau,
    float *C, magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sormlq(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *A, magma_int_t lda,
    float *tau,
    float *C, magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info);

// not yet implemented
//magma_int_t magma_sunmrq( magma_side_t side, magma_trans_t trans,
//                          magma_int_t m, magma_int_t n, magma_int_t k,
//                          float *A, magma_int_t lda,
//                          float *tau,
//                          float *C, magma_int_t ldc,
//                          float *work, magma_int_t lwork,
//                          magma_int_t *info);

magma_int_t
magma_sormql(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *A, magma_int_t lda,
    float *tau,
    float *C, magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sormqr(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *A, magma_int_t lda,
    float *tau,
    float *C, magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sormtr(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    float *A,    magma_int_t lda,
    float *tau,
    float *C,    magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU / Multi-GPU (alphabetical order)
*/
magma_int_t
magma_sgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    float *A, magma_int_t lda,
    #ifdef COMPLEX
    float *w,
    #else
    float *wr, float *wi,
    #endif
    float *VL, magma_int_t ldvl,
    float *VR, magma_int_t ldvr,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_sgehrd_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    float *T,
    magma_int_t *info);

magma_int_t
magma_sgetrf_m(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_ssyevd_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    float *A, magma_int_t lda,
    float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssyevdx_2stage_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    float *A, magma_int_t lda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssyevdx_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    float *A, magma_int_t lda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssygst_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_ssygvd_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssygvdx_2stage_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssygvdx_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    float *work, magma_int_t lwork,
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
magma_slahr2_m(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    float *A, magma_int_t lda,
    float *tau,
    float *T, magma_int_t ldt,
    float *Y, magma_int_t ldy,
    struct sgehrd_data *data );

magma_int_t
magma_slahru_m(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    float *A, magma_int_t lda,
    struct sgehrd_data *data );

magma_int_t
magma_spotrf_m(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_sstedx_m(
    magma_int_t ngpu,
    magma_range_t range, magma_int_t n, float vl, float vu,
    magma_int_t il, magma_int_t iu, float *d, float *e,
    float *Z, magma_int_t ldz,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_strsm_m(
    magma_int_t ngpu,
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transa, magma_diag_t diag,
    magma_int_t m, magma_int_t n, float alpha,
    float *A, magma_int_t lda,
    float *B, magma_int_t ldb);

magma_int_t
magma_sorghr_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    float *A, magma_int_t lda,
    float *tau,
    float *T, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_sorgqr_m(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *A, magma_int_t lda,
    float *tau,
    float *T, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_sormqr_m(
    magma_int_t ngpu,
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *A,    magma_int_t lda,
    float *tau,
    float *C,    magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sormtr_m(
    magma_int_t ngpu,
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    float *A,    magma_int_t lda,
    float *tau,
    float *C,    magma_int_t ldc,
    float *work, magma_int_t lwork,
    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on GPU (alphabetical order)
*/
magma_int_t
magma_sgegqr_gpu(
    magma_int_t ikind, magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dwork, float *work,
    magma_int_t *info);

magma_int_t
magma_sgelqf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgels_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    float *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgels3_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    float *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgeqp3_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, float *tau,
    magmaFloat_ptr dwork, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_sgeqr2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dwork,
    magma_int_t *info);

magma_int_t
magma_sgeqr2x_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dtau,
    magmaFloat_ptr dT, magmaFloat_ptr ddA,
    magmaFloat_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_sgeqr2x2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dtau,
    magmaFloat_ptr dT, magmaFloat_ptr ddA,
    magmaFloat_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_sgeqr2x3_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dtau,
    magmaFloat_ptr dT,
    magmaFloat_ptr ddA,
    magmaFloat_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_sgeqr2x4_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dtau,
    magmaFloat_ptr dT, magmaFloat_ptr ddA,
    magmaFloat_ptr dwork,
    magma_queue_t queue,
    magma_int_t *info);

magma_int_t
magma_sgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dT,
    magma_int_t *info);

magma_int_t
magma_sgeqrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magma_int_t *info);

magma_int_t
magma_sgeqrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dlA[], magma_int_t ldda,
    float *tau,
    magma_int_t *info);

magma_int_t
magma_sgeqrf3_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dT,
    magma_int_t *info);

magma_int_t
magma_sgeqrs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dT,
    magmaFloat_ptr dB, magma_int_t lddb,
    float *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgeqrs3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dT,
    magmaFloat_ptr dB, magma_int_t lddb,
    float *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgerbt_gpu(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs, 
    magmaFloat_ptr dA, magma_int_t ldda, 
    magmaFloat_ptr dB, magma_int_t lddb, 
    float *U, float *V,
    magma_int_t *info);

magma_int_t
magma_sgessm_gpu(
    magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib,
    magma_int_t *ipiv,
    magmaFloat_ptr dL1, magma_int_t lddl1,
    magmaFloat_ptr dL,  magma_int_t lddl,
    magmaFloat_ptr dA,  magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_sgesv_gpu(
    magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_sgesv_nopiv_gpu( 
    magma_int_t n, magma_int_t nrhs, 
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb, 
                 magma_int_t *info);

magma_int_t
magma_sgetf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_sgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_sgetrf_incpiv_gpu(
    magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t ib,
    float    *hA, magma_int_t ldha,
    magmaFloat_ptr dA, magma_int_t ldda,
    float    *hL, magma_int_t ldhl,
    magmaFloat_ptr dL, magma_int_t lddl,
    magma_int_t *ipiv,
    magmaFloat_ptr dwork, magma_int_t lddwork,
    magma_int_t *info);

magma_int_t
magma_sgetrf_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr d_lA[], magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_sgetrf_nopiv_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_sgetrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
    magmaFloat_ptr d_lAT[], magma_int_t lddat, magma_int_t *ipiv,
    magmaFloat_ptr d_lAP[],
    float *W, magma_int_t ldw,
    magma_queue_t queues[][2],
    magma_int_t *info);

magma_int_t
magma_sgetri_gpu(
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaFloat_ptr dwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_sgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_sgetrs_nopiv_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_ssyevd_gpu(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *w,
    float *wA,  magma_int_t ldwa,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssyevdx_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float vl, float vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *w,
    float *wA,  magma_int_t ldwa,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t
magma_ssyevr_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float vl, float vu,
    magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
    float *w,
    magmaFloat_ptr dZ, magma_int_t lddz,
    magma_int_t *isuppz,
    float *wA, magma_int_t ldwa,
    float *wZ, magma_int_t ldwz,
    float *work, magma_int_t lwork,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_ssyevx_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    float abstol, magma_int_t *m,
    float *w,
    magmaFloat_ptr dZ, magma_int_t lddz,
    float *wA, magma_int_t ldwa,
    float *wZ, magma_int_t ldwz,
    float *work, magma_int_t lwork,
    float *rwork, magma_int_t *iwork,
    magma_int_t *ifail,
    magma_int_t *info);
#endif  // COMPLEX

magma_int_t
magma_ssygst_gpu(
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_ssytrd_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *d, float *e, float *tau,
    float *wA,  magma_int_t ldwa,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_ssytrd_sy2sb_mgpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magmaFloat_ptr dAmgpu[], magma_int_t ldda,
    magmaFloat_ptr dTmgpu[], magma_int_t lddt,
    magma_int_t ngpu, magma_int_t distblk,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_int_t *info);

magma_int_t
magma_ssytrd_sy2sb_mgpu_spec(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    magmaFloat_ptr dAmgpu[], magma_int_t ldda,
    magmaFloat_ptr dTmgpu[], magma_int_t lddt,
    magma_int_t ngpu, magma_int_t distblk,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_int_t *info);

magma_int_t
magma_ssytrd_mgpu(
    magma_int_t ngpu, magma_int_t nqueue,
    magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda,
    float *d, float *e, float *tau,
    float *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_ssytrd2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *d, float *e, float *tau,
    float *wA,  magma_int_t ldwa,
    float *work, magma_int_t lwork,
    magmaFloat_ptr dwork, magma_int_t ldwork,
    magma_int_t *info);

magma_int_t
magma_ssytrf_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_slabrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    float     *A, magma_int_t lda,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *d, float *e, float *tauq, float *taup,
    float     *X, magma_int_t ldx,
    magmaFloat_ptr dX, magma_int_t lddx,
    float     *Y, magma_int_t ldy,
    magmaFloat_ptr dY, magma_int_t lddy );

magma_int_t
magma_slaqps_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloat_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt, float *tau,
    float *vn1, float *vn2,
    magmaFloat_ptr dauxv,
    magmaFloat_ptr dF, magma_int_t lddf);

magma_int_t
magma_slaqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloat_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaFloat_ptr dtau,
    magmaFloat_ptr dvn1, magmaFloat_ptr dvn2,
    magmaFloat_ptr dauxv,
    magmaFloat_ptr dF, magma_int_t lddf);

magma_int_t
magma_slaqps3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloat_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaFloat_ptr dtau,
    magmaFloat_ptr dvn1, magmaFloat_ptr dvn2,
    magmaFloat_ptr dauxv,
    magmaFloat_ptr dF, magma_int_t lddf);

magma_int_t
magma_slarf_gpu(
    magma_int_t m,  magma_int_t n,
    magmaFloat_const_ptr dv, magmaFloat_const_ptr dtau,
    magmaFloat_ptr dC,  magma_int_t lddc);

magma_int_t
magma_slarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV, magma_int_t lddv,
    magmaFloat_const_ptr dT, magma_int_t lddt,
    magmaFloat_ptr dC,       magma_int_t lddc,
    magmaFloat_ptr dwork,    magma_int_t ldwork );

magma_int_t
magma_slarfb_gpu_gemm(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV, magma_int_t lddv,
    magmaFloat_const_ptr dT, magma_int_t lddt,
    magmaFloat_ptr dC,       magma_int_t lddc,
    magmaFloat_ptr dwork,    magma_int_t ldwork,
    magmaFloat_ptr dworkvt,  magma_int_t ldworkvt);

magma_int_t
magma_slarfb2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_const_ptr dV, magma_int_t lddv,
    magmaFloat_const_ptr dT, magma_int_t lddt,
    magmaFloat_ptr dC,       magma_int_t lddc,
    magmaFloat_ptr dwork,    magma_int_t ldwork );

magma_int_t
magma_slatrd_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb, magma_int_t nb0,
    float *A,  magma_int_t lda,
    float *e, float *tau,
    float    *W,       magma_int_t ldw,
    magmaFloat_ptr dA[],    magma_int_t ldda, magma_int_t offset,
    magmaFloat_ptr dW[],    magma_int_t lddw,
    float    *hwork,   magma_int_t lhwork,
    magmaFloat_ptr dwork[], magma_int_t ldwork,
    magma_queue_t queues[] );

magma_int_t
magma_slauum_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_sposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_spotf2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_spotrf_gpu(
    magma_uplo_t uplo,  magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_spotrf_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_spotrf_mgpu_right(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_spotrf3_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaFloat_ptr d_lA[],  magma_int_t ldda,
    magmaFloat_ptr d_lP[],  magma_int_t lddp,
    float *A, magma_int_t lda, magma_int_t h,
    magma_queue_t queues[][3], magma_event_t events[][5],
    magma_int_t *info);

magma_int_t
magma_spotri_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_spotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaFloat_ptr dA, magma_int_t ldda,
    magmaFloat_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_sssssm_gpu(
    magma_order_t order, magma_int_t m1, magma_int_t n1,
    magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib,
    magmaFloat_ptr dA1, magma_int_t ldda1,
    magmaFloat_ptr dA2, magma_int_t ldda2,
    magmaFloat_ptr dL1, magma_int_t lddl1,
    magmaFloat_ptr dL2, magma_int_t lddl2,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_strtri_gpu(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_stsqrt_gpu(
    magma_int_t *m, magma_int_t *n,
    float *A1, float *A2, magma_int_t *lda,
    float *tau,
    float *work, magma_int_t *lwork,
    magmaFloat_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_ststrf_gpu(
    magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
    float    *hU, magma_int_t ldhu,
    magmaFloat_ptr dU, magma_int_t lddu,
    float    *hA, magma_int_t ldha,
    magmaFloat_ptr dA, magma_int_t ldda,
    float    *hL, magma_int_t ldhl,
    magmaFloat_ptr dL, magma_int_t lddl,
    magma_int_t *ipiv,
    float *hwork, magma_int_t ldhwork,
    magmaFloat_ptr dwork, magma_int_t lddwork,
    magma_int_t *info);

magma_int_t
magma_sorgqr_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_sormql2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dC, magma_int_t lddc,
    float *wA, magma_int_t ldwa,
    magma_int_t *info);

magma_int_t
magma_sormqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dC, magma_int_t lddc,
    float *hwork, magma_int_t lwork,
    magmaFloat_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_sormqr2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dC, magma_int_t lddc,
    float    *wA, magma_int_t ldwa,
    magma_int_t *info);

magma_int_t
magma_sormtr_gpu(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *tau,
    magmaFloat_ptr dC, magma_int_t lddc,
    float    *wA, magma_int_t ldwa,
    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA utility function definitions
*/

extern const float MAGMA_S_NAN;
extern const float MAGMA_S_INF;

magma_int_t
magma_snan_inf(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const float *A, magma_int_t lda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf );

magma_int_t
magma_snan_inf_gpu(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf );

void magma_sprint(
    magma_int_t m, magma_int_t n,
    const float *A, magma_int_t lda );

void magma_sprint_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dA, magma_int_t ldda );

void spanel_to_q(
    magma_uplo_t uplo, magma_int_t ib,
    float *A, magma_int_t lda,
    float *work );

void sq_to_panel(
    magma_uplo_t uplo, magma_int_t ib,
    float *A, magma_int_t lda,
    float *work );

#ifdef __cplusplus
}
#endif

#undef REAL

#endif /* MAGMA_S_H */
