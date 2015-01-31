/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_z.h normal z -> d, Fri Jan 30 19:00:06 2015
*/

#ifndef MAGMA_D_H
#define MAGMA_D_H

#include "magma_types.h"
#include "magma_dgehrd_m.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
#ifdef REAL
magma_int_t magma_get_dlaex3_m_nb();       // defined in dlaex3_m.cpp
#endif

magma_int_t magma_get_dpotrf_nb( magma_int_t m );
magma_int_t magma_get_dgetrf_nb( magma_int_t m );
magma_int_t magma_get_dgetri_nb( magma_int_t m );
magma_int_t magma_get_dgeqp3_nb( magma_int_t m );
magma_int_t magma_get_dgeqrf_nb( magma_int_t m );
magma_int_t magma_get_dgeqlf_nb( magma_int_t m );
magma_int_t magma_get_dgehrd_nb( magma_int_t m );
magma_int_t magma_get_dsytrd_nb( magma_int_t m );
magma_int_t magma_get_dsytrf_nb( magma_int_t m );
magma_int_t magma_get_dsytrf_nopiv_nb( magma_int_t m );
magma_int_t magma_get_dgelqf_nb( magma_int_t m );
magma_int_t magma_get_dgebrd_nb( magma_int_t m );
magma_int_t magma_get_dsygst_nb( magma_int_t m );
magma_int_t magma_get_dgesvd_nb( magma_int_t m );
magma_int_t magma_get_dsygst_nb_m( magma_int_t m );
magma_int_t magma_get_dbulge_nb( magma_int_t m, magma_int_t nbthreads );
magma_int_t magma_get_dbulge_nb_mgpu( magma_int_t m );
magma_int_t magma_dbulge_get_Vblksiz( magma_int_t m, magma_int_t nb, magma_int_t nbthreads );
magma_int_t magma_get_dbulge_gcperf();


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU (alphabetical order)
*/

#ifdef REAL
// only applicable to real [sd] precisions
void
magma_dmove_eig(
    magma_range_t range, magma_int_t n, double *w,
    magma_int_t *il, magma_int_t *iu, double vl, double vu, magma_int_t *m);

// defined in dlaex3.cpp
void
magma_dvrange(
    magma_int_t k, double *d, magma_int_t *il, magma_int_t *iu, double vl, double vu);

void
magma_dirange(
    magma_int_t k, magma_int_t *indxq, magma_int_t *iil, magma_int_t *iiu, magma_int_t il, magma_int_t iu);
#endif

magma_int_t
magma_dgebrd(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    double *d, double *e,
    double *tauq, double *taup,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgeev(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    double *A, magma_int_t lda,
    #ifdef COMPLEX
    double *w,
    #else
    double *wr, double *wi,
    #endif
    double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_dgehrd(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magmaDouble_ptr dT,
    magma_int_t *info);

magma_int_t
magma_dgehrd2(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgelqf(
    magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgeqlf(
    magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgeqp3(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *jpvt, double *tau,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_dgeqrf(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgeqrf_ooc(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgeqrf4(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgesdd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    double *s,
    double *U, magma_int_t ldu,
    double *VT, magma_int_t ldvt,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *iwork,
    magma_int_t *info);

magma_int_t
magma_dgesv(
    magma_int_t n, magma_int_t nrhs,
    double *A, magma_int_t lda, magma_int_t *ipiv,
    double *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_dgesv_rbt(
    magma_bool_t ref, magma_int_t n, magma_int_t nrhs,
    double *A, magma_int_t lda, 
    double *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_dgesvd(
    magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda, double *s,
    double *U,    magma_int_t ldu,
    double *VT,   magma_int_t ldvt,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_dgetf2_nopiv(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_dgetrf(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_dgetrf_nopiv(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_dgetrf_piv(
    magma_int_t m, magma_int_t n, magma_int_t NB,
    double *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_dgetrf2(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_dsyevd(
    magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsyevdx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsyevdx_2stage(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t
magma_dsyevr(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double vl, double vu,
    magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
    double *w,
    double *Z, magma_int_t ldz,
    magma_int_t *isuppz,
    double *work, magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

// no real [sd] precisions available
magma_int_t
magma_dsyevx(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double vl, double vu,
    magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
    double *w,
    double *Z, magma_int_t ldz,
    double *work, magma_int_t lwork,
    double *rwork, magma_int_t *iwork,
    magma_int_t *ifail,
    magma_int_t *info);
#endif  // COMPLEX

magma_int_t
magma_dsygst(
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_dsygvd(
    magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    double *w, double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsygvdx(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n, double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsygvdx_2stage(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsygvr(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    double abstol, magma_int_t *m, double *w,
    double *Z, magma_int_t ldz,
    magma_int_t *isuppz, double *work, magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsygvx(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n, double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    double abstol, magma_int_t *m, double *w,
    double *Z, magma_int_t ldz,
    double *work, magma_int_t lwork, double *rwork,
    magma_int_t *iwork, magma_int_t *ifail,
    magma_int_t *info);

magma_int_t
magma_dsysv(magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
            double *A, magma_int_t lda, magma_int_t *ipiv,
            double *B, magma_int_t ldb,
            magma_int_t *info );

magma_int_t
magma_dsytrd(
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double *d, double *e, double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dsytrf(
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_dsytrf_nopiv(
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_dsytrd_sb2st(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
    double *A, magma_int_t lda,
    double *d, double *e,
    double *V, magma_int_t ldv,
    double *TAU, magma_int_t compT,
    double *T, magma_int_t ldt);

magma_int_t
magma_dsytrd_sy2sb(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magmaDouble_ptr dT,
    magma_int_t *info);

#ifdef REAL
// only applicable to real [sd] precisions
magma_int_t
magma_dlaex0(
    magma_int_t n, double *d, double *e,
    double *Q, magma_int_t ldq,
    double *work, magma_int_t *iwork,
    magmaDouble_ptr dwork,
    magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);

magma_int_t
magma_dlaex1(
    magma_int_t n, double *d,
    double *Q, magma_int_t ldq,
    magma_int_t *indxq, double rho, magma_int_t cutpnt,
    double *work, magma_int_t *iwork,
    magmaDouble_ptr dwork,
    magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);

magma_int_t
magma_dlaex3(
    magma_int_t k, magma_int_t n, magma_int_t n1, double *d,
    double *Q, magma_int_t ldq,
    double rho,
    double *dlamda, double *Q2, magma_int_t *indx,
    magma_int_t *ctot, double *w, double *s, magma_int_t *indxq,
    magmaDouble_ptr dwork,
    magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);
#endif  // REAL

magma_int_t
magma_dlasyf_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t *kb,
    double    *hA, magma_int_t lda,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaDouble_ptr dW, magma_int_t lddw,
    magma_queue_t queues[], magma_event_t event[],
    magma_int_t *info);

magma_int_t
dsytrf_nopiv_cpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t ib,
    double *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_dsytrs_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_dsysv_nopiv_gpu(
    magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs, 
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb, 
    magma_int_t *info);

magma_int_t
magma_dlahr2(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dV, magma_int_t lddv,
    double *A,  magma_int_t lda,
    double *tau,
    double *T,  magma_int_t ldt,
    double *Y,  magma_int_t ldy);

magma_int_t
magma_dlahru(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    double     *A, magma_int_t lda,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dY, magma_int_t lddy,
    magmaDouble_ptr dV, magma_int_t lddv,
    magmaDouble_ptr dT,
    magmaDouble_ptr dwork);

#ifdef REAL
magma_int_t
magma_dlaln2(
    magma_int_t trans, magma_int_t na, magma_int_t nw,
    double smin, double ca, const double *A, magma_int_t lda,
    double d1, double d2,   const double *B, magma_int_t ldb,
    double wr, double wi, double *X, magma_int_t ldx,
    double *scale, double *xnorm,
    magma_int_t *info);
#endif

magma_int_t
magma_dlaqps(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    double *A,  magma_int_t lda,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, double *tau, double *vn1, double *vn2,
    double *auxv,
    double *F,  magma_int_t ldf,
    magmaDouble_ptr dF, magma_int_t lddf );

#ifdef REAL
magma_int_t
magma_dlaqtrsd(
    magma_trans_t trans, magma_int_t n,
    const double *T, magma_int_t ldt,
    double *x,       magma_int_t ldx,
    const double *cnorm,
    magma_int_t *info);
#endif

magma_int_t
magma_dlatrd(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    double *A, magma_int_t lda,
    double *e, double *tau,
    double *W, magma_int_t ldw,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dW, magma_int_t lddw);

magma_int_t
magma_dlatrd2(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    double *A,  magma_int_t lda,
    double *e, double *tau,
    double *W,  magma_int_t ldw,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dW, magma_int_t lddw,
    magmaDouble_ptr dwork, magma_int_t ldwork);

#ifdef COMPLEX
magma_int_t
magma_dlatrsd(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_diag_t diag, magma_bool_t normin,
    magma_int_t n, const double *A, magma_int_t lda,
    double lambda,
    double *x,
    double *scale, double *cnorm,
    magma_int_t *info);
#endif

magma_int_t
magma_dlauum(
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_dposv(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_dpotrf(
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_dpotri(
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_dstedx(
    magma_range_t range, magma_int_t n, double vl, double vu,
    magma_int_t il, magma_int_t iu, double *d, double *e,
    double *Z, magma_int_t ldz,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magmaDouble_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_dtrevc3(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select, magma_int_t n,
    double *T,  magma_int_t ldt,
    double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_dtrevc3_mt(
    magma_side_t side, magma_vec_t howmany,
    magma_int_t *select, magma_int_t n,
    double *T,  magma_int_t ldt,
    double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr,
    magma_int_t mm, magma_int_t *mout,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_dtrtri(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_dorghr(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    double *A, magma_int_t lda,
    double *tau,
    magmaDouble_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_dorgqr(
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    magmaDouble_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_dorgqr2(
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    magma_int_t *info);

magma_int_t
magma_dormbr(
    magma_vect_t vect, magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    double *C, magma_int_t ldc,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dormlq(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    double *C, magma_int_t ldc,
    double *work, magma_int_t lwork,
    magma_int_t *info);

// not yet implemented
//magma_int_t magma_dunmrq( magma_side_t side, magma_trans_t trans,
//                          magma_int_t m, magma_int_t n, magma_int_t k,
//                          double *A, magma_int_t lda,
//                          double *tau,
//                          double *C, magma_int_t ldc,
//                          double *work, magma_int_t lwork,
//                          magma_int_t *info);

magma_int_t
magma_dormql(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    double *C, magma_int_t ldc,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dormqr(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    double *C, magma_int_t ldc,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dormtr(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda,
    double *tau,
    double *C,    magma_int_t ldc,
    double *work, magma_int_t lwork,
    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU / Multi-GPU (alphabetical order)
*/
magma_int_t
magma_dgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    double *A, magma_int_t lda,
    #ifdef COMPLEX
    double *w,
    #else
    double *wr, double *wi,
    #endif
    double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_dgehrd_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    double *T,
    magma_int_t *info);

magma_int_t
magma_dgetrf_m(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_dsyevd_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsyevdx_2stage_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsyevdx_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsygst_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    magma_int_t *info);

magma_int_t
magma_dsygvd_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsygvdx_2stage_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsygvdx_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef REAL
magma_int_t
magma_dlaex0_m(
    magma_int_t ngpu,
    magma_int_t n, double *d, double *e,
    double *Q, magma_int_t ldq,
    double *work, magma_int_t *iwork,
    magma_range_t range, double vl, double vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *info);

magma_int_t
magma_dlaex1_m(
    magma_int_t ngpu,
    magma_int_t n, double *d,
    double *Q, magma_int_t ldq,
    magma_int_t *indxq, double rho, magma_int_t cutpnt,
    double *work, magma_int_t *iwork,
    magmaDouble_ptr dwork[],
    magma_queue_t queues[MagmaMaxGPUs][2],
    magma_range_t range, double vl, double vu,
    magma_int_t il, magma_int_t iu, magma_int_t *info);

magma_int_t
magma_dlaex3_m(
    magma_int_t ngpu,
    magma_int_t k, magma_int_t n, magma_int_t n1, double *d,
    double *Q, magma_int_t ldq, double rho,
    double *dlamda, double *Q2, magma_int_t *indx,
    magma_int_t *ctot, double *w, double *s, magma_int_t *indxq,
    magmaDouble_ptr dwork[],
    magma_queue_t queues[MagmaMaxGPUs][2],
    magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info);
#endif

magma_int_t
magma_dlahr2_m(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    double *A, magma_int_t lda,
    double *tau,
    double *T, magma_int_t ldt,
    double *Y, magma_int_t ldy,
    struct dgehrd_data *data );

magma_int_t
magma_dlahru_m(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    double *A, magma_int_t lda,
    struct dgehrd_data *data );

magma_int_t
magma_dpotrf_m(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info);

magma_int_t
magma_dstedx_m(
    magma_int_t ngpu,
    magma_range_t range, magma_int_t n, double vl, double vu,
    magma_int_t il, magma_int_t iu, double *d, double *e,
    double *Z, magma_int_t ldz,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dtrsm_m(
    magma_int_t ngpu,
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transa, magma_diag_t diag,
    magma_int_t m, magma_int_t n, double alpha,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb);

magma_int_t
magma_dorghr_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    double *A, magma_int_t lda,
    double *tau,
    double *T, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_dorgqr_m(
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    double *T, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_dormqr_m(
    magma_int_t ngpu,
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A,    magma_int_t lda,
    double *tau,
    double *C,    magma_int_t ldc,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dormtr_m(
    magma_int_t ngpu,
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda,
    double *tau,
    double *C,    magma_int_t ldc,
    double *work, magma_int_t lwork,
    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on GPU (alphabetical order)
*/
magma_int_t
magma_dgegqr_gpu(
    magma_int_t ikind, magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, double *work,
    magma_int_t *info);

magma_int_t
magma_dgelqf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgels_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    double *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgels3_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    double *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgeqp3_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *jpvt, double *tau,
    magmaDouble_ptr dwork, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info);

magma_int_t
magma_dgeqr2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dwork,
    magma_int_t *info);

magma_int_t
magma_dgeqr2x_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dtau,
    magmaDouble_ptr dT, magmaDouble_ptr ddA,
    magmaDouble_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_dgeqr2x2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dtau,
    magmaDouble_ptr dT, magmaDouble_ptr ddA,
    magmaDouble_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_dgeqr2x3_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dtau,
    magmaDouble_ptr dT,
    magmaDouble_ptr ddA,
    magmaDouble_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_dgeqr2x4_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dtau,
    magmaDouble_ptr dT, magmaDouble_ptr ddA,
    magmaDouble_ptr dwork,
    magma_queue_t queue,
    magma_int_t *info);

magma_int_t
magma_dgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT,
    magma_int_t *info);

magma_int_t
magma_dgeqrf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magma_int_t *info);

magma_int_t
magma_dgeqrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dlA[], magma_int_t ldda,
    double *tau,
    magma_int_t *info);

magma_int_t
magma_dgeqrf3_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT,
    magma_int_t *info);

magma_int_t
magma_dgeqrs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT,
    magmaDouble_ptr dB, magma_int_t lddb,
    double *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgeqrs3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT,
    magmaDouble_ptr dB, magma_int_t lddb,
    double *hwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgerbt_gpu(
    magma_bool_t gen, magma_int_t n, magma_int_t nrhs, 
    magmaDouble_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dB, magma_int_t lddb, 
    double *U, double *V,
    magma_int_t *info);

magma_int_t
magma_dgessm_gpu(
    magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib,
    magma_int_t *ipiv,
    magmaDouble_ptr dL1, magma_int_t lddl1,
    magmaDouble_ptr dL,  magma_int_t lddl,
    magmaDouble_ptr dA,  magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dgesv_gpu(
    magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_dgesv_nopiv_gpu( 
    magma_int_t n, magma_int_t nrhs, 
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb, 
                 magma_int_t *info);

magma_int_t
magma_dgetf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_dgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_dgetrf_incpiv_gpu(
    magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t ib,
    double    *hA, magma_int_t ldha,
    magmaDouble_ptr dA, magma_int_t ldda,
    double    *hL, magma_int_t ldhl,
    magmaDouble_ptr dL, magma_int_t lddl,
    magma_int_t *ipiv,
    magmaDouble_ptr dwork, magma_int_t lddwork,
    magma_int_t *info);

magma_int_t
magma_dgetrf_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr d_lA[], magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_dgetrf_nopiv_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dgetrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
    magmaDouble_ptr d_lAT[], magma_int_t lddat, magma_int_t *ipiv,
    magmaDouble_ptr d_lAP[],
    double *W, magma_int_t ldw,
    magma_queue_t queues[][2],
    magma_int_t *info);

magma_int_t
magma_dgetri_gpu(
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaDouble_ptr dwork, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dgetrs_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda, magma_int_t *ipiv,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_dgetrs_nopiv_gpu(
    magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_dsyevd_gpu(
    magma_vec_t jobz, magma_uplo_t uplo,
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *w,
    double *wA,  magma_int_t ldwa,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsyevdx_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double vl, double vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *wA,  magma_int_t ldwa,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t
magma_dsyevr_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double vl, double vu,
    magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
    double *w,
    magmaDouble_ptr dZ, magma_int_t lddz,
    magma_int_t *isuppz,
    double *wA, magma_int_t ldwa,
    double *wZ, magma_int_t ldwz,
    double *work, magma_int_t lwork,
    double *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info);

magma_int_t
magma_dsyevx_gpu(
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    double abstol, magma_int_t *m,
    double *w,
    magmaDouble_ptr dZ, magma_int_t lddz,
    double *wA, magma_int_t ldwa,
    double *wZ, magma_int_t ldwz,
    double *work, magma_int_t lwork,
    double *rwork, magma_int_t *iwork,
    magma_int_t *ifail,
    magma_int_t *info);
#endif  // COMPLEX

magma_int_t
magma_dsygst_gpu(
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_dsytrd_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *d, double *e, double *tau,
    double *wA,  magma_int_t ldwa,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dsytrd_sy2sb_mgpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magmaDouble_ptr dAmgpu[], magma_int_t ldda,
    magmaDouble_ptr dTmgpu[], magma_int_t lddt,
    magma_int_t ngpu, magma_int_t distblk,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_int_t *info);

magma_int_t
magma_dsytrd_sy2sb_mgpu_spec(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    magmaDouble_ptr dAmgpu[], magma_int_t ldda,
    magmaDouble_ptr dTmgpu[], magma_int_t lddt,
    magma_int_t ngpu, magma_int_t distblk,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_int_t *info);

magma_int_t
magma_dsytrd_mgpu(
    magma_int_t ngpu, magma_int_t nqueue,
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double *d, double *e, double *tau,
    double *work, magma_int_t lwork,
    magma_int_t *info);

magma_int_t
magma_dsytrd2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *d, double *e, double *tau,
    double *wA,  magma_int_t ldwa,
    double *work, magma_int_t lwork,
    magmaDouble_ptr dwork, magma_int_t ldwork,
    magma_int_t *info);

magma_int_t
magma_dsytrf_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dlabrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    double     *A, magma_int_t lda,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *d, double *e, double *tauq, double *taup,
    double     *X, magma_int_t ldx,
    magmaDouble_ptr dX, magma_int_t lddx,
    double     *Y, magma_int_t ldy,
    magmaDouble_ptr dY, magma_int_t lddy );

magma_int_t
magma_dlaqps_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDouble_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt, double *tau,
    double *vn1, double *vn2,
    magmaDouble_ptr dauxv,
    magmaDouble_ptr dF, magma_int_t lddf);

magma_int_t
magma_dlaqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDouble_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaDouble_ptr dtau,
    magmaDouble_ptr dvn1, magmaDouble_ptr dvn2,
    magmaDouble_ptr dauxv,
    magmaDouble_ptr dF, magma_int_t lddf);

magma_int_t
magma_dlaqps3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDouble_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaDouble_ptr dtau,
    magmaDouble_ptr dvn1, magmaDouble_ptr dvn2,
    magmaDouble_ptr dauxv,
    magmaDouble_ptr dF, magma_int_t lddf);

magma_int_t
magma_dlarf_gpu(
    magma_int_t m,  magma_int_t n,
    magmaDouble_const_ptr dv, magmaDouble_const_ptr dtau,
    magmaDouble_ptr dC,  magma_int_t lddc);

magma_int_t
magma_dlarfb_gpu(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_const_ptr dV, magma_int_t lddv,
    magmaDouble_const_ptr dT, magma_int_t lddt,
    magmaDouble_ptr dC,       magma_int_t lddc,
    magmaDouble_ptr dwork,    magma_int_t ldwork );

magma_int_t
magma_dlarfb_gpu_gemm(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_const_ptr dV, magma_int_t lddv,
    magmaDouble_const_ptr dT, magma_int_t lddt,
    magmaDouble_ptr dC,       magma_int_t lddc,
    magmaDouble_ptr dwork,    magma_int_t ldwork,
    magmaDouble_ptr dworkvt,  magma_int_t ldworkvt);

magma_int_t
magma_dlarfb2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_const_ptr dV, magma_int_t lddv,
    magmaDouble_const_ptr dT, magma_int_t lddt,
    magmaDouble_ptr dC,       magma_int_t lddc,
    magmaDouble_ptr dwork,    magma_int_t ldwork );

magma_int_t
magma_dlatrd_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb, magma_int_t nb0,
    double *A,  magma_int_t lda,
    double *e, double *tau,
    double    *W,       magma_int_t ldw,
    magmaDouble_ptr dA[],    magma_int_t ldda, magma_int_t offset,
    magmaDouble_ptr dW[],    magma_int_t lddw,
    double    *hwork,   magma_int_t lhwork,
    magmaDouble_ptr dwork[], magma_int_t ldwork,
    magma_queue_t queues[] );

magma_int_t
magma_dlauum_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dposv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_dpotf2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dpotrf_gpu(
    magma_uplo_t uplo,  magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dpotrf_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dpotrf_mgpu_right(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dpotrf3_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaDouble_ptr d_lA[],  magma_int_t ldda,
    magmaDouble_ptr d_lP[],  magma_int_t lddp,
    double *A, magma_int_t lda, magma_int_t h,
    magma_queue_t queues[][3], magma_event_t events[][5],
    magma_int_t *info);

magma_int_t
magma_dpotri_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dpotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t *info);

magma_int_t
magma_dssssm_gpu(
    magma_order_t order, magma_int_t m1, magma_int_t n1,
    magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib,
    magmaDouble_ptr dA1, magma_int_t ldda1,
    magmaDouble_ptr dA2, magma_int_t ldda2,
    magmaDouble_ptr dL1, magma_int_t lddl1,
    magmaDouble_ptr dL2, magma_int_t lddl2,
    magma_int_t *ipiv,
    magma_int_t *info);

magma_int_t
magma_dtrtri_gpu(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info);

magma_int_t
magma_dtsqrt_gpu(
    magma_int_t *m, magma_int_t *n,
    double *A1, double *A2, magma_int_t *lda,
    double *tau,
    double *work, magma_int_t *lwork,
    magmaDouble_ptr dwork,
    magma_int_t *info);

magma_int_t
magma_dtstrf_gpu(
    magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
    double    *hU, magma_int_t ldhu,
    magmaDouble_ptr dU, magma_int_t lddu,
    double    *hA, magma_int_t ldha,
    magmaDouble_ptr dA, magma_int_t ldda,
    double    *hL, magma_int_t ldhl,
    magmaDouble_ptr dL, magma_int_t lddl,
    magma_int_t *ipiv,
    double *hwork, magma_int_t ldhwork,
    magmaDouble_ptr dwork, magma_int_t lddwork,
    magma_int_t *info);

magma_int_t
magma_dorgqr_gpu(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_dormql2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dC, magma_int_t lddc,
    double *wA, magma_int_t ldwa,
    magma_int_t *info);

magma_int_t
magma_dormqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dC, magma_int_t lddc,
    double *hwork, magma_int_t lwork,
    magmaDouble_ptr dT, magma_int_t nb,
    magma_int_t *info);

magma_int_t
magma_dormqr2_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dC, magma_int_t lddc,
    double    *wA, magma_int_t ldwa,
    magma_int_t *info);

magma_int_t
magma_dormtr_gpu(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dC, magma_int_t lddc,
    double    *wA, magma_int_t ldwa,
    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA utility function definitions
*/

extern const double MAGMA_D_NAN;
extern const double MAGMA_D_INF;

magma_int_t
magma_dnan_inf(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const double *A, magma_int_t lda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf );

magma_int_t
magma_dnan_inf_gpu(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf );

void magma_dprint(
    magma_int_t m, magma_int_t n,
    const double *A, magma_int_t lda );

void magma_dprint_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda );

void dpanel_to_q(
    magma_uplo_t uplo, magma_int_t ib,
    double *A, magma_int_t lda,
    double *work );

void dq_to_panel(
    magma_uplo_t uplo, magma_int_t ib,
    double *A, magma_int_t lda,
    double *work );

#ifdef __cplusplus
}
#endif

#undef REAL

#endif /* MAGMA_D_H */
