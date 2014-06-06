/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:17 2013
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
magma_int_t magma_get_dpotrf_nb( magma_int_t m );
magma_int_t magma_get_dgetrf_nb( magma_int_t m );
magma_int_t magma_get_dgetri_nb( magma_int_t m );
magma_int_t magma_get_dgeqp3_nb( magma_int_t m );
magma_int_t magma_get_dgeqrf_nb( magma_int_t m );
magma_int_t magma_get_dgeqlf_nb( magma_int_t m );
magma_int_t magma_get_dgehrd_nb( magma_int_t m );
magma_int_t magma_get_dsytrd_nb( magma_int_t m );
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
   -- MAGMA function definitions / Data on CPU
*/

#ifdef REAL
// only applicable to real [sd] precisions
void magma_dmove_eig(magma_range_t range, magma_int_t n, double *w, magma_int_t *il,
                          magma_int_t *iu, double vl, double vu, magma_int_t *m);
#endif

magma_int_t magma_dgebrd( magma_int_t m, magma_int_t n, double *A,
                          magma_int_t lda, double *d, double *e,
                          double *tauq,  double *taup,
                          double *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_dgehrd2(magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          double *A, magma_int_t lda, double *tau,
                          double *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_dgehrd( magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          double *A, magma_int_t lda, double *tau,
                          double *work, magma_int_t lwork,
                          double *dT, magma_int_t *info);

magma_int_t magma_dgelqf( magma_int_t m, magma_int_t n,
                          double *A,    magma_int_t lda,   double *tau,
                          double *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_dgeqlf( magma_int_t m, magma_int_t n,
                          double *A,    magma_int_t lda,   double *tau,
                          double *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_dgeqrf( magma_int_t m, magma_int_t n, double *A,
                          magma_int_t lda, double *tau, double *work,
                          magma_int_t lwork, magma_int_t *info);

magma_int_t magma_dgeqrf4(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                          double *a,    magma_int_t lda, double *tau,
                          double *work, magma_int_t lwork, magma_int_t *info );

magma_int_t magma_dgeqrf_ooc( magma_int_t m, magma_int_t n, double *A,
                          magma_int_t lda, double *tau, double *work,
                          magma_int_t lwork, magma_int_t *info);

magma_int_t magma_dgesv ( magma_int_t n, magma_int_t nrhs,
                          double *A, magma_int_t lda, magma_int_t *ipiv,
                          double *B, magma_int_t ldb, magma_int_t *info);

magma_int_t magma_dgetrf( magma_int_t m, magma_int_t n, double *A,
                          magma_int_t lda, magma_int_t *ipiv,
                          magma_int_t *info);

magma_int_t magma_dgetrf2(magma_int_t m, magma_int_t n, double *a,
                          magma_int_t lda, magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_dlaqps( magma_int_t m, magma_int_t n, magma_int_t offset,
                          magma_int_t nb, magma_int_t *kb,
                          double *A,  magma_int_t lda,
                          double *dA, magma_int_t ldda,
                          magma_int_t *jpvt, double *tau, double *vn1, double *vn2,
                          double *auxv,
                          double *F,  magma_int_t ldf,
                          double *dF, magma_int_t lddf );

void        magma_dlarfg( magma_int_t n, double *alpha, double *x,
                          magma_int_t incx, double *tau);

magma_int_t magma_dlatrd( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, double *a,
                          magma_int_t lda, double *e, double *tau,
                          double *w, magma_int_t ldw,
                          double *da, magma_int_t ldda,
                          double *dw, magma_int_t lddw);

magma_int_t magma_dlatrd2(magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                          double *a,  magma_int_t lda,
                          double *e, double *tau,
                          double *w,  magma_int_t ldw,
                          double *da, magma_int_t ldda,
                          double *dw, magma_int_t lddw,
                          double *dwork, magma_int_t ldwork);

magma_int_t magma_dlahr2( magma_int_t m, magma_int_t n, magma_int_t nb,
                          double *da, double *dv, double *a,
                          magma_int_t lda, double *tau, double *t,
                          magma_int_t ldt, double *y, magma_int_t ldy);

magma_int_t magma_dlahru( magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
                          double *a, magma_int_t lda,
                          double *da, double *y,
                          double *v, double *t,
                          double *dwork);

magma_int_t magma_dposv ( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                          double *A, magma_int_t lda,
                          double *B, magma_int_t ldb, magma_int_t *info);

magma_int_t magma_dpotrf( magma_uplo_t uplo, magma_int_t n, double *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_dpotri( magma_uplo_t uplo, magma_int_t n, double *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_dlauum( magma_uplo_t uplo, magma_int_t n, double *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_dtrtri( magma_uplo_t uplo, magma_diag_t diag, magma_int_t n, double *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_dsytrd( magma_uplo_t uplo, magma_int_t n, double *A,
                          magma_int_t lda, double *d, double *e,
                          double *tau, double *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_dorgqr( magma_int_t m, magma_int_t n, magma_int_t k,
                          double *a, magma_int_t lda,
                          double *tau, double *dT,
                          magma_int_t nb, magma_int_t *info );

magma_int_t magma_dorgqr2(magma_int_t m, magma_int_t n, magma_int_t k,
                          double *a, magma_int_t lda,
                          double *tau, magma_int_t *info );

magma_int_t magma_dormql( magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          double *a, magma_int_t lda,
                          double *tau,
                          double *c, magma_int_t ldc,
                          double *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_dormqr( magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          double *a, magma_int_t lda, double *tau,
                          double *c, magma_int_t ldc,
                          double *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_dormtr( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                          magma_int_t m, magma_int_t n,
                          double *a,    magma_int_t lda,
                          double *tau,
                          double *c,    magma_int_t ldc,
                          double *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_dorghr( magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          double *a, magma_int_t lda,
                          double *tau,
                          double *dT, magma_int_t nb,
                          magma_int_t *info);

magma_int_t  magma_dgeev( magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
                          double *a, magma_int_t lda,
                          #ifdef COMPLEX
                          double *w,
                          #else
                          double *wr, double *wi,
                          #endif
                          double *vl, magma_int_t ldvl,
                          double *vr, magma_int_t ldvr,
                          double *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *info);

magma_int_t magma_dgeqp3( magma_int_t m, magma_int_t n,
                          double *a, magma_int_t lda,
                          magma_int_t *jpvt, double *tau,
                          double *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *info);

magma_int_t magma_dgesvd( magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
                          double *a,    magma_int_t lda, double *s,
                          double *u,    magma_int_t ldu,
                          double *vt,   magma_int_t ldvt,
                          double *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *info );

magma_int_t magma_dsyevd( magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
                          double *a, magma_int_t lda, double *w,
                          double *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_dsyevdx(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          double *a, magma_int_t lda,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, double *w,
                          double *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_dsyevdx_2stage(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n,
                          double *a, magma_int_t lda,
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
magma_int_t magma_dsyevx( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          double *a, magma_int_t lda, double vl, double vu,
                          magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
                          double *w, double *z, magma_int_t ldz,
                          double *work, magma_int_t lwork,
                          double *rwork, magma_int_t *iwork,
                          magma_int_t *ifail, magma_int_t *info);

// no real [sd] precisions available
magma_int_t magma_dsyevr( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          double *a, magma_int_t lda, double vl, double vu,
                          magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
                          double *w, double *z, magma_int_t ldz,
                          magma_int_t *isuppz,
                          double *work, magma_int_t lwork,
                          double *rwork, magma_int_t lrwork, magma_int_t *iwork,
                          magma_int_t liwork, magma_int_t *info);
#endif  // COMPLEX

magma_int_t magma_dsygvd( magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
                          double *a, magma_int_t lda,
                          double *b, magma_int_t ldb,
                          double *w, double *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
                          
magma_int_t magma_dsygvdx(magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, double *a, magma_int_t lda,
                          double *b, magma_int_t ldb,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, double *w,
                          double *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_dsygvdx_2stage(magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          double *a, magma_int_t lda,
                          double *b, magma_int_t ldb,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, double *w,
                          double *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork,
                          magma_int_t *info);

magma_int_t magma_dsygvx( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, double *a, magma_int_t lda,
                          double *b, magma_int_t ldb,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          double abstol, magma_int_t *m, double *w,
                          double *z, magma_int_t ldz,
                          double *work, magma_int_t lwork, double *rwork,
                          magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info);

magma_int_t magma_dsygvr( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, double *a, magma_int_t lda,
                          double *b, magma_int_t ldb,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          double abstol, magma_int_t *m, double *w,
                          double *z, magma_int_t ldz,
                          magma_int_t *isuppz, double *work, magma_int_t lwork,
                          double *rwork, magma_int_t lrwork, magma_int_t *iwork,
                          magma_int_t liwork, magma_int_t *info);

magma_int_t magma_dstedx( magma_range_t range, magma_int_t n, double vl, double vu,
                          magma_int_t il, magma_int_t iu, double *D, double *E,
                          double *Z, magma_int_t ldz,
                          double *rwork, magma_int_t lrwork,
                          magma_int_t *iwork, magma_int_t liwork,
                          double *dwork, magma_int_t *info);

#ifdef REAL
// only applicable to real [sd] precisions
magma_int_t magma_dlaex0( magma_int_t n, double *d, double *e, double *q, magma_int_t ldq,
                          double *work, magma_int_t *iwork, double *dwork,
                          magma_range_t range, double vl, double vu,
                          magma_int_t il, magma_int_t iu, magma_int_t *info);

magma_int_t magma_dlaex1( magma_int_t n, double *d, double *q, magma_int_t ldq,
                          magma_int_t *indxq, double rho, magma_int_t cutpnt,
                          double *work, magma_int_t *iwork, double *dwork,
                          magma_range_t range, double vl, double vu,
                          magma_int_t il, magma_int_t iu, magma_int_t *info);

magma_int_t magma_dlaex3( magma_int_t k, magma_int_t n, magma_int_t n1, double *d,
                          double *q, magma_int_t ldq, double rho,
                          double *dlamda, double *q2, magma_int_t *indx,
                          magma_int_t *ctot, double *w, double *s, magma_int_t *indxq,
                          double *dwork,
                          magma_range_t range, double vl, double vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *info );
#endif  // REAL

magma_int_t magma_dsygst( magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                          double *a, magma_int_t lda,
                          double *b, magma_int_t ldb, magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on CPU / Multi-GPU
*/
magma_int_t magma_dlahr2_m(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    double *A, magma_int_t lda,
    double *tau,
    double *T, magma_int_t ldt,
    double *Y, magma_int_t ldy,
    struct dgehrd_data *data );

magma_int_t magma_dlahru_m(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    double *A, magma_int_t lda,
    struct dgehrd_data *data );

magma_int_t magma_dgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    double *A, magma_int_t lda,
    #ifdef COMPLEX
    double *w,
    #else
    double *wr, double *wi,
    #endif
    double *vl, magma_int_t ldvl,
    double *vr, magma_int_t ldvr,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info );

magma_int_t magma_dgehrd_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    double *A, magma_int_t lda,
    double *tau,
    double *work, magma_int_t lwork,
    double *T,
    magma_int_t *info );

magma_int_t magma_dorghr_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    double *A, magma_int_t lda,
    double *tau,
    double *T, magma_int_t nb,
    magma_int_t *info );

magma_int_t magma_dorgqr_m(
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    double *T, magma_int_t nb,
    magma_int_t *info );

magma_int_t magma_dpotrf_m( magma_int_t num_gpus,
                            magma_uplo_t uplo, magma_int_t n,
                            double *A, magma_int_t lda,
                            magma_int_t *info);

magma_int_t magma_dstedx_m( magma_int_t nrgpu,
                            magma_range_t range, magma_int_t n, double vl, double vu,
                            magma_int_t il, magma_int_t iu, double *D, double *E,
                            double *Z, magma_int_t ldz,
                            double *rwork, magma_int_t ldrwork, magma_int_t *iwork,
                            magma_int_t liwork, magma_int_t *info);

magma_int_t magma_dtrsm_m ( magma_int_t nrgpu,
                            magma_side_t side, magma_uplo_t uplo, magma_trans_t transa, magma_diag_t diag,
                            magma_int_t m, magma_int_t n, double alpha,
                            double *a, magma_int_t lda,
                            double *b, magma_int_t ldb);

magma_int_t magma_dormqr_m( magma_int_t nrgpu, magma_side_t side, magma_trans_t trans,
                            magma_int_t m, magma_int_t n, magma_int_t k,
                            double *a,    magma_int_t lda,
                            double *tau,
                            double *c,    magma_int_t ldc,
                            double *work, magma_int_t lwork,
                            magma_int_t *info);

magma_int_t magma_dormtr_m( magma_int_t nrgpu,
                            magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                            magma_int_t m, magma_int_t n,
                            double *a,    magma_int_t lda,
                            double *tau,
                            double *c,    magma_int_t ldc,
                            double *work, magma_int_t lwork,
                            magma_int_t *info);

magma_int_t magma_dsygst_m( magma_int_t nrgpu,
                            magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                            double *a, magma_int_t lda,
                            double *b, magma_int_t ldb,
                            magma_int_t *info);

magma_int_t magma_dsyevd_m( magma_int_t nrgpu,
                            magma_vec_t jobz, magma_uplo_t uplo,
                            magma_int_t n,
                            double *a, magma_int_t lda,
                            double *w,
                            double *work, magma_int_t lwork,
                            #ifdef COMPLEX
                            double *rwork, magma_int_t lrwork,
                            #endif
                            magma_int_t *iwork, magma_int_t liwork,
                            magma_int_t *info);

magma_int_t magma_dsygvd_m( magma_int_t nrgpu,
                            magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo,
                            magma_int_t n,
                            double *a, magma_int_t lda,
                            double *b, magma_int_t ldb,
                            double *w,
                            double *work, magma_int_t lwork,
                            #ifdef COMPLEX
                            double *rwork, magma_int_t lrwork,
                            #endif
                            magma_int_t *iwork, magma_int_t liwork,
                            magma_int_t *info);

magma_int_t magma_dsyevdx_m( magma_int_t nrgpu,
                             magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                             magma_int_t n,
                             double *a, magma_int_t lda,
                             double vl, double vu, magma_int_t il, magma_int_t iu,
                             magma_int_t *m, double *w,
                             double *work, magma_int_t lwork,
                             #ifdef COMPLEX
                             double *rwork, magma_int_t lrwork,
                             #endif
                             magma_int_t *iwork, magma_int_t liwork,
                             magma_int_t *info);

magma_int_t magma_dsygvdx_m( magma_int_t nrgpu,
                             magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                             magma_int_t n,
                             double *a, magma_int_t lda,
                             double *b, magma_int_t ldb,
                             double vl, double vu, magma_int_t il, magma_int_t iu,
                             magma_int_t *m, double *w,
                             double *work, magma_int_t lwork,
                             #ifdef COMPLEX
                             double *rwork, magma_int_t lrwork,
                             #endif
                             magma_int_t *iwork, magma_int_t liwork,
                             magma_int_t *info);

magma_int_t magma_dsyevdx_2stage_m( magma_int_t nrgpu,
                                    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                                    magma_int_t n,
                                    double *a, magma_int_t lda,
                                    double vl, double vu, magma_int_t il, magma_int_t iu,
                                    magma_int_t *m, double *w,
                                    double *work, magma_int_t lwork,
                                    #ifdef COMPLEX
                                    double *rwork, magma_int_t lrwork,
                                    #endif
                                    magma_int_t *iwork, magma_int_t liwork,
                                    magma_int_t *info);

magma_int_t magma_dsygvdx_2stage_m( magma_int_t nrgpu,
                                    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                                    magma_int_t n,
                                    double *a, magma_int_t lda,
                                    double *b, magma_int_t ldb,
                                    double vl, double vu, magma_int_t il, magma_int_t iu,
                                    magma_int_t *m, double *w,
                                    double *work, magma_int_t lwork,
                                    #ifdef COMPLEX
                                    double *rwork, magma_int_t lrwork,
                                    #endif
                                    magma_int_t *iwork, magma_int_t liwork,
                                    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on GPU
*/
magma_int_t magma_dgels_gpu(  magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              double *dA,    magma_int_t ldda,
                              double *dB,    magma_int_t lddb,
                              double *hwork, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_dgels3_gpu( magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              double *dA,    magma_int_t ldda,
                              double *dB,    magma_int_t lddb,
                              double *hwork, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_dgelqf_gpu( magma_int_t m, magma_int_t n,
                              double *dA,    magma_int_t ldda,   double *tau,
                              double *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_dgeqr2x_gpu(
    magma_int_t *m, magma_int_t *n, double *dA,
    magma_int_t *ldda, double *dtau,
    double *dT, double *ddA,
    double *dwork, magma_int_t *info);

magma_int_t magma_dgeqr2x2_gpu(
    magma_int_t *m, magma_int_t *n, double *dA,
    magma_int_t *ldda, double *dtau,
    double *dT, double *ddA,
    double *dwork, magma_int_t *info);

magma_int_t magma_dgeqr2x3_gpu(
    magma_int_t *m, magma_int_t *n, double *dA,
    magma_int_t *ldda, double *dtau,
    double *dT, double *ddA,
    double *dwork, magma_int_t *info);

magma_int_t magma_dgeqr2x4_gpu(
    magma_int_t *m, magma_int_t *n, double *dA,
    magma_int_t *ldda, double *dtau,
    double *dT, double *ddA,
    double *dwork, magma_int_t *info, magma_queue_t stream);

magma_int_t magma_dgeqrf_gpu( magma_int_t m, magma_int_t n,
                              double *dA,  magma_int_t ldda,
                              double *tau, double *dT,
                              magma_int_t *info);

magma_int_t magma_dgeqrf2_gpu(magma_int_t m, magma_int_t n,
                              double *dA,  magma_int_t ldda,
                              double *tau, magma_int_t *info);

magma_int_t magma_dgeqrf2_mgpu(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                               double **dlA, magma_int_t ldda,
                               double *tau, magma_int_t *info );

magma_int_t magma_dgeqrf3_gpu(magma_int_t m, magma_int_t n,
                              double *dA,  magma_int_t ldda,
                              double *tau, double *dT,
                              magma_int_t *info);

magma_int_t magma_dgeqr2_gpu( magma_int_t m, magma_int_t n,
                              double *dA,  magma_int_t lda,
                              double *tau, double *work,
                              magma_int_t *info);

magma_int_t magma_dgeqrs_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              double *dA,     magma_int_t ldda,
                              double *tau,   double *dT,
                              double *dB,    magma_int_t lddb,
                              double *hwork, magma_int_t lhwork,
                              magma_int_t *info);

magma_int_t magma_dgeqrs3_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              double *dA,     magma_int_t ldda,
                              double *tau,   double *dT,
                              double *dB,    magma_int_t lddb,
                              double *hwork, magma_int_t lhwork,
                              magma_int_t *info);

magma_int_t magma_dgessm_gpu( magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib,
                              magma_int_t *ipiv,
                              double *dL1, magma_int_t lddl1,
                              double *dL,  magma_int_t lddl,
                              double *dA,  magma_int_t ldda,
                              magma_int_t *info);

magma_int_t magma_dgesv_gpu(  magma_int_t n, magma_int_t nrhs,
                              double *dA, magma_int_t ldda, magma_int_t *ipiv,
                              double *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_dgetf2_gpu( magma_int_t m, magma_int_t n,
                              double *dA, magma_int_t lda, magma_int_t *ipiv,
                              magma_int_t* info );

magma_int_t magma_dgetrf_incpiv_gpu( magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t ib,
                              double *hA, magma_int_t ldha, double *dA, magma_int_t ldda,
                              double *hL, magma_int_t ldhl, double *dL, magma_int_t lddl,
                              magma_int_t *ipiv,
                              double *dwork, magma_int_t lddwork,
                              magma_int_t *info);

magma_int_t magma_dgetrf_gpu( magma_int_t m, magma_int_t n,
                              double *dA, magma_int_t ldda,
                              magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_dgetrf_mgpu(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                              double **d_lA, magma_int_t ldda,
                              magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_dgetrf_m(magma_int_t num_gpus0, magma_int_t m, magma_int_t n, double *a, magma_int_t lda,
                           magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_dgetrf_piv(magma_int_t m, magma_int_t n, magma_int_t NB,
                             double *a, magma_int_t lda, magma_int_t *ipiv,
                             magma_int_t *info);

magma_int_t magma_dgetrf2_mgpu(magma_int_t num_gpus,
                               magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
                               double *d_lAT[], magma_int_t lddat, magma_int_t *ipiv,
                               double *d_lAP[], double *a, magma_int_t lda,
                               magma_queue_t streaml[][2], magma_int_t *info);

magma_int_t
      magma_dgetrf_nopiv_gpu( magma_int_t m, magma_int_t n,
                              double *dA, magma_int_t ldda,
                              magma_int_t *info);

magma_int_t magma_dgetri_gpu( magma_int_t n,
                              double *dA, magma_int_t ldda, magma_int_t *ipiv,
                              double *dwork, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_dgetrs_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                              double *dA, magma_int_t ldda, magma_int_t *ipiv,
                              double *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_dlabrd_gpu( magma_int_t m, magma_int_t n, magma_int_t nb,
                              double *a, magma_int_t lda, double *da, magma_int_t ldda,
                              double *d, double *e, double *tauq, double *taup,
                              double *x, magma_int_t ldx, double *dx, magma_int_t lddx,
                              double *y, magma_int_t ldy, double *dy, magma_int_t lddy);

magma_int_t magma_dlaqps_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    double *A,  magma_int_t lda,
    magma_int_t *jpvt, double *tau,
    double *vn1, double *vn2,
    double *auxv,
    double *dF, magma_int_t lddf);

magma_int_t magma_dlaqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    double *A,  magma_int_t lda,
    magma_int_t *jpvt, double *tau,
    double *vn1, double *vn2,
    double *auxv,
    double *dF, magma_int_t lddf);

magma_int_t magma_dlaqps3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    double *A,  magma_int_t lda,
    magma_int_t *jpvt, double *tau,
    double *vn1, double *vn2,
    double *auxv,
    double *dF, magma_int_t lddf);

magma_int_t magma_dlarf_gpu(  magma_int_t m, magma_int_t n, double *v, double *tau,
                              double *c, magma_int_t ldc, double *xnorm);

magma_int_t magma_dlarfb_gpu( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              const double *dv, magma_int_t ldv,
                              const double *dt, magma_int_t ldt,
                              double *dc,       magma_int_t ldc,
                              double *dwork,    magma_int_t ldwork );

magma_int_t magma_dlarfb2_gpu(magma_int_t m, magma_int_t n, magma_int_t k,
                              const double *dV,    magma_int_t ldv,
                              const double *dT,    magma_int_t ldt,
                              double *dC,          magma_int_t ldc,
                              double *dwork,       magma_int_t ldwork );

magma_int_t magma_dlarfb_gpu_gemm( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              const double *dv, magma_int_t ldv,
                              const double *dt, magma_int_t ldt,
                              double *dc,       magma_int_t ldc,
                              double *dwork,    magma_int_t ldwork,
                              double *dworkvt,  magma_int_t ldworkvt);

magma_int_t magma_dlarfg_gpu( magma_int_t n, double *dx0, double *dx,
                              double *dtau, double *dxnorm, double *dAkk);

magma_int_t magma_dposv_gpu(  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                              double *dA, magma_int_t ldda,
                              double *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_dpotf2_gpu( magma_uplo_t uplo, magma_int_t n,
                              double *dA, magma_int_t lda,
                              magma_int_t *info );

magma_int_t magma_dpotrf_gpu( magma_uplo_t uplo,  magma_int_t n,
                              double *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_dpotrf_mgpu(magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n,
                              double **d_lA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_dpotrf3_mgpu(magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                               magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                               double *d_lA[],  magma_int_t ldda,
                               double *d_lP[],  magma_int_t lddp,
                               double *a,      magma_int_t lda,   magma_int_t h,
                               magma_queue_t stream[][3], magma_event_t event[][5],
                               magma_int_t *info );

magma_int_t magma_dpotri_gpu( magma_uplo_t uplo,  magma_int_t n,
                              double *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_dlauum_gpu( magma_uplo_t uplo,  magma_int_t n,
                              double *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_dtrtri_gpu( magma_uplo_t uplo,  magma_diag_t diag, magma_int_t n,
                              double *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_dsytrd_gpu( magma_uplo_t uplo, magma_int_t n,
                              double *da, magma_int_t ldda,
                              double *d, double *e, double *tau,
                              double *wa,  magma_int_t ldwa,
                              double *work, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_dsytrd2_gpu(magma_uplo_t uplo, magma_int_t n,
                              double *da, magma_int_t ldda,
                              double *d, double *e, double *tau,
                              double *wa,  magma_int_t ldwa,
                              double *work, magma_int_t lwork,
                              double *dwork, magma_int_t ldwork,
                              magma_int_t *info);

double magma_dlatrd_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo,
    magma_int_t n0, magma_int_t n, magma_int_t nb, magma_int_t nb0,
    double *a,  magma_int_t lda,
    double *e, double *tau,
    double *w,   magma_int_t ldw,
    double **da, magma_int_t ldda, magma_int_t offset,
    double **dw, magma_int_t lddw,
    double *dwork[MagmaMaxGPUs], magma_int_t ldwork,
    magma_int_t k,
    double  *dx[MagmaMaxGPUs], double *dy[MagmaMaxGPUs],
    double *work,
    magma_queue_t stream[][10],
    double *times );

magma_int_t magma_dsytrd_mgpu(magma_int_t num_gpus, magma_int_t k, magma_uplo_t uplo, magma_int_t n,
                              double *a, magma_int_t lda,
                              double *d, double *e, double *tau,
                              double *work, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_dsytrd_sb2st(magma_int_t threads, magma_uplo_t uplo,
                              magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
                              double *A, magma_int_t lda,
                              double *D, double *E,
                              double *V, magma_int_t ldv,
                              double *TAU, magma_int_t compT,
                              double *T, magma_int_t ldt);

magma_int_t magma_dsytrd_sy2sb(magma_uplo_t uplo, magma_int_t n, magma_int_t NB,
                              double *a, magma_int_t lda,
                              double *tau, double *work, magma_int_t lwork,
                              double *dT, magma_int_t threads,
                              magma_int_t *info);

magma_int_t magma_dsytrd_sy2sb_mgpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                              double *a, magma_int_t lda,
                              double *tau,
                              double *work, magma_int_t lwork,
                              double *dAmgpu[], magma_int_t ldda,
                              double *dTmgpu[], magma_int_t lddt,
                              magma_int_t ngpu, magma_int_t distblk,
                              magma_queue_t streams[][20], magma_int_t nstream,
                              magma_int_t threads, magma_int_t *info);

magma_int_t magma_dsytrd_sy2sb_mgpu_spec( magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                              double *a, magma_int_t lda,
                              double *tau,
                              double *work, magma_int_t lwork,
                              double *dAmgpu[], magma_int_t ldda,
                              double *dTmgpu[], magma_int_t lddt,
                              magma_int_t ngpu, magma_int_t distblk,
                              magma_queue_t streams[][20], magma_int_t nstream,
                              magma_int_t threads, magma_int_t *info);

magma_int_t magma_dpotrs_gpu( magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs,
                              double *dA, magma_int_t ldda,
                              double *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_dssssm_gpu( magma_storev_t storev, magma_int_t m1, magma_int_t n1,
                              magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib,
                              double *dA1, magma_int_t ldda1,
                              double *dA2, magma_int_t ldda2,
                              double *dL1, magma_int_t lddl1,
                              double *dL2, magma_int_t lddl2,
                              magma_int_t *IPIV, magma_int_t *info);

magma_int_t magma_dtstrf_gpu( magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
                              double *hU, magma_int_t ldhu, double *dU, magma_int_t lddu,
                              double *hA, magma_int_t ldha, double *dA, magma_int_t ldda,
                              double *hL, magma_int_t ldhl, double *dL, magma_int_t lddl,
                              magma_int_t *ipiv,
                              double *hwork, magma_int_t ldhwork,
                              double *dwork, magma_int_t lddwork,
                              magma_int_t *info);

magma_int_t magma_dorgqr_gpu( magma_int_t m, magma_int_t n, magma_int_t k,
                              double *da, magma_int_t ldda,
                              double *tau, double *dwork,
                              magma_int_t nb, magma_int_t *info );

magma_int_t magma_dormql2_gpu(magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              double *da, magma_int_t ldda,
                              double *tau,
                              double *dc, magma_int_t lddc,
                              double *wa, magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_dormqr_gpu( magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              double *dA,    magma_int_t ldda, double *tau,
                              double *dC,    magma_int_t lddc,
                              double *hwork, magma_int_t lwork,
                              double *dT,    magma_int_t nb, magma_int_t *info);

magma_int_t magma_dormqr2_gpu(magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              double *da,   magma_int_t ldda,
                              double *tau,
                              double *dc,    magma_int_t lddc,
                              double *wa,    magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_dormtr_gpu( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                              magma_int_t m, magma_int_t n,
                              double *da,    magma_int_t ldda,
                              double *tau,
                              double *dc,    magma_int_t lddc,
                              double *wa,    magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_dgeqp3_gpu( magma_int_t m, magma_int_t n,
                              double *A, magma_int_t lda,
                              magma_int_t *jpvt, double *tau,
                              double *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              double *rwork,
                              #endif
                              magma_int_t *info );

magma_int_t magma_dsyevd_gpu( magma_vec_t jobz, magma_uplo_t uplo,
                              magma_int_t n,
                              double *da, magma_int_t ldda,
                              double *w,
                              double *wa,  magma_int_t ldwa,
                              double *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              double *rwork, magma_int_t lrwork,
                              #endif
                              magma_int_t *iwork, magma_int_t liwork,
                              magma_int_t *info);

magma_int_t magma_dsyevdx_gpu(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                              magma_int_t n, double *da,
                              magma_int_t ldda, double vl, double vu,
                              magma_int_t il, magma_int_t iu,
                              magma_int_t *m, double *w,
                              double *wa,  magma_int_t ldwa,
                              double *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              double *rwork, magma_int_t lrwork,
                              #endif
                              magma_int_t *iwork, magma_int_t liwork,
                              magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t magma_dsyevx_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                              double *da, magma_int_t ldda, double vl,
                              double vu, magma_int_t il, magma_int_t iu,
                              double abstol, magma_int_t *m,
                              double *w, double *dz, magma_int_t lddz,
                              double *wa, magma_int_t ldwa,
                              double *wz, magma_int_t ldwz,
                              double *work, magma_int_t lwork,
                              double *rwork, magma_int_t *iwork,
                              magma_int_t *ifail, magma_int_t *info);

magma_int_t magma_dsyevr_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                              double *da, magma_int_t ldda, double vl, double vu,
                              magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
                              double *w, double *dz, magma_int_t lddz,
                              magma_int_t *isuppz,
                              double *wa, magma_int_t ldwa,
                              double *wz, magma_int_t ldwz,
                              double *work, magma_int_t lwork,
                              double *rwork, magma_int_t lrwork, magma_int_t *iwork,
                              magma_int_t liwork, magma_int_t *info);
#endif  // COMPLEX

magma_int_t magma_dsygst_gpu(magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                             double *da, magma_int_t ldda,
                             double *db, magma_int_t lddb, magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA utility function definitions
*/

void magma_dprint    ( magma_int_t m, magma_int_t n, const double  *A, magma_int_t lda  );
void magma_dprint_gpu( magma_int_t m, magma_int_t n, const double *dA, magma_int_t ldda );

void dpanel_to_q( magma_uplo_t uplo, magma_int_t ib, double *A, magma_int_t lda, double *work );
void dq_to_panel( magma_uplo_t uplo, magma_int_t ib, double *A, magma_int_t lda, double *work );

#ifdef __cplusplus
}
#endif

#undef REAL

#endif /* MAGMA_D_H */
