/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:17 2013
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
magma_int_t magma_get_spotrf_nb( magma_int_t m );
magma_int_t magma_get_sgetrf_nb( magma_int_t m );
magma_int_t magma_get_sgetri_nb( magma_int_t m );
magma_int_t magma_get_sgeqp3_nb( magma_int_t m );
magma_int_t magma_get_sgeqrf_nb( magma_int_t m );
magma_int_t magma_get_sgeqlf_nb( magma_int_t m );
magma_int_t magma_get_sgehrd_nb( magma_int_t m );
magma_int_t magma_get_ssytrd_nb( magma_int_t m );
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
   -- MAGMA function definitions / Data on CPU
*/

#ifdef REAL
// only applicable to real [sd] precisions
void magma_smove_eig(magma_range_t range, magma_int_t n, float *w, magma_int_t *il,
                          magma_int_t *iu, float vl, float vu, magma_int_t *m);
#endif

magma_int_t magma_sgebrd( magma_int_t m, magma_int_t n, float *A,
                          magma_int_t lda, float *d, float *e,
                          float *tauq,  float *taup,
                          float *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_sgehrd2(magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          float *A, magma_int_t lda, float *tau,
                          float *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_sgehrd( magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          float *A, magma_int_t lda, float *tau,
                          float *work, magma_int_t lwork,
                          float *dT, magma_int_t *info);

magma_int_t magma_sgelqf( magma_int_t m, magma_int_t n,
                          float *A,    magma_int_t lda,   float *tau,
                          float *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_sgeqlf( magma_int_t m, magma_int_t n,
                          float *A,    magma_int_t lda,   float *tau,
                          float *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_sgeqrf( magma_int_t m, magma_int_t n, float *A,
                          magma_int_t lda, float *tau, float *work,
                          magma_int_t lwork, magma_int_t *info);

magma_int_t magma_sgeqrf4(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                          float *a,    magma_int_t lda, float *tau,
                          float *work, magma_int_t lwork, magma_int_t *info );

magma_int_t magma_sgeqrf_ooc( magma_int_t m, magma_int_t n, float *A,
                          magma_int_t lda, float *tau, float *work,
                          magma_int_t lwork, magma_int_t *info);

magma_int_t magma_sgesv ( magma_int_t n, magma_int_t nrhs,
                          float *A, magma_int_t lda, magma_int_t *ipiv,
                          float *B, magma_int_t ldb, magma_int_t *info);

magma_int_t magma_sgetrf( magma_int_t m, magma_int_t n, float *A,
                          magma_int_t lda, magma_int_t *ipiv,
                          magma_int_t *info);

magma_int_t magma_sgetrf2(magma_int_t m, magma_int_t n, float *a,
                          magma_int_t lda, magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_slaqps( magma_int_t m, magma_int_t n, magma_int_t offset,
                          magma_int_t nb, magma_int_t *kb,
                          float *A,  magma_int_t lda,
                          float *dA, magma_int_t ldda,
                          magma_int_t *jpvt, float *tau, float *vn1, float *vn2,
                          float *auxv,
                          float *F,  magma_int_t ldf,
                          float *dF, magma_int_t lddf );

void        magma_slarfg( magma_int_t n, float *alpha, float *x,
                          magma_int_t incx, float *tau);

magma_int_t magma_slatrd( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, float *a,
                          magma_int_t lda, float *e, float *tau,
                          float *w, magma_int_t ldw,
                          float *da, magma_int_t ldda,
                          float *dw, magma_int_t lddw);

magma_int_t magma_slatrd2(magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                          float *a,  magma_int_t lda,
                          float *e, float *tau,
                          float *w,  magma_int_t ldw,
                          float *da, magma_int_t ldda,
                          float *dw, magma_int_t lddw,
                          float *dwork, magma_int_t ldwork);

magma_int_t magma_slahr2( magma_int_t m, magma_int_t n, magma_int_t nb,
                          float *da, float *dv, float *a,
                          magma_int_t lda, float *tau, float *t,
                          magma_int_t ldt, float *y, magma_int_t ldy);

magma_int_t magma_slahru( magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
                          float *a, magma_int_t lda,
                          float *da, float *y,
                          float *v, float *t,
                          float *dwork);

magma_int_t magma_sposv ( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                          float *A, magma_int_t lda,
                          float *B, magma_int_t ldb, magma_int_t *info);

magma_int_t magma_spotrf( magma_uplo_t uplo, magma_int_t n, float *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_spotri( magma_uplo_t uplo, magma_int_t n, float *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_slauum( magma_uplo_t uplo, magma_int_t n, float *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_strtri( magma_uplo_t uplo, magma_diag_t diag, magma_int_t n, float *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_ssytrd( magma_uplo_t uplo, magma_int_t n, float *A,
                          magma_int_t lda, float *d, float *e,
                          float *tau, float *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_sorgqr( magma_int_t m, magma_int_t n, magma_int_t k,
                          float *a, magma_int_t lda,
                          float *tau, float *dT,
                          magma_int_t nb, magma_int_t *info );

magma_int_t magma_sorgqr2(magma_int_t m, magma_int_t n, magma_int_t k,
                          float *a, magma_int_t lda,
                          float *tau, magma_int_t *info );

magma_int_t magma_sormql( magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          float *a, magma_int_t lda,
                          float *tau,
                          float *c, magma_int_t ldc,
                          float *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_sormqr( magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          float *a, magma_int_t lda, float *tau,
                          float *c, magma_int_t ldc,
                          float *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_sormtr( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                          magma_int_t m, magma_int_t n,
                          float *a,    magma_int_t lda,
                          float *tau,
                          float *c,    magma_int_t ldc,
                          float *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_sorghr( magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          float *a, magma_int_t lda,
                          float *tau,
                          float *dT, magma_int_t nb,
                          magma_int_t *info);

magma_int_t  magma_sgeev( magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
                          float *a, magma_int_t lda,
                          #ifdef COMPLEX
                          float *w,
                          #else
                          float *wr, float *wi,
                          #endif
                          float *vl, magma_int_t ldvl,
                          float *vr, magma_int_t ldvr,
                          float *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork,
                          #endif
                          magma_int_t *info);

magma_int_t magma_sgeqp3( magma_int_t m, magma_int_t n,
                          float *a, magma_int_t lda,
                          magma_int_t *jpvt, float *tau,
                          float *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork,
                          #endif
                          magma_int_t *info);

magma_int_t magma_sgesvd( magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
                          float *a,    magma_int_t lda, float *s,
                          float *u,    magma_int_t ldu,
                          float *vt,   magma_int_t ldvt,
                          float *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork,
                          #endif
                          magma_int_t *info );

magma_int_t magma_ssyevd( magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
                          float *a, magma_int_t lda, float *w,
                          float *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_ssyevdx(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          float *a, magma_int_t lda,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, float *w,
                          float *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_ssyevdx_2stage(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n,
                          float *a, magma_int_t lda,
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
magma_int_t magma_ssyevx( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          float *a, magma_int_t lda, float vl, float vu,
                          magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
                          float *w, float *z, magma_int_t ldz,
                          float *work, magma_int_t lwork,
                          float *rwork, magma_int_t *iwork,
                          magma_int_t *ifail, magma_int_t *info);

// no real [sd] precisions available
magma_int_t magma_ssyevr( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          float *a, magma_int_t lda, float vl, float vu,
                          magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
                          float *w, float *z, magma_int_t ldz,
                          magma_int_t *isuppz,
                          float *work, magma_int_t lwork,
                          float *rwork, magma_int_t lrwork, magma_int_t *iwork,
                          magma_int_t liwork, magma_int_t *info);
#endif  // COMPLEX

magma_int_t magma_ssygvd( magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
                          float *a, magma_int_t lda,
                          float *b, magma_int_t ldb,
                          float *w, float *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
                          
magma_int_t magma_ssygvdx(magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, float *a, magma_int_t lda,
                          float *b, magma_int_t ldb,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, float *w,
                          float *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_ssygvdx_2stage(magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          float *a, magma_int_t lda,
                          float *b, magma_int_t ldb,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, float *w,
                          float *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork,
                          magma_int_t *info);

magma_int_t magma_ssygvx( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, float *a, magma_int_t lda,
                          float *b, magma_int_t ldb,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          float abstol, magma_int_t *m, float *w,
                          float *z, magma_int_t ldz,
                          float *work, magma_int_t lwork, float *rwork,
                          magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info);

magma_int_t magma_ssygvr( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, float *a, magma_int_t lda,
                          float *b, magma_int_t ldb,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          float abstol, magma_int_t *m, float *w,
                          float *z, magma_int_t ldz,
                          magma_int_t *isuppz, float *work, magma_int_t lwork,
                          float *rwork, magma_int_t lrwork, magma_int_t *iwork,
                          magma_int_t liwork, magma_int_t *info);

magma_int_t magma_sstedx( magma_range_t range, magma_int_t n, float vl, float vu,
                          magma_int_t il, magma_int_t iu, float *D, float *E,
                          float *Z, magma_int_t ldz,
                          float *rwork, magma_int_t lrwork,
                          magma_int_t *iwork, magma_int_t liwork,
                          float *dwork, magma_int_t *info);

#ifdef REAL
// only applicable to real [sd] precisions
magma_int_t magma_slaex0( magma_int_t n, float *d, float *e, float *q, magma_int_t ldq,
                          float *work, magma_int_t *iwork, float *dwork,
                          magma_range_t range, float vl, float vu,
                          magma_int_t il, magma_int_t iu, magma_int_t *info);

magma_int_t magma_slaex1( magma_int_t n, float *d, float *q, magma_int_t ldq,
                          magma_int_t *indxq, float rho, magma_int_t cutpnt,
                          float *work, magma_int_t *iwork, float *dwork,
                          magma_range_t range, float vl, float vu,
                          magma_int_t il, magma_int_t iu, magma_int_t *info);

magma_int_t magma_slaex3( magma_int_t k, magma_int_t n, magma_int_t n1, float *d,
                          float *q, magma_int_t ldq, float rho,
                          float *dlamda, float *q2, magma_int_t *indx,
                          magma_int_t *ctot, float *w, float *s, magma_int_t *indxq,
                          float *dwork,
                          magma_range_t range, float vl, float vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *info );
#endif  // REAL

magma_int_t magma_ssygst( magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                          float *a, magma_int_t lda,
                          float *b, magma_int_t ldb, magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on CPU / Multi-GPU
*/
magma_int_t magma_slahr2_m(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    float *A, magma_int_t lda,
    float *tau,
    float *T, magma_int_t ldt,
    float *Y, magma_int_t ldy,
    struct sgehrd_data *data );

magma_int_t magma_slahru_m(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    float *A, magma_int_t lda,
    struct sgehrd_data *data );

magma_int_t magma_sgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    float *A, magma_int_t lda,
    #ifdef COMPLEX
    float *w,
    #else
    float *wr, float *wi,
    #endif
    float *vl, magma_int_t ldvl,
    float *vr, magma_int_t ldvr,
    float *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info );

magma_int_t magma_sgehrd_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    float *A, magma_int_t lda,
    float *tau,
    float *work, magma_int_t lwork,
    float *T,
    magma_int_t *info );

magma_int_t magma_sorghr_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    float *A, magma_int_t lda,
    float *tau,
    float *T, magma_int_t nb,
    magma_int_t *info );

magma_int_t magma_sorgqr_m(
    magma_int_t m, magma_int_t n, magma_int_t k,
    float *A, magma_int_t lda,
    float *tau,
    float *T, magma_int_t nb,
    magma_int_t *info );

magma_int_t magma_spotrf_m( magma_int_t num_gpus,
                            magma_uplo_t uplo, magma_int_t n,
                            float *A, magma_int_t lda,
                            magma_int_t *info);

magma_int_t magma_sstedx_m( magma_int_t nrgpu,
                            magma_range_t range, magma_int_t n, float vl, float vu,
                            magma_int_t il, magma_int_t iu, float *D, float *E,
                            float *Z, magma_int_t ldz,
                            float *rwork, magma_int_t ldrwork, magma_int_t *iwork,
                            magma_int_t liwork, magma_int_t *info);

magma_int_t magma_strsm_m ( magma_int_t nrgpu,
                            magma_side_t side, magma_uplo_t uplo, magma_trans_t transa, magma_diag_t diag,
                            magma_int_t m, magma_int_t n, float alpha,
                            float *a, magma_int_t lda,
                            float *b, magma_int_t ldb);

magma_int_t magma_sormqr_m( magma_int_t nrgpu, magma_side_t side, magma_trans_t trans,
                            magma_int_t m, magma_int_t n, magma_int_t k,
                            float *a,    magma_int_t lda,
                            float *tau,
                            float *c,    magma_int_t ldc,
                            float *work, magma_int_t lwork,
                            magma_int_t *info);

magma_int_t magma_sormtr_m( magma_int_t nrgpu,
                            magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                            magma_int_t m, magma_int_t n,
                            float *a,    magma_int_t lda,
                            float *tau,
                            float *c,    magma_int_t ldc,
                            float *work, magma_int_t lwork,
                            magma_int_t *info);

magma_int_t magma_ssygst_m( magma_int_t nrgpu,
                            magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                            float *a, magma_int_t lda,
                            float *b, magma_int_t ldb,
                            magma_int_t *info);

magma_int_t magma_ssyevd_m( magma_int_t nrgpu,
                            magma_vec_t jobz, magma_uplo_t uplo,
                            magma_int_t n,
                            float *a, magma_int_t lda,
                            float *w,
                            float *work, magma_int_t lwork,
                            #ifdef COMPLEX
                            float *rwork, magma_int_t lrwork,
                            #endif
                            magma_int_t *iwork, magma_int_t liwork,
                            magma_int_t *info);

magma_int_t magma_ssygvd_m( magma_int_t nrgpu,
                            magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo,
                            magma_int_t n,
                            float *a, magma_int_t lda,
                            float *b, magma_int_t ldb,
                            float *w,
                            float *work, magma_int_t lwork,
                            #ifdef COMPLEX
                            float *rwork, magma_int_t lrwork,
                            #endif
                            magma_int_t *iwork, magma_int_t liwork,
                            magma_int_t *info);

magma_int_t magma_ssyevdx_m( magma_int_t nrgpu,
                             magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                             magma_int_t n,
                             float *a, magma_int_t lda,
                             float vl, float vu, magma_int_t il, magma_int_t iu,
                             magma_int_t *m, float *w,
                             float *work, magma_int_t lwork,
                             #ifdef COMPLEX
                             float *rwork, magma_int_t lrwork,
                             #endif
                             magma_int_t *iwork, magma_int_t liwork,
                             magma_int_t *info);

magma_int_t magma_ssygvdx_m( magma_int_t nrgpu,
                             magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                             magma_int_t n,
                             float *a, magma_int_t lda,
                             float *b, magma_int_t ldb,
                             float vl, float vu, magma_int_t il, magma_int_t iu,
                             magma_int_t *m, float *w,
                             float *work, magma_int_t lwork,
                             #ifdef COMPLEX
                             float *rwork, magma_int_t lrwork,
                             #endif
                             magma_int_t *iwork, magma_int_t liwork,
                             magma_int_t *info);

magma_int_t magma_ssyevdx_2stage_m( magma_int_t nrgpu,
                                    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                                    magma_int_t n,
                                    float *a, magma_int_t lda,
                                    float vl, float vu, magma_int_t il, magma_int_t iu,
                                    magma_int_t *m, float *w,
                                    float *work, magma_int_t lwork,
                                    #ifdef COMPLEX
                                    float *rwork, magma_int_t lrwork,
                                    #endif
                                    magma_int_t *iwork, magma_int_t liwork,
                                    magma_int_t *info);

magma_int_t magma_ssygvdx_2stage_m( magma_int_t nrgpu,
                                    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                                    magma_int_t n,
                                    float *a, magma_int_t lda,
                                    float *b, magma_int_t ldb,
                                    float vl, float vu, magma_int_t il, magma_int_t iu,
                                    magma_int_t *m, float *w,
                                    float *work, magma_int_t lwork,
                                    #ifdef COMPLEX
                                    float *rwork, magma_int_t lrwork,
                                    #endif
                                    magma_int_t *iwork, magma_int_t liwork,
                                    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on GPU
*/
magma_int_t magma_sgels_gpu(  magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              float *dA,    magma_int_t ldda,
                              float *dB,    magma_int_t lddb,
                              float *hwork, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_sgels3_gpu( magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              float *dA,    magma_int_t ldda,
                              float *dB,    magma_int_t lddb,
                              float *hwork, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_sgelqf_gpu( magma_int_t m, magma_int_t n,
                              float *dA,    magma_int_t ldda,   float *tau,
                              float *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_sgeqr2x_gpu(
    magma_int_t *m, magma_int_t *n, float *dA,
    magma_int_t *ldda, float *dtau,
    float *dT, float *ddA,
    float *dwork, magma_int_t *info);

magma_int_t magma_sgeqr2x2_gpu(
    magma_int_t *m, magma_int_t *n, float *dA,
    magma_int_t *ldda, float *dtau,
    float *dT, float *ddA,
    float *dwork, magma_int_t *info);

magma_int_t magma_sgeqr2x3_gpu(
    magma_int_t *m, magma_int_t *n, float *dA,
    magma_int_t *ldda, float *dtau,
    float *dT, float *ddA,
    float *dwork, magma_int_t *info);

magma_int_t magma_sgeqr2x4_gpu(
    magma_int_t *m, magma_int_t *n, float *dA,
    magma_int_t *ldda, float *dtau,
    float *dT, float *ddA,
    float *dwork, magma_int_t *info, magma_queue_t stream);

magma_int_t magma_sgeqrf_gpu( magma_int_t m, magma_int_t n,
                              float *dA,  magma_int_t ldda,
                              float *tau, float *dT,
                              magma_int_t *info);

magma_int_t magma_sgeqrf2_gpu(magma_int_t m, magma_int_t n,
                              float *dA,  magma_int_t ldda,
                              float *tau, magma_int_t *info);

magma_int_t magma_sgeqrf2_mgpu(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                               float **dlA, magma_int_t ldda,
                               float *tau, magma_int_t *info );

magma_int_t magma_sgeqrf3_gpu(magma_int_t m, magma_int_t n,
                              float *dA,  magma_int_t ldda,
                              float *tau, float *dT,
                              magma_int_t *info);

magma_int_t magma_sgeqr2_gpu( magma_int_t m, magma_int_t n,
                              float *dA,  magma_int_t lda,
                              float *tau, float *work,
                              magma_int_t *info);

magma_int_t magma_sgeqrs_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              float *dA,     magma_int_t ldda,
                              float *tau,   float *dT,
                              float *dB,    magma_int_t lddb,
                              float *hwork, magma_int_t lhwork,
                              magma_int_t *info);

magma_int_t magma_sgeqrs3_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              float *dA,     magma_int_t ldda,
                              float *tau,   float *dT,
                              float *dB,    magma_int_t lddb,
                              float *hwork, magma_int_t lhwork,
                              magma_int_t *info);

magma_int_t magma_sgessm_gpu( magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib,
                              magma_int_t *ipiv,
                              float *dL1, magma_int_t lddl1,
                              float *dL,  magma_int_t lddl,
                              float *dA,  magma_int_t ldda,
                              magma_int_t *info);

magma_int_t magma_sgesv_gpu(  magma_int_t n, magma_int_t nrhs,
                              float *dA, magma_int_t ldda, magma_int_t *ipiv,
                              float *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_sgetf2_gpu( magma_int_t m, magma_int_t n,
                              float *dA, magma_int_t lda, magma_int_t *ipiv,
                              magma_int_t* info );

magma_int_t magma_sgetrf_incpiv_gpu( magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t ib,
                              float *hA, magma_int_t ldha, float *dA, magma_int_t ldda,
                              float *hL, magma_int_t ldhl, float *dL, magma_int_t lddl,
                              magma_int_t *ipiv,
                              float *dwork, magma_int_t lddwork,
                              magma_int_t *info);

magma_int_t magma_sgetrf_gpu( magma_int_t m, magma_int_t n,
                              float *dA, magma_int_t ldda,
                              magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_sgetrf_mgpu(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                              float **d_lA, magma_int_t ldda,
                              magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_sgetrf_m(magma_int_t num_gpus0, magma_int_t m, magma_int_t n, float *a, magma_int_t lda,
                           magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_sgetrf_piv(magma_int_t m, magma_int_t n, magma_int_t NB,
                             float *a, magma_int_t lda, magma_int_t *ipiv,
                             magma_int_t *info);

magma_int_t magma_sgetrf2_mgpu(magma_int_t num_gpus,
                               magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
                               float *d_lAT[], magma_int_t lddat, magma_int_t *ipiv,
                               float *d_lAP[], float *a, magma_int_t lda,
                               magma_queue_t streaml[][2], magma_int_t *info);

magma_int_t
      magma_sgetrf_nopiv_gpu( magma_int_t m, magma_int_t n,
                              float *dA, magma_int_t ldda,
                              magma_int_t *info);

magma_int_t magma_sgetri_gpu( magma_int_t n,
                              float *dA, magma_int_t ldda, magma_int_t *ipiv,
                              float *dwork, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_sgetrs_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                              float *dA, magma_int_t ldda, magma_int_t *ipiv,
                              float *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_slabrd_gpu( magma_int_t m, magma_int_t n, magma_int_t nb,
                              float *a, magma_int_t lda, float *da, magma_int_t ldda,
                              float *d, float *e, float *tauq, float *taup,
                              float *x, magma_int_t ldx, float *dx, magma_int_t lddx,
                              float *y, magma_int_t ldy, float *dy, magma_int_t lddy);

magma_int_t magma_slaqps_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    float *A,  magma_int_t lda,
    magma_int_t *jpvt, float *tau,
    float *vn1, float *vn2,
    float *auxv,
    float *dF, magma_int_t lddf);

magma_int_t magma_slaqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    float *A,  magma_int_t lda,
    magma_int_t *jpvt, float *tau,
    float *vn1, float *vn2,
    float *auxv,
    float *dF, magma_int_t lddf);

magma_int_t magma_slaqps3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    float *A,  magma_int_t lda,
    magma_int_t *jpvt, float *tau,
    float *vn1, float *vn2,
    float *auxv,
    float *dF, magma_int_t lddf);

magma_int_t magma_slarf_gpu(  magma_int_t m, magma_int_t n, float *v, float *tau,
                              float *c, magma_int_t ldc, float *xnorm);

magma_int_t magma_slarfb_gpu( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              const float *dv, magma_int_t ldv,
                              const float *dt, magma_int_t ldt,
                              float *dc,       magma_int_t ldc,
                              float *dwork,    magma_int_t ldwork );

magma_int_t magma_slarfb2_gpu(magma_int_t m, magma_int_t n, magma_int_t k,
                              const float *dV,    magma_int_t ldv,
                              const float *dT,    magma_int_t ldt,
                              float *dC,          magma_int_t ldc,
                              float *dwork,       magma_int_t ldwork );

magma_int_t magma_slarfb_gpu_gemm( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              const float *dv, magma_int_t ldv,
                              const float *dt, magma_int_t ldt,
                              float *dc,       magma_int_t ldc,
                              float *dwork,    magma_int_t ldwork,
                              float *dworkvt,  magma_int_t ldworkvt);

magma_int_t magma_slarfg_gpu( magma_int_t n, float *dx0, float *dx,
                              float *dtau, float *dxnorm, float *dAkk);

magma_int_t magma_sposv_gpu(  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                              float *dA, magma_int_t ldda,
                              float *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_spotf2_gpu( magma_uplo_t uplo, magma_int_t n,
                              float *dA, magma_int_t lda,
                              magma_int_t *info );

magma_int_t magma_spotrf_gpu( magma_uplo_t uplo,  magma_int_t n,
                              float *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_spotrf_mgpu(magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n,
                              float **d_lA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_spotrf3_mgpu(magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                               magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                               float *d_lA[],  magma_int_t ldda,
                               float *d_lP[],  magma_int_t lddp,
                               float *a,      magma_int_t lda,   magma_int_t h,
                               magma_queue_t stream[][3], magma_event_t event[][5],
                               magma_int_t *info );

magma_int_t magma_spotri_gpu( magma_uplo_t uplo,  magma_int_t n,
                              float *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_slauum_gpu( magma_uplo_t uplo,  magma_int_t n,
                              float *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_strtri_gpu( magma_uplo_t uplo,  magma_diag_t diag, magma_int_t n,
                              float *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_ssytrd_gpu( magma_uplo_t uplo, magma_int_t n,
                              float *da, magma_int_t ldda,
                              float *d, float *e, float *tau,
                              float *wa,  magma_int_t ldwa,
                              float *work, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_ssytrd2_gpu(magma_uplo_t uplo, magma_int_t n,
                              float *da, magma_int_t ldda,
                              float *d, float *e, float *tau,
                              float *wa,  magma_int_t ldwa,
                              float *work, magma_int_t lwork,
                              float *dwork, magma_int_t ldwork,
                              magma_int_t *info);

float magma_slatrd_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo,
    magma_int_t n0, magma_int_t n, magma_int_t nb, magma_int_t nb0,
    float *a,  magma_int_t lda,
    float *e, float *tau,
    float *w,   magma_int_t ldw,
    float **da, magma_int_t ldda, magma_int_t offset,
    float **dw, magma_int_t lddw,
    float *dwork[MagmaMaxGPUs], magma_int_t ldwork,
    magma_int_t k,
    float  *dx[MagmaMaxGPUs], float *dy[MagmaMaxGPUs],
    float *work,
    magma_queue_t stream[][10],
    float *times );

magma_int_t magma_ssytrd_mgpu(magma_int_t num_gpus, magma_int_t k, magma_uplo_t uplo, magma_int_t n,
                              float *a, magma_int_t lda,
                              float *d, float *e, float *tau,
                              float *work, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_ssytrd_sb2st(magma_int_t threads, magma_uplo_t uplo,
                              magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
                              float *A, magma_int_t lda,
                              float *D, float *E,
                              float *V, magma_int_t ldv,
                              float *TAU, magma_int_t compT,
                              float *T, magma_int_t ldt);

magma_int_t magma_ssytrd_sy2sb(magma_uplo_t uplo, magma_int_t n, magma_int_t NB,
                              float *a, magma_int_t lda,
                              float *tau, float *work, magma_int_t lwork,
                              float *dT, magma_int_t threads,
                              magma_int_t *info);

magma_int_t magma_ssytrd_sy2sb_mgpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                              float *a, magma_int_t lda,
                              float *tau,
                              float *work, magma_int_t lwork,
                              float *dAmgpu[], magma_int_t ldda,
                              float *dTmgpu[], magma_int_t lddt,
                              magma_int_t ngpu, magma_int_t distblk,
                              magma_queue_t streams[][20], magma_int_t nstream,
                              magma_int_t threads, magma_int_t *info);

magma_int_t magma_ssytrd_sy2sb_mgpu_spec( magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                              float *a, magma_int_t lda,
                              float *tau,
                              float *work, magma_int_t lwork,
                              float *dAmgpu[], magma_int_t ldda,
                              float *dTmgpu[], magma_int_t lddt,
                              magma_int_t ngpu, magma_int_t distblk,
                              magma_queue_t streams[][20], magma_int_t nstream,
                              magma_int_t threads, magma_int_t *info);

magma_int_t magma_spotrs_gpu( magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs,
                              float *dA, magma_int_t ldda,
                              float *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_sssssm_gpu( magma_storev_t storev, magma_int_t m1, magma_int_t n1,
                              magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib,
                              float *dA1, magma_int_t ldda1,
                              float *dA2, magma_int_t ldda2,
                              float *dL1, magma_int_t lddl1,
                              float *dL2, magma_int_t lddl2,
                              magma_int_t *IPIV, magma_int_t *info);

magma_int_t magma_ststrf_gpu( magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
                              float *hU, magma_int_t ldhu, float *dU, magma_int_t lddu,
                              float *hA, magma_int_t ldha, float *dA, magma_int_t ldda,
                              float *hL, magma_int_t ldhl, float *dL, magma_int_t lddl,
                              magma_int_t *ipiv,
                              float *hwork, magma_int_t ldhwork,
                              float *dwork, magma_int_t lddwork,
                              magma_int_t *info);

magma_int_t magma_sorgqr_gpu( magma_int_t m, magma_int_t n, magma_int_t k,
                              float *da, magma_int_t ldda,
                              float *tau, float *dwork,
                              magma_int_t nb, magma_int_t *info );

magma_int_t magma_sormql2_gpu(magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              float *da, magma_int_t ldda,
                              float *tau,
                              float *dc, magma_int_t lddc,
                              float *wa, magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_sormqr_gpu( magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              float *dA,    magma_int_t ldda, float *tau,
                              float *dC,    magma_int_t lddc,
                              float *hwork, magma_int_t lwork,
                              float *dT,    magma_int_t nb, magma_int_t *info);

magma_int_t magma_sormqr2_gpu(magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              float *da,   magma_int_t ldda,
                              float *tau,
                              float *dc,    magma_int_t lddc,
                              float *wa,    magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_sormtr_gpu( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                              magma_int_t m, magma_int_t n,
                              float *da,    magma_int_t ldda,
                              float *tau,
                              float *dc,    magma_int_t lddc,
                              float *wa,    magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_sgeqp3_gpu( magma_int_t m, magma_int_t n,
                              float *A, magma_int_t lda,
                              magma_int_t *jpvt, float *tau,
                              float *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              float *rwork,
                              #endif
                              magma_int_t *info );

magma_int_t magma_ssyevd_gpu( magma_vec_t jobz, magma_uplo_t uplo,
                              magma_int_t n,
                              float *da, magma_int_t ldda,
                              float *w,
                              float *wa,  magma_int_t ldwa,
                              float *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              float *rwork, magma_int_t lrwork,
                              #endif
                              magma_int_t *iwork, magma_int_t liwork,
                              magma_int_t *info);

magma_int_t magma_ssyevdx_gpu(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                              magma_int_t n, float *da,
                              magma_int_t ldda, float vl, float vu,
                              magma_int_t il, magma_int_t iu,
                              magma_int_t *m, float *w,
                              float *wa,  magma_int_t ldwa,
                              float *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              float *rwork, magma_int_t lrwork,
                              #endif
                              magma_int_t *iwork, magma_int_t liwork,
                              magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t magma_ssyevx_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                              float *da, magma_int_t ldda, float vl,
                              float vu, magma_int_t il, magma_int_t iu,
                              float abstol, magma_int_t *m,
                              float *w, float *dz, magma_int_t lddz,
                              float *wa, magma_int_t ldwa,
                              float *wz, magma_int_t ldwz,
                              float *work, magma_int_t lwork,
                              float *rwork, magma_int_t *iwork,
                              magma_int_t *ifail, magma_int_t *info);

magma_int_t magma_ssyevr_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                              float *da, magma_int_t ldda, float vl, float vu,
                              magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
                              float *w, float *dz, magma_int_t lddz,
                              magma_int_t *isuppz,
                              float *wa, magma_int_t ldwa,
                              float *wz, magma_int_t ldwz,
                              float *work, magma_int_t lwork,
                              float *rwork, magma_int_t lrwork, magma_int_t *iwork,
                              magma_int_t liwork, magma_int_t *info);
#endif  // COMPLEX

magma_int_t magma_ssygst_gpu(magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                             float *da, magma_int_t ldda,
                             float *db, magma_int_t lddb, magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA utility function definitions
*/

void magma_sprint    ( magma_int_t m, magma_int_t n, const float  *A, magma_int_t lda  );
void magma_sprint_gpu( magma_int_t m, magma_int_t n, const float *dA, magma_int_t ldda );

void spanel_to_q( magma_uplo_t uplo, magma_int_t ib, float *A, magma_int_t lda, float *work );
void sq_to_panel( magma_uplo_t uplo, magma_int_t ib, float *A, magma_int_t lda, float *work );

#ifdef __cplusplus
}
#endif

#undef REAL

#endif /* MAGMA_S_H */
