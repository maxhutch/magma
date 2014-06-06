/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:17 2013
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
magma_int_t magma_get_cpotrf_nb( magma_int_t m );
magma_int_t magma_get_cgetrf_nb( magma_int_t m );
magma_int_t magma_get_cgetri_nb( magma_int_t m );
magma_int_t magma_get_cgeqp3_nb( magma_int_t m );
magma_int_t magma_get_cgeqrf_nb( magma_int_t m );
magma_int_t magma_get_cgeqlf_nb( magma_int_t m );
magma_int_t magma_get_cgehrd_nb( magma_int_t m );
magma_int_t magma_get_chetrd_nb( magma_int_t m );
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
   -- MAGMA function definitions / Data on CPU
*/

#ifdef REAL
// only applicable to real [sd] precisions
void magma_smove_eig(magma_range_t range, magma_int_t n, float *w, magma_int_t *il,
                          magma_int_t *iu, float vl, float vu, magma_int_t *m);
#endif

magma_int_t magma_cgebrd( magma_int_t m, magma_int_t n, magmaFloatComplex *A,
                          magma_int_t lda, float *d, float *e,
                          magmaFloatComplex *tauq,  magmaFloatComplex *taup,
                          magmaFloatComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_cgehrd2(magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          magmaFloatComplex *A, magma_int_t lda, magmaFloatComplex *tau,
                          magmaFloatComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_cgehrd( magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          magmaFloatComplex *A, magma_int_t lda, magmaFloatComplex *tau,
                          magmaFloatComplex *work, magma_int_t lwork,
                          magmaFloatComplex *dT, magma_int_t *info);

magma_int_t magma_cgelqf( magma_int_t m, magma_int_t n,
                          magmaFloatComplex *A,    magma_int_t lda,   magmaFloatComplex *tau,
                          magmaFloatComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_cgeqlf( magma_int_t m, magma_int_t n,
                          magmaFloatComplex *A,    magma_int_t lda,   magmaFloatComplex *tau,
                          magmaFloatComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_cgeqrf( magma_int_t m, magma_int_t n, magmaFloatComplex *A,
                          magma_int_t lda, magmaFloatComplex *tau, magmaFloatComplex *work,
                          magma_int_t lwork, magma_int_t *info);

magma_int_t magma_cgeqrf4(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                          magmaFloatComplex *a,    magma_int_t lda, magmaFloatComplex *tau,
                          magmaFloatComplex *work, magma_int_t lwork, magma_int_t *info );

magma_int_t magma_cgeqrf_ooc( magma_int_t m, magma_int_t n, magmaFloatComplex *A,
                          magma_int_t lda, magmaFloatComplex *tau, magmaFloatComplex *work,
                          magma_int_t lwork, magma_int_t *info);

magma_int_t magma_cgesv ( magma_int_t n, magma_int_t nrhs,
                          magmaFloatComplex *A, magma_int_t lda, magma_int_t *ipiv,
                          magmaFloatComplex *B, magma_int_t ldb, magma_int_t *info);

magma_int_t magma_cgetrf( magma_int_t m, magma_int_t n, magmaFloatComplex *A,
                          magma_int_t lda, magma_int_t *ipiv,
                          magma_int_t *info);

magma_int_t magma_cgetrf2(magma_int_t m, magma_int_t n, magmaFloatComplex *a,
                          magma_int_t lda, magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_claqps( magma_int_t m, magma_int_t n, magma_int_t offset,
                          magma_int_t nb, magma_int_t *kb,
                          magmaFloatComplex *A,  magma_int_t lda,
                          magmaFloatComplex *dA, magma_int_t ldda,
                          magma_int_t *jpvt, magmaFloatComplex *tau, float *vn1, float *vn2,
                          magmaFloatComplex *auxv,
                          magmaFloatComplex *F,  magma_int_t ldf,
                          magmaFloatComplex *dF, magma_int_t lddf );

void        magma_clarfg( magma_int_t n, magmaFloatComplex *alpha, magmaFloatComplex *x,
                          magma_int_t incx, magmaFloatComplex *tau);

magma_int_t magma_clatrd( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magmaFloatComplex *a,
                          magma_int_t lda, float *e, magmaFloatComplex *tau,
                          magmaFloatComplex *w, magma_int_t ldw,
                          magmaFloatComplex *da, magma_int_t ldda,
                          magmaFloatComplex *dw, magma_int_t lddw);

magma_int_t magma_clatrd2(magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                          magmaFloatComplex *a,  magma_int_t lda,
                          float *e, magmaFloatComplex *tau,
                          magmaFloatComplex *w,  magma_int_t ldw,
                          magmaFloatComplex *da, magma_int_t ldda,
                          magmaFloatComplex *dw, magma_int_t lddw,
                          magmaFloatComplex *dwork, magma_int_t ldwork);

magma_int_t magma_clahr2( magma_int_t m, magma_int_t n, magma_int_t nb,
                          magmaFloatComplex *da, magmaFloatComplex *dv, magmaFloatComplex *a,
                          magma_int_t lda, magmaFloatComplex *tau, magmaFloatComplex *t,
                          magma_int_t ldt, magmaFloatComplex *y, magma_int_t ldy);

magma_int_t magma_clahru( magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
                          magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *da, magmaFloatComplex *y,
                          magmaFloatComplex *v, magmaFloatComplex *t,
                          magmaFloatComplex *dwork);

magma_int_t magma_cposv ( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                          magmaFloatComplex *A, magma_int_t lda,
                          magmaFloatComplex *B, magma_int_t ldb, magma_int_t *info);

magma_int_t magma_cpotrf( magma_uplo_t uplo, magma_int_t n, magmaFloatComplex *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_cpotri( magma_uplo_t uplo, magma_int_t n, magmaFloatComplex *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_clauum( magma_uplo_t uplo, magma_int_t n, magmaFloatComplex *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_ctrtri( magma_uplo_t uplo, magma_diag_t diag, magma_int_t n, magmaFloatComplex *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_chetrd( magma_uplo_t uplo, magma_int_t n, magmaFloatComplex *A,
                          magma_int_t lda, float *d, float *e,
                          magmaFloatComplex *tau, magmaFloatComplex *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_cungqr( magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *tau, magmaFloatComplex *dT,
                          magma_int_t nb, magma_int_t *info );

magma_int_t magma_cungqr2(magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *tau, magma_int_t *info );

magma_int_t magma_cunmql( magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *tau,
                          magmaFloatComplex *c, magma_int_t ldc,
                          magmaFloatComplex *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_cunmqr( magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaFloatComplex *a, magma_int_t lda, magmaFloatComplex *tau,
                          magmaFloatComplex *c, magma_int_t ldc,
                          magmaFloatComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_cunmtr( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                          magma_int_t m, magma_int_t n,
                          magmaFloatComplex *a,    magma_int_t lda,
                          magmaFloatComplex *tau,
                          magmaFloatComplex *c,    magma_int_t ldc,
                          magmaFloatComplex *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_cunghr( magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *tau,
                          magmaFloatComplex *dT, magma_int_t nb,
                          magma_int_t *info);

magma_int_t  magma_cgeev( magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda,
                          #ifdef COMPLEX
                          magmaFloatComplex *w,
                          #else
                          float *wr, float *wi,
                          #endif
                          magmaFloatComplex *vl, magma_int_t ldvl,
                          magmaFloatComplex *vr, magma_int_t ldvr,
                          magmaFloatComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork,
                          #endif
                          magma_int_t *info);

magma_int_t magma_cgeqp3( magma_int_t m, magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda,
                          magma_int_t *jpvt, magmaFloatComplex *tau,
                          magmaFloatComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork,
                          #endif
                          magma_int_t *info);

magma_int_t magma_cgesvd( magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
                          magmaFloatComplex *a,    magma_int_t lda, float *s,
                          magmaFloatComplex *u,    magma_int_t ldu,
                          magmaFloatComplex *vt,   magma_int_t ldvt,
                          magmaFloatComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork,
                          #endif
                          magma_int_t *info );

magma_int_t magma_cheevd( magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda, float *w,
                          magmaFloatComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_cheevdx(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, float *w,
                          magmaFloatComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_cheevdx_2stage(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda,
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
magma_int_t magma_cheevx( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda, float vl, float vu,
                          magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
                          float *w, magmaFloatComplex *z, magma_int_t ldz,
                          magmaFloatComplex *work, magma_int_t lwork,
                          float *rwork, magma_int_t *iwork,
                          magma_int_t *ifail, magma_int_t *info);

// no real [sd] precisions available
magma_int_t magma_cheevr( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda, float vl, float vu,
                          magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
                          float *w, magmaFloatComplex *z, magma_int_t ldz,
                          magma_int_t *isuppz,
                          magmaFloatComplex *work, magma_int_t lwork,
                          float *rwork, magma_int_t lrwork, magma_int_t *iwork,
                          magma_int_t liwork, magma_int_t *info);
#endif  // COMPLEX

magma_int_t magma_chegvd( magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *b, magma_int_t ldb,
                          float *w, magmaFloatComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
                          
magma_int_t magma_chegvdx(magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *b, magma_int_t ldb,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, float *w,
                          magmaFloatComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_chegvdx_2stage(magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *b, magma_int_t ldb,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, float *w,
                          magmaFloatComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          float *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork,
                          magma_int_t *info);

magma_int_t magma_chegvx( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *b, magma_int_t ldb,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          float abstol, magma_int_t *m, float *w,
                          magmaFloatComplex *z, magma_int_t ldz,
                          magmaFloatComplex *work, magma_int_t lwork, float *rwork,
                          magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info);

magma_int_t magma_chegvr( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *b, magma_int_t ldb,
                          float vl, float vu, magma_int_t il, magma_int_t iu,
                          float abstol, magma_int_t *m, float *w,
                          magmaFloatComplex *z, magma_int_t ldz,
                          magma_int_t *isuppz, magmaFloatComplex *work, magma_int_t lwork,
                          float *rwork, magma_int_t lrwork, magma_int_t *iwork,
                          magma_int_t liwork, magma_int_t *info);

magma_int_t magma_cstedx( magma_range_t range, magma_int_t n, float vl, float vu,
                          magma_int_t il, magma_int_t iu, float *D, float *E,
                          magmaFloatComplex *Z, magma_int_t ldz,
                          float *rwork, magma_int_t lrwork,
                          magma_int_t *iwork, magma_int_t liwork,
                          float *dwork, magma_int_t *info);

#ifdef REAL
// only applicable to real [sd] precisions
magma_int_t magma_claex0( magma_int_t n, float *d, float *e, float *q, magma_int_t ldq,
                          float *work, magma_int_t *iwork, float *dwork,
                          magma_range_t range, float vl, float vu,
                          magma_int_t il, magma_int_t iu, magma_int_t *info);

magma_int_t magma_claex1( magma_int_t n, float *d, float *q, magma_int_t ldq,
                          magma_int_t *indxq, float rho, magma_int_t cutpnt,
                          float *work, magma_int_t *iwork, float *dwork,
                          magma_range_t range, float vl, float vu,
                          magma_int_t il, magma_int_t iu, magma_int_t *info);

magma_int_t magma_claex3( magma_int_t k, magma_int_t n, magma_int_t n1, float *d,
                          float *q, magma_int_t ldq, float rho,
                          float *dlamda, float *q2, magma_int_t *indx,
                          magma_int_t *ctot, float *w, float *s, magma_int_t *indxq,
                          float *dwork,
                          magma_range_t range, float vl, float vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *info );
#endif  // REAL

magma_int_t magma_chegst( magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                          magmaFloatComplex *a, magma_int_t lda,
                          magmaFloatComplex *b, magma_int_t ldb, magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on CPU / Multi-GPU
*/
magma_int_t magma_clahr2_m(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *T, magma_int_t ldt,
    magmaFloatComplex *Y, magma_int_t ldy,
    struct cgehrd_data *data );

magma_int_t magma_clahru_m(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    magmaFloatComplex *A, magma_int_t lda,
    struct cgehrd_data *data );

magma_int_t magma_cgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    #ifdef COMPLEX
    magmaFloatComplex *w,
    #else
    float *wr, float *wi,
    #endif
    magmaFloatComplex *vl, magma_int_t ldvl,
    magmaFloatComplex *vr, magma_int_t ldvr,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info );

magma_int_t magma_cgehrd_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magmaFloatComplex *T,
    magma_int_t *info );

magma_int_t magma_cunghr_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *T, magma_int_t nb,
    magma_int_t *info );

magma_int_t magma_cungqr_m(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *tau,
    magmaFloatComplex *T, magma_int_t nb,
    magma_int_t *info );

magma_int_t magma_cpotrf_m( magma_int_t num_gpus,
                            magma_uplo_t uplo, magma_int_t n,
                            magmaFloatComplex *A, magma_int_t lda,
                            magma_int_t *info);

magma_int_t magma_cstedx_m( magma_int_t nrgpu,
                            magma_range_t range, magma_int_t n, float vl, float vu,
                            magma_int_t il, magma_int_t iu, float *D, float *E,
                            magmaFloatComplex *Z, magma_int_t ldz,
                            float *rwork, magma_int_t ldrwork, magma_int_t *iwork,
                            magma_int_t liwork, magma_int_t *info);

magma_int_t magma_ctrsm_m ( magma_int_t nrgpu,
                            magma_side_t side, magma_uplo_t uplo, magma_trans_t transa, magma_diag_t diag,
                            magma_int_t m, magma_int_t n, magmaFloatComplex alpha,
                            magmaFloatComplex *a, magma_int_t lda,
                            magmaFloatComplex *b, magma_int_t ldb);

magma_int_t magma_cunmqr_m( magma_int_t nrgpu, magma_side_t side, magma_trans_t trans,
                            magma_int_t m, magma_int_t n, magma_int_t k,
                            magmaFloatComplex *a,    magma_int_t lda,
                            magmaFloatComplex *tau,
                            magmaFloatComplex *c,    magma_int_t ldc,
                            magmaFloatComplex *work, magma_int_t lwork,
                            magma_int_t *info);

magma_int_t magma_cunmtr_m( magma_int_t nrgpu,
                            magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                            magma_int_t m, magma_int_t n,
                            magmaFloatComplex *a,    magma_int_t lda,
                            magmaFloatComplex *tau,
                            magmaFloatComplex *c,    magma_int_t ldc,
                            magmaFloatComplex *work, magma_int_t lwork,
                            magma_int_t *info);

magma_int_t magma_chegst_m( magma_int_t nrgpu,
                            magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                            magmaFloatComplex *a, magma_int_t lda,
                            magmaFloatComplex *b, magma_int_t ldb,
                            magma_int_t *info);

magma_int_t magma_cheevd_m( magma_int_t nrgpu,
                            magma_vec_t jobz, magma_uplo_t uplo,
                            magma_int_t n,
                            magmaFloatComplex *a, magma_int_t lda,
                            float *w,
                            magmaFloatComplex *work, magma_int_t lwork,
                            #ifdef COMPLEX
                            float *rwork, magma_int_t lrwork,
                            #endif
                            magma_int_t *iwork, magma_int_t liwork,
                            magma_int_t *info);

magma_int_t magma_chegvd_m( magma_int_t nrgpu,
                            magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo,
                            magma_int_t n,
                            magmaFloatComplex *a, magma_int_t lda,
                            magmaFloatComplex *b, magma_int_t ldb,
                            float *w,
                            magmaFloatComplex *work, magma_int_t lwork,
                            #ifdef COMPLEX
                            float *rwork, magma_int_t lrwork,
                            #endif
                            magma_int_t *iwork, magma_int_t liwork,
                            magma_int_t *info);

magma_int_t magma_cheevdx_m( magma_int_t nrgpu,
                             magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                             magma_int_t n,
                             magmaFloatComplex *a, magma_int_t lda,
                             float vl, float vu, magma_int_t il, magma_int_t iu,
                             magma_int_t *m, float *w,
                             magmaFloatComplex *work, magma_int_t lwork,
                             #ifdef COMPLEX
                             float *rwork, magma_int_t lrwork,
                             #endif
                             magma_int_t *iwork, magma_int_t liwork,
                             magma_int_t *info);

magma_int_t magma_chegvdx_m( magma_int_t nrgpu,
                             magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                             magma_int_t n,
                             magmaFloatComplex *a, magma_int_t lda,
                             magmaFloatComplex *b, magma_int_t ldb,
                             float vl, float vu, magma_int_t il, magma_int_t iu,
                             magma_int_t *m, float *w,
                             magmaFloatComplex *work, magma_int_t lwork,
                             #ifdef COMPLEX
                             float *rwork, magma_int_t lrwork,
                             #endif
                             magma_int_t *iwork, magma_int_t liwork,
                             magma_int_t *info);

magma_int_t magma_cheevdx_2stage_m( magma_int_t nrgpu,
                                    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                                    magma_int_t n,
                                    magmaFloatComplex *a, magma_int_t lda,
                                    float vl, float vu, magma_int_t il, magma_int_t iu,
                                    magma_int_t *m, float *w,
                                    magmaFloatComplex *work, magma_int_t lwork,
                                    #ifdef COMPLEX
                                    float *rwork, magma_int_t lrwork,
                                    #endif
                                    magma_int_t *iwork, magma_int_t liwork,
                                    magma_int_t *info);

magma_int_t magma_chegvdx_2stage_m( magma_int_t nrgpu,
                                    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                                    magma_int_t n,
                                    magmaFloatComplex *a, magma_int_t lda,
                                    magmaFloatComplex *b, magma_int_t ldb,
                                    float vl, float vu, magma_int_t il, magma_int_t iu,
                                    magma_int_t *m, float *w,
                                    magmaFloatComplex *work, magma_int_t lwork,
                                    #ifdef COMPLEX
                                    float *rwork, magma_int_t lrwork,
                                    #endif
                                    magma_int_t *iwork, magma_int_t liwork,
                                    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on GPU
*/
magma_int_t magma_cgels_gpu(  magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              magmaFloatComplex *dA,    magma_int_t ldda,
                              magmaFloatComplex *dB,    magma_int_t lddb,
                              magmaFloatComplex *hwork, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_cgels3_gpu( magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              magmaFloatComplex *dA,    magma_int_t ldda,
                              magmaFloatComplex *dB,    magma_int_t lddb,
                              magmaFloatComplex *hwork, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_cgelqf_gpu( magma_int_t m, magma_int_t n,
                              magmaFloatComplex *dA,    magma_int_t ldda,   magmaFloatComplex *tau,
                              magmaFloatComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_cgeqr2x_gpu(
    magma_int_t *m, magma_int_t *n, magmaFloatComplex *dA,
    magma_int_t *ldda, magmaFloatComplex *dtau,
    magmaFloatComplex *dT, magmaFloatComplex *ddA,
    float *dwork, magma_int_t *info);

magma_int_t magma_cgeqr2x2_gpu(
    magma_int_t *m, magma_int_t *n, magmaFloatComplex *dA,
    magma_int_t *ldda, magmaFloatComplex *dtau,
    magmaFloatComplex *dT, magmaFloatComplex *ddA,
    float *dwork, magma_int_t *info);

magma_int_t magma_cgeqr2x3_gpu(
    magma_int_t *m, magma_int_t *n, magmaFloatComplex *dA,
    magma_int_t *ldda, magmaFloatComplex *dtau,
    magmaFloatComplex *dT, magmaFloatComplex *ddA,
    float *dwork, magma_int_t *info);

magma_int_t magma_cgeqr2x4_gpu(
    magma_int_t *m, magma_int_t *n, magmaFloatComplex *dA,
    magma_int_t *ldda, magmaFloatComplex *dtau,
    magmaFloatComplex *dT, magmaFloatComplex *ddA,
    float *dwork, magma_int_t *info, magma_queue_t stream);

magma_int_t magma_cgeqrf_gpu( magma_int_t m, magma_int_t n,
                              magmaFloatComplex *dA,  magma_int_t ldda,
                              magmaFloatComplex *tau, magmaFloatComplex *dT,
                              magma_int_t *info);

magma_int_t magma_cgeqrf2_gpu(magma_int_t m, magma_int_t n,
                              magmaFloatComplex *dA,  magma_int_t ldda,
                              magmaFloatComplex *tau, magma_int_t *info);

magma_int_t magma_cgeqrf2_mgpu(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                               magmaFloatComplex **dlA, magma_int_t ldda,
                               magmaFloatComplex *tau, magma_int_t *info );

magma_int_t magma_cgeqrf3_gpu(magma_int_t m, magma_int_t n,
                              magmaFloatComplex *dA,  magma_int_t ldda,
                              magmaFloatComplex *tau, magmaFloatComplex *dT,
                              magma_int_t *info);

magma_int_t magma_cgeqr2_gpu( magma_int_t m, magma_int_t n,
                              magmaFloatComplex *dA,  magma_int_t lda,
                              magmaFloatComplex *tau, float *work,
                              magma_int_t *info);

magma_int_t magma_cgeqrs_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              magmaFloatComplex *dA,     magma_int_t ldda,
                              magmaFloatComplex *tau,   magmaFloatComplex *dT,
                              magmaFloatComplex *dB,    magma_int_t lddb,
                              magmaFloatComplex *hwork, magma_int_t lhwork,
                              magma_int_t *info);

magma_int_t magma_cgeqrs3_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              magmaFloatComplex *dA,     magma_int_t ldda,
                              magmaFloatComplex *tau,   magmaFloatComplex *dT,
                              magmaFloatComplex *dB,    magma_int_t lddb,
                              magmaFloatComplex *hwork, magma_int_t lhwork,
                              magma_int_t *info);

magma_int_t magma_cgessm_gpu( magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib,
                              magma_int_t *ipiv,
                              magmaFloatComplex *dL1, magma_int_t lddl1,
                              magmaFloatComplex *dL,  magma_int_t lddl,
                              magmaFloatComplex *dA,  magma_int_t ldda,
                              magma_int_t *info);

magma_int_t magma_cgesv_gpu(  magma_int_t n, magma_int_t nrhs,
                              magmaFloatComplex *dA, magma_int_t ldda, magma_int_t *ipiv,
                              magmaFloatComplex *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_cgetf2_gpu( magma_int_t m, magma_int_t n,
                              magmaFloatComplex *dA, magma_int_t lda, magma_int_t *ipiv,
                              magma_int_t* info );

magma_int_t magma_cgetrf_incpiv_gpu( magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t ib,
                              magmaFloatComplex *hA, magma_int_t ldha, magmaFloatComplex *dA, magma_int_t ldda,
                              magmaFloatComplex *hL, magma_int_t ldhl, magmaFloatComplex *dL, magma_int_t lddl,
                              magma_int_t *ipiv,
                              magmaFloatComplex *dwork, magma_int_t lddwork,
                              magma_int_t *info);

magma_int_t magma_cgetrf_gpu( magma_int_t m, magma_int_t n,
                              magmaFloatComplex *dA, magma_int_t ldda,
                              magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_cgetrf_mgpu(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                              magmaFloatComplex **d_lA, magma_int_t ldda,
                              magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_cgetrf_m(magma_int_t num_gpus0, magma_int_t m, magma_int_t n, magmaFloatComplex *a, magma_int_t lda,
                           magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_cgetrf_piv(magma_int_t m, magma_int_t n, magma_int_t NB,
                             magmaFloatComplex *a, magma_int_t lda, magma_int_t *ipiv,
                             magma_int_t *info);

magma_int_t magma_cgetrf2_mgpu(magma_int_t num_gpus,
                               magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
                               magmaFloatComplex *d_lAT[], magma_int_t lddat, magma_int_t *ipiv,
                               magmaFloatComplex *d_lAP[], magmaFloatComplex *a, magma_int_t lda,
                               magma_queue_t streaml[][2], magma_int_t *info);

magma_int_t
      magma_cgetrf_nopiv_gpu( magma_int_t m, magma_int_t n,
                              magmaFloatComplex *dA, magma_int_t ldda,
                              magma_int_t *info);

magma_int_t magma_cgetri_gpu( magma_int_t n,
                              magmaFloatComplex *dA, magma_int_t ldda, magma_int_t *ipiv,
                              magmaFloatComplex *dwork, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_cgetrs_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                              magmaFloatComplex *dA, magma_int_t ldda, magma_int_t *ipiv,
                              magmaFloatComplex *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_clabrd_gpu( magma_int_t m, magma_int_t n, magma_int_t nb,
                              magmaFloatComplex *a, magma_int_t lda, magmaFloatComplex *da, magma_int_t ldda,
                              float *d, float *e, magmaFloatComplex *tauq, magmaFloatComplex *taup,
                              magmaFloatComplex *x, magma_int_t ldx, magmaFloatComplex *dx, magma_int_t lddx,
                              magmaFloatComplex *y, magma_int_t ldy, magmaFloatComplex *dy, magma_int_t lddy);

magma_int_t magma_claqps_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloatComplex *A,  magma_int_t lda,
    magma_int_t *jpvt, magmaFloatComplex *tau,
    float *vn1, float *vn2,
    magmaFloatComplex *auxv,
    magmaFloatComplex *dF, magma_int_t lddf);

magma_int_t magma_claqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloatComplex *A,  magma_int_t lda,
    magma_int_t *jpvt, magmaFloatComplex *tau,
    float *vn1, float *vn2,
    magmaFloatComplex *auxv,
    magmaFloatComplex *dF, magma_int_t lddf);

magma_int_t magma_claqps3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaFloatComplex *A,  magma_int_t lda,
    magma_int_t *jpvt, magmaFloatComplex *tau,
    float *vn1, float *vn2,
    magmaFloatComplex *auxv,
    magmaFloatComplex *dF, magma_int_t lddf);

magma_int_t magma_clarf_gpu(  magma_int_t m, magma_int_t n, magmaFloatComplex *v, magmaFloatComplex *tau,
                              magmaFloatComplex *c, magma_int_t ldc, float *xnorm);

magma_int_t magma_clarfb_gpu( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              const magmaFloatComplex *dv, magma_int_t ldv,
                              const magmaFloatComplex *dt, magma_int_t ldt,
                              magmaFloatComplex *dc,       magma_int_t ldc,
                              magmaFloatComplex *dwork,    magma_int_t ldwork );

magma_int_t magma_clarfb2_gpu(magma_int_t m, magma_int_t n, magma_int_t k,
                              const magmaFloatComplex *dV,    magma_int_t ldv,
                              const magmaFloatComplex *dT,    magma_int_t ldt,
                              magmaFloatComplex *dC,          magma_int_t ldc,
                              magmaFloatComplex *dwork,       magma_int_t ldwork );

magma_int_t magma_clarfb_gpu_gemm( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              const magmaFloatComplex *dv, magma_int_t ldv,
                              const magmaFloatComplex *dt, magma_int_t ldt,
                              magmaFloatComplex *dc,       magma_int_t ldc,
                              magmaFloatComplex *dwork,    magma_int_t ldwork,
                              magmaFloatComplex *dworkvt,  magma_int_t ldworkvt);

magma_int_t magma_clarfg_gpu( magma_int_t n, magmaFloatComplex *dx0, magmaFloatComplex *dx,
                              magmaFloatComplex *dtau, float *dxnorm, magmaFloatComplex *dAkk);

magma_int_t magma_cposv_gpu(  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                              magmaFloatComplex *dA, magma_int_t ldda,
                              magmaFloatComplex *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_cpotf2_gpu( magma_uplo_t uplo, magma_int_t n,
                              magmaFloatComplex *dA, magma_int_t lda,
                              magma_int_t *info );

magma_int_t magma_cpotrf_gpu( magma_uplo_t uplo,  magma_int_t n,
                              magmaFloatComplex *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_cpotrf_mgpu(magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n,
                              magmaFloatComplex **d_lA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_cpotrf3_mgpu(magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                               magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                               magmaFloatComplex *d_lA[],  magma_int_t ldda,
                               magmaFloatComplex *d_lP[],  magma_int_t lddp,
                               magmaFloatComplex *a,      magma_int_t lda,   magma_int_t h,
                               magma_queue_t stream[][3], magma_event_t event[][5],
                               magma_int_t *info );

magma_int_t magma_cpotri_gpu( magma_uplo_t uplo,  magma_int_t n,
                              magmaFloatComplex *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_clauum_gpu( magma_uplo_t uplo,  magma_int_t n,
                              magmaFloatComplex *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_ctrtri_gpu( magma_uplo_t uplo,  magma_diag_t diag, magma_int_t n,
                              magmaFloatComplex *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_chetrd_gpu( magma_uplo_t uplo, magma_int_t n,
                              magmaFloatComplex *da, magma_int_t ldda,
                              float *d, float *e, magmaFloatComplex *tau,
                              magmaFloatComplex *wa,  magma_int_t ldwa,
                              magmaFloatComplex *work, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_chetrd2_gpu(magma_uplo_t uplo, magma_int_t n,
                              magmaFloatComplex *da, magma_int_t ldda,
                              float *d, float *e, magmaFloatComplex *tau,
                              magmaFloatComplex *wa,  magma_int_t ldwa,
                              magmaFloatComplex *work, magma_int_t lwork,
                              magmaFloatComplex *dwork, magma_int_t ldwork,
                              magma_int_t *info);

float magma_clatrd_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo,
    magma_int_t n0, magma_int_t n, magma_int_t nb, magma_int_t nb0,
    magmaFloatComplex *a,  magma_int_t lda,
    float *e, magmaFloatComplex *tau,
    magmaFloatComplex *w,   magma_int_t ldw,
    magmaFloatComplex **da, magma_int_t ldda, magma_int_t offset,
    magmaFloatComplex **dw, magma_int_t lddw,
    magmaFloatComplex *dwork[MagmaMaxGPUs], magma_int_t ldwork,
    magma_int_t k,
    magmaFloatComplex  *dx[MagmaMaxGPUs], magmaFloatComplex *dy[MagmaMaxGPUs],
    magmaFloatComplex *work,
    magma_queue_t stream[][10],
    float *times );

magma_int_t magma_chetrd_mgpu(magma_int_t num_gpus, magma_int_t k, magma_uplo_t uplo, magma_int_t n,
                              magmaFloatComplex *a, magma_int_t lda,
                              float *d, float *e, magmaFloatComplex *tau,
                              magmaFloatComplex *work, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_chetrd_hb2st(magma_int_t threads, magma_uplo_t uplo,
                              magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
                              magmaFloatComplex *A, magma_int_t lda,
                              float *D, float *E,
                              magmaFloatComplex *V, magma_int_t ldv,
                              magmaFloatComplex *TAU, magma_int_t compT,
                              magmaFloatComplex *T, magma_int_t ldt);

magma_int_t magma_chetrd_he2hb(magma_uplo_t uplo, magma_int_t n, magma_int_t NB,
                              magmaFloatComplex *a, magma_int_t lda,
                              magmaFloatComplex *tau, magmaFloatComplex *work, magma_int_t lwork,
                              magmaFloatComplex *dT, magma_int_t threads,
                              magma_int_t *info);

magma_int_t magma_chetrd_he2hb_mgpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                              magmaFloatComplex *a, magma_int_t lda,
                              magmaFloatComplex *tau,
                              magmaFloatComplex *work, magma_int_t lwork,
                              magmaFloatComplex *dAmgpu[], magma_int_t ldda,
                              magmaFloatComplex *dTmgpu[], magma_int_t lddt,
                              magma_int_t ngpu, magma_int_t distblk,
                              magma_queue_t streams[][20], magma_int_t nstream,
                              magma_int_t threads, magma_int_t *info);

magma_int_t magma_chetrd_he2hb_mgpu_spec( magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                              magmaFloatComplex *a, magma_int_t lda,
                              magmaFloatComplex *tau,
                              magmaFloatComplex *work, magma_int_t lwork,
                              magmaFloatComplex *dAmgpu[], magma_int_t ldda,
                              magmaFloatComplex *dTmgpu[], magma_int_t lddt,
                              magma_int_t ngpu, magma_int_t distblk,
                              magma_queue_t streams[][20], magma_int_t nstream,
                              magma_int_t threads, magma_int_t *info);

magma_int_t magma_cpotrs_gpu( magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs,
                              magmaFloatComplex *dA, magma_int_t ldda,
                              magmaFloatComplex *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_cssssm_gpu( magma_storev_t storev, magma_int_t m1, magma_int_t n1,
                              magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib,
                              magmaFloatComplex *dA1, magma_int_t ldda1,
                              magmaFloatComplex *dA2, magma_int_t ldda2,
                              magmaFloatComplex *dL1, magma_int_t lddl1,
                              magmaFloatComplex *dL2, magma_int_t lddl2,
                              magma_int_t *IPIV, magma_int_t *info);

magma_int_t magma_ctstrf_gpu( magma_storev_t storev, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
                              magmaFloatComplex *hU, magma_int_t ldhu, magmaFloatComplex *dU, magma_int_t lddu,
                              magmaFloatComplex *hA, magma_int_t ldha, magmaFloatComplex *dA, magma_int_t ldda,
                              magmaFloatComplex *hL, magma_int_t ldhl, magmaFloatComplex *dL, magma_int_t lddl,
                              magma_int_t *ipiv,
                              magmaFloatComplex *hwork, magma_int_t ldhwork,
                              magmaFloatComplex *dwork, magma_int_t lddwork,
                              magma_int_t *info);

magma_int_t magma_cungqr_gpu( magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaFloatComplex *da, magma_int_t ldda,
                              magmaFloatComplex *tau, magmaFloatComplex *dwork,
                              magma_int_t nb, magma_int_t *info );

magma_int_t magma_cunmql2_gpu(magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaFloatComplex *da, magma_int_t ldda,
                              magmaFloatComplex *tau,
                              magmaFloatComplex *dc, magma_int_t lddc,
                              magmaFloatComplex *wa, magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_cunmqr_gpu( magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaFloatComplex *dA,    magma_int_t ldda, magmaFloatComplex *tau,
                              magmaFloatComplex *dC,    magma_int_t lddc,
                              magmaFloatComplex *hwork, magma_int_t lwork,
                              magmaFloatComplex *dT,    magma_int_t nb, magma_int_t *info);

magma_int_t magma_cunmqr2_gpu(magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaFloatComplex *da,   magma_int_t ldda,
                              magmaFloatComplex *tau,
                              magmaFloatComplex *dc,    magma_int_t lddc,
                              magmaFloatComplex *wa,    magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_cunmtr_gpu( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                              magma_int_t m, magma_int_t n,
                              magmaFloatComplex *da,    magma_int_t ldda,
                              magmaFloatComplex *tau,
                              magmaFloatComplex *dc,    magma_int_t lddc,
                              magmaFloatComplex *wa,    magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_cgeqp3_gpu( magma_int_t m, magma_int_t n,
                              magmaFloatComplex *A, magma_int_t lda,
                              magma_int_t *jpvt, magmaFloatComplex *tau,
                              magmaFloatComplex *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              float *rwork,
                              #endif
                              magma_int_t *info );

magma_int_t magma_cheevd_gpu( magma_vec_t jobz, magma_uplo_t uplo,
                              magma_int_t n,
                              magmaFloatComplex *da, magma_int_t ldda,
                              float *w,
                              magmaFloatComplex *wa,  magma_int_t ldwa,
                              magmaFloatComplex *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              float *rwork, magma_int_t lrwork,
                              #endif
                              magma_int_t *iwork, magma_int_t liwork,
                              magma_int_t *info);

magma_int_t magma_cheevdx_gpu(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                              magma_int_t n, magmaFloatComplex *da,
                              magma_int_t ldda, float vl, float vu,
                              magma_int_t il, magma_int_t iu,
                              magma_int_t *m, float *w,
                              magmaFloatComplex *wa,  magma_int_t ldwa,
                              magmaFloatComplex *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              float *rwork, magma_int_t lrwork,
                              #endif
                              magma_int_t *iwork, magma_int_t liwork,
                              magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t magma_cheevx_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                              magmaFloatComplex *da, magma_int_t ldda, float vl,
                              float vu, magma_int_t il, magma_int_t iu,
                              float abstol, magma_int_t *m,
                              float *w, magmaFloatComplex *dz, magma_int_t lddz,
                              magmaFloatComplex *wa, magma_int_t ldwa,
                              magmaFloatComplex *wz, magma_int_t ldwz,
                              magmaFloatComplex *work, magma_int_t lwork,
                              float *rwork, magma_int_t *iwork,
                              magma_int_t *ifail, magma_int_t *info);

magma_int_t magma_cheevr_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                              magmaFloatComplex *da, magma_int_t ldda, float vl, float vu,
                              magma_int_t il, magma_int_t iu, float abstol, magma_int_t *m,
                              float *w, magmaFloatComplex *dz, magma_int_t lddz,
                              magma_int_t *isuppz,
                              magmaFloatComplex *wa, magma_int_t ldwa,
                              magmaFloatComplex *wz, magma_int_t ldwz,
                              magmaFloatComplex *work, magma_int_t lwork,
                              float *rwork, magma_int_t lrwork, magma_int_t *iwork,
                              magma_int_t liwork, magma_int_t *info);
#endif  // COMPLEX

magma_int_t magma_chegst_gpu(magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                             magmaFloatComplex *da, magma_int_t ldda,
                             magmaFloatComplex *db, magma_int_t lddb, magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA utility function definitions
*/

void magma_cprint    ( magma_int_t m, magma_int_t n, const magmaFloatComplex  *A, magma_int_t lda  );
void magma_cprint_gpu( magma_int_t m, magma_int_t n, const magmaFloatComplex *dA, magma_int_t ldda );

void cpanel_to_q( magma_uplo_t uplo, magma_int_t ib, magmaFloatComplex *A, magma_int_t lda, magmaFloatComplex *work );
void cq_to_panel( magma_uplo_t uplo, magma_int_t ib, magmaFloatComplex *A, magma_int_t lda, magmaFloatComplex *work );

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif /* MAGMA_C_H */
