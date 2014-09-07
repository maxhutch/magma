/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
*/

#ifndef MAGMA_Z_H
#define MAGMA_Z_H

#include "magma_types.h"
#include "magma_zgehrd_m.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA Auxiliary functions to get the NB used
*/
magma_int_t magma_get_zpotrf_nb( magma_int_t m );
magma_int_t magma_get_zgetrf_nb( magma_int_t m );
magma_int_t magma_get_zgetri_nb( magma_int_t m );
magma_int_t magma_get_zgeqp3_nb( magma_int_t m );
magma_int_t magma_get_zgeqrf_nb( magma_int_t m );
magma_int_t magma_get_zgeqlf_nb( magma_int_t m );
magma_int_t magma_get_zgehrd_nb( magma_int_t m );
magma_int_t magma_get_zhetrd_nb( magma_int_t m );
magma_int_t magma_get_zgelqf_nb( magma_int_t m );
magma_int_t magma_get_zgebrd_nb( magma_int_t m );
magma_int_t magma_get_zhegst_nb( magma_int_t m );
magma_int_t magma_get_zgesvd_nb( magma_int_t m );
magma_int_t magma_get_zhegst_nb_m( magma_int_t m );
magma_int_t magma_get_zbulge_nb( magma_int_t m, magma_int_t nbthreads );
magma_int_t magma_get_zbulge_nb_mgpu( magma_int_t m );
magma_int_t magma_zbulge_get_Vblksiz( magma_int_t m, magma_int_t nb, magma_int_t nbthreads );
magma_int_t magma_get_zbulge_gcperf();

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU
*/

#ifdef REAL
// only applicable to real [sd] precisions
void magma_dmove_eig(magma_range_t range, magma_int_t n, double *w, magma_int_t *il,
                          magma_int_t *iu, double vl, double vu, magma_int_t *m);
#endif

magma_int_t magma_zgebrd( magma_int_t m, magma_int_t n, magmaDoubleComplex *A,
                          magma_int_t lda, double *d, double *e,
                          magmaDoubleComplex *tauq,  magmaDoubleComplex *taup,
                          magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_zgehrd2(magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau,
                          magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_zgehrd( magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *tau,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          magmaDoubleComplex *dT, magma_int_t *info);

magma_int_t magma_zgelqf( magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *A,    magma_int_t lda,   magmaDoubleComplex *tau,
                          magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_zgeqlf( magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *A,    magma_int_t lda,   magmaDoubleComplex *tau,
                          magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_zgeqrf( magma_int_t m, magma_int_t n, magmaDoubleComplex *A,
                          magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work,
                          magma_int_t lwork, magma_int_t *info);

magma_int_t magma_zgeqrf4(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *a,    magma_int_t lda, magmaDoubleComplex *tau,
                          magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info );

magma_int_t magma_zgeqrf_ooc( magma_int_t m, magma_int_t n, magmaDoubleComplex *A,
                          magma_int_t lda, magmaDoubleComplex *tau, magmaDoubleComplex *work,
                          magma_int_t lwork, magma_int_t *info);

magma_int_t magma_zgesv ( magma_int_t n, magma_int_t nrhs,
                          magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv,
                          magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info);

magma_int_t
      magma_zgetf2_nopiv( magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);

magma_int_t magma_zgetrf( magma_int_t m, magma_int_t n, magmaDoubleComplex *A,
                          magma_int_t lda, magma_int_t *ipiv,
                          magma_int_t *info);

magma_int_t magma_zgetrf2(magma_int_t m, magma_int_t n, magmaDoubleComplex *a,
                          magma_int_t lda, magma_int_t *ipiv, magma_int_t *info);

magma_int_t
      magma_zgetrf_nopiv( magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *A, magma_int_t lda, magma_int_t *info);

#ifdef REAL
magma_int_t magma_dlaln2( magma_int_t trans, magma_int_t na, magma_int_t nw,
                          double smin, double ca, const double *a, magma_int_t lda,
                          double d1, double d2, const double *b, magma_int_t ldb,
                          double wr, double wi, double *x, magma_int_t ldx,
                          double *scale, double *xnorm, magma_int_t *info );
#endif

magma_int_t magma_zlaqps( magma_int_t m, magma_int_t n, magma_int_t offset,
                          magma_int_t nb, magma_int_t *kb,
                          magmaDoubleComplex *A,  magma_int_t lda,
                          magmaDoubleComplex *dA, magma_int_t ldda,
                          magma_int_t *jpvt, magmaDoubleComplex *tau, double *vn1, double *vn2,
                          magmaDoubleComplex *auxv,
                          magmaDoubleComplex *F,  magma_int_t ldf,
                          magmaDoubleComplex *dF, magma_int_t lddf );

#ifdef REAL
magma_int_t magma_zlaqtrsd( magma_trans_t trans,
                            magma_int_t n, const double *T, magma_int_t ldt,
                            double *x, magma_int_t ldx,
                            const double *cnorm, magma_int_t *info );
#endif

magma_int_t magma_zlatrd( magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magmaDoubleComplex *a,
                          magma_int_t lda, double *e, magmaDoubleComplex *tau,
                          magmaDoubleComplex *w, magma_int_t ldw,
                          magmaDoubleComplex *da, magma_int_t ldda,
                          magmaDoubleComplex *dw, magma_int_t lddw);

magma_int_t magma_zlatrd2(magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                          magmaDoubleComplex *a,  magma_int_t lda,
                          double *e, magmaDoubleComplex *tau,
                          magmaDoubleComplex *w,  magma_int_t ldw,
                          magmaDoubleComplex *da, magma_int_t ldda,
                          magmaDoubleComplex *dw, magma_int_t lddw,
                          magmaDoubleComplex *dwork, magma_int_t ldwork);

#ifdef COMPLEX
magma_int_t magma_zlatrsd(magma_uplo_t uplo, magma_trans_t trans,
                          magma_diag_t diag, magma_bool_t normin,
                          magma_int_t n, const magmaDoubleComplex *A, magma_int_t lda,
                          magmaDoubleComplex lambda,
                          magmaDoubleComplex *x,
                          double *scale, double *cnorm, magma_int_t *info);
#endif

magma_int_t magma_zlahr2( magma_int_t m, magma_int_t n, magma_int_t nb,
                          magmaDoubleComplex *dA, magma_int_t ldda,
                          magmaDoubleComplex *dV, magma_int_t lddv,
                          magmaDoubleComplex *A,  magma_int_t lda,
                          magmaDoubleComplex *tau,
                          magmaDoubleComplex *T,  magma_int_t ldt,
                          magmaDoubleComplex *Y,  magma_int_t ldy);

magma_int_t magma_zlahru( magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
                          magmaDoubleComplex *A,  magma_int_t lda,
                          magmaDoubleComplex *dA, magma_int_t ldda,
                          magmaDoubleComplex *dY, magma_int_t lddy,
                          magmaDoubleComplex *dV, magma_int_t lddv,
                          magmaDoubleComplex *dT,
                          magmaDoubleComplex *dwork);

magma_int_t magma_zposv ( magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                          magmaDoubleComplex *A, magma_int_t lda,
                          magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info);

magma_int_t magma_zpotrf( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_zpotri( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_zlauum( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_ztrevc3(magma_side_t side, magma_vec_t howmny,
                          magma_int_t *select, magma_int_t n,
                          magmaDoubleComplex *T,  magma_int_t ldt,
                          magmaDoubleComplex *VL, magma_int_t ldvl,
                          magmaDoubleComplex *VR, magma_int_t ldvr,
                          magma_int_t mm, magma_int_t *mout,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *info );
                          
magma_int_t magma_ztrevc3_mt( magma_side_t side, magma_vec_t howmny,
                          magma_int_t *select, magma_int_t n,
                          magmaDoubleComplex *T,  magma_int_t ldt,
                          magmaDoubleComplex *VL, magma_int_t ldvl,
                          magmaDoubleComplex *VR, magma_int_t ldvr,
                          magma_int_t mm, magma_int_t *mout,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *info );

magma_int_t magma_ztrtri( magma_uplo_t uplo, magma_diag_t diag, magma_int_t n, magmaDoubleComplex *A,
                          magma_int_t lda, magma_int_t *info);

magma_int_t magma_zhetrd( magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex *A,
                          magma_int_t lda, double *d, double *e,
                          magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_zungqr( magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *tau, magmaDoubleComplex *dT,
                          magma_int_t nb, magma_int_t *info );

magma_int_t magma_zungqr2(magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *tau, magma_int_t *info );

magma_int_t magma_zunmbr( magma_vect_t vect, magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaDoubleComplex *A, magma_int_t lda,
                          magmaDoubleComplex *tau,
                          magmaDoubleComplex *C, magma_int_t ldc,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_zunmlq( magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaDoubleComplex *A, magma_int_t lda,
                          magmaDoubleComplex *tau,
                          magmaDoubleComplex *C, magma_int_t ldc,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          magma_int_t *info);

// not yet implemented
//magma_int_t magma_zunmrq( magma_side_t side, magma_trans_t trans,
//                          magma_int_t m, magma_int_t n, magma_int_t k,
//                          magmaDoubleComplex *A, magma_int_t lda,
//                          magmaDoubleComplex *tau,
//                          magmaDoubleComplex *C, magma_int_t ldc,
//                          magmaDoubleComplex *work, magma_int_t lwork,
//                          magma_int_t *info);

magma_int_t magma_zunmql( magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *tau,
                          magmaDoubleComplex *c, magma_int_t ldc,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_zunmqr( magma_side_t side, magma_trans_t trans,
                          magma_int_t m, magma_int_t n, magma_int_t k,
                          magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *tau,
                          magmaDoubleComplex *c, magma_int_t ldc,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_zunmtr( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                          magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *a,    magma_int_t lda,
                          magmaDoubleComplex *tau,
                          magmaDoubleComplex *c,    magma_int_t ldc,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          magma_int_t *info);

magma_int_t magma_zunghr( magma_int_t n, magma_int_t ilo, magma_int_t ihi,
                          magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *tau,
                          magmaDoubleComplex *dT, magma_int_t nb,
                          magma_int_t *info);

magma_int_t  magma_zgeev( magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda,
                          #ifdef COMPLEX
                          magmaDoubleComplex *w,
                          #else
                          double *wr, double *wi,
                          #endif
                          magmaDoubleComplex *vl, magma_int_t ldvl,
                          magmaDoubleComplex *vr, magma_int_t ldvr,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *info);

magma_int_t magma_zgeqp3( magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda,
                          magma_int_t *jpvt, magmaDoubleComplex *tau,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *info);

magma_int_t magma_zgesdd( magma_vec_t jobz, magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *A, magma_int_t lda,
                          double *s,
                          magmaDoubleComplex *U, magma_int_t ldu,
                          magmaDoubleComplex *VT, magma_int_t ldvt,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *iwork, magma_int_t *info);

magma_int_t magma_zgesvd( magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
                          magmaDoubleComplex *a,    magma_int_t lda, double *s,
                          magmaDoubleComplex *u,    magma_int_t ldu,
                          magmaDoubleComplex *vt,   magma_int_t ldvt,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *info );

magma_int_t magma_zheevd( magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda, double *w,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_zheevdx(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, double *w,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_zheevdx_2stage(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, double *w,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork,
                          magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t magma_zheevx( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda, double vl, double vu,
                          magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
                          double *w, magmaDoubleComplex *z, magma_int_t ldz,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          double *rwork, magma_int_t *iwork,
                          magma_int_t *ifail, magma_int_t *info);

// no real [sd] precisions available
magma_int_t magma_zheevr( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda, double vl, double vu,
                          magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
                          double *w, magmaDoubleComplex *z, magma_int_t ldz,
                          magma_int_t *isuppz,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          double *rwork, magma_int_t lrwork, magma_int_t *iwork,
                          magma_int_t liwork, magma_int_t *info);
#endif  // COMPLEX

magma_int_t magma_zhegvd( magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *b, magma_int_t ldb,
                          double *w, magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);
                          
magma_int_t magma_zhegvdx(magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *b, magma_int_t ldb,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, double *w,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork, magma_int_t *info);

magma_int_t magma_zhegvdx_2stage(magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *b, magma_int_t ldb,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          magma_int_t *m, double *w,
                          magmaDoubleComplex *work, magma_int_t lwork,
                          #ifdef COMPLEX
                          double *rwork, magma_int_t lrwork,
                          #endif
                          magma_int_t *iwork, magma_int_t liwork,
                          magma_int_t *info);

magma_int_t magma_zhegvx( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *b, magma_int_t ldb,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          double abstol, magma_int_t *m, double *w,
                          magmaDoubleComplex *z, magma_int_t ldz,
                          magmaDoubleComplex *work, magma_int_t lwork, double *rwork,
                          magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info);

magma_int_t magma_zhegvr( magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                          magma_int_t n, magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *b, magma_int_t ldb,
                          double vl, double vu, magma_int_t il, magma_int_t iu,
                          double abstol, magma_int_t *m, double *w,
                          magmaDoubleComplex *z, magma_int_t ldz,
                          magma_int_t *isuppz, magmaDoubleComplex *work, magma_int_t lwork,
                          double *rwork, magma_int_t lrwork, magma_int_t *iwork,
                          magma_int_t liwork, magma_int_t *info);

magma_int_t magma_zstedx( magma_range_t range, magma_int_t n, double vl, double vu,
                          magma_int_t il, magma_int_t iu, double *D, double *E,
                          magmaDoubleComplex *Z, magma_int_t ldz,
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

magma_int_t magma_zhegst( magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                          magmaDoubleComplex *a, magma_int_t lda,
                          magmaDoubleComplex *b, magma_int_t ldb, magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on CPU / Multi-GPU
*/
magma_int_t magma_zlahr2_m(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *T, magma_int_t ldt,
    magmaDoubleComplex *Y, magma_int_t ldy,
    struct zgehrd_data *data );

magma_int_t magma_zlahru_m(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    magmaDoubleComplex *A, magma_int_t lda,
    struct zgehrd_data *data );

magma_int_t magma_zgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    #ifdef COMPLEX
    magmaDoubleComplex *w,
    #else
    double *wr, double *wi,
    #endif
    magmaDoubleComplex *vl, magma_int_t ldvl,
    magmaDoubleComplex *vr, magma_int_t ldvr,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info );

magma_int_t magma_zgehrd_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magmaDoubleComplex *T,
    magma_int_t *info );

magma_int_t magma_zunghr_m(
    magma_int_t n, magma_int_t ilo, magma_int_t ihi,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *T, magma_int_t nb,
    magma_int_t *info );

magma_int_t magma_zungqr_m(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *T, magma_int_t nb,
    magma_int_t *info );

magma_int_t magma_zpotrf_m( magma_int_t num_gpus,
                            magma_uplo_t uplo, magma_int_t n,
                            magmaDoubleComplex *A, magma_int_t lda,
                            magma_int_t *info);

magma_int_t magma_zstedx_m( magma_int_t nrgpu,
                            magma_range_t range, magma_int_t n, double vl, double vu,
                            magma_int_t il, magma_int_t iu, double *D, double *E,
                            magmaDoubleComplex *Z, magma_int_t ldz,
                            double *rwork, magma_int_t ldrwork, magma_int_t *iwork,
                            magma_int_t liwork, magma_int_t *info);

magma_int_t magma_ztrsm_m ( magma_int_t nrgpu,
                            magma_side_t side, magma_uplo_t uplo, magma_trans_t transa, magma_diag_t diag,
                            magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
                            magmaDoubleComplex *a, magma_int_t lda,
                            magmaDoubleComplex *b, magma_int_t ldb);

magma_int_t magma_zunmqr_m( magma_int_t nrgpu, magma_side_t side, magma_trans_t trans,
                            magma_int_t m, magma_int_t n, magma_int_t k,
                            magmaDoubleComplex *a,    magma_int_t lda,
                            magmaDoubleComplex *tau,
                            magmaDoubleComplex *c,    magma_int_t ldc,
                            magmaDoubleComplex *work, magma_int_t lwork,
                            magma_int_t *info);

magma_int_t magma_zunmtr_m( magma_int_t nrgpu,
                            magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                            magma_int_t m, magma_int_t n,
                            magmaDoubleComplex *a,    magma_int_t lda,
                            magmaDoubleComplex *tau,
                            magmaDoubleComplex *c,    magma_int_t ldc,
                            magmaDoubleComplex *work, magma_int_t lwork,
                            magma_int_t *info);

magma_int_t magma_zhegst_m( magma_int_t nrgpu,
                            magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                            magmaDoubleComplex *a, magma_int_t lda,
                            magmaDoubleComplex *b, magma_int_t ldb,
                            magma_int_t *info);

magma_int_t magma_zheevd_m( magma_int_t nrgpu,
                            magma_vec_t jobz, magma_uplo_t uplo,
                            magma_int_t n,
                            magmaDoubleComplex *a, magma_int_t lda,
                            double *w,
                            magmaDoubleComplex *work, magma_int_t lwork,
                            #ifdef COMPLEX
                            double *rwork, magma_int_t lrwork,
                            #endif
                            magma_int_t *iwork, magma_int_t liwork,
                            magma_int_t *info);

magma_int_t magma_zhegvd_m( magma_int_t nrgpu,
                            magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo,
                            magma_int_t n,
                            magmaDoubleComplex *a, magma_int_t lda,
                            magmaDoubleComplex *b, magma_int_t ldb,
                            double *w,
                            magmaDoubleComplex *work, magma_int_t lwork,
                            #ifdef COMPLEX
                            double *rwork, magma_int_t lrwork,
                            #endif
                            magma_int_t *iwork, magma_int_t liwork,
                            magma_int_t *info);

magma_int_t magma_zheevdx_m( magma_int_t nrgpu,
                             magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                             magma_int_t n,
                             magmaDoubleComplex *a, magma_int_t lda,
                             double vl, double vu, magma_int_t il, magma_int_t iu,
                             magma_int_t *m, double *w,
                             magmaDoubleComplex *work, magma_int_t lwork,
                             #ifdef COMPLEX
                             double *rwork, magma_int_t lrwork,
                             #endif
                             magma_int_t *iwork, magma_int_t liwork,
                             magma_int_t *info);

magma_int_t magma_zhegvdx_m( magma_int_t nrgpu,
                             magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                             magma_int_t n,
                             magmaDoubleComplex *a, magma_int_t lda,
                             magmaDoubleComplex *b, magma_int_t ldb,
                             double vl, double vu, magma_int_t il, magma_int_t iu,
                             magma_int_t *m, double *w,
                             magmaDoubleComplex *work, magma_int_t lwork,
                             #ifdef COMPLEX
                             double *rwork, magma_int_t lrwork,
                             #endif
                             magma_int_t *iwork, magma_int_t liwork,
                             magma_int_t *info);

magma_int_t magma_zheevdx_2stage_m( magma_int_t nrgpu,
                                    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                                    magma_int_t n,
                                    magmaDoubleComplex *a, magma_int_t lda,
                                    double vl, double vu, magma_int_t il, magma_int_t iu,
                                    magma_int_t *m, double *w,
                                    magmaDoubleComplex *work, magma_int_t lwork,
                                    #ifdef COMPLEX
                                    double *rwork, magma_int_t lrwork,
                                    #endif
                                    magma_int_t *iwork, magma_int_t liwork,
                                    magma_int_t *info);

magma_int_t magma_zhegvdx_2stage_m( magma_int_t nrgpu,
                                    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                                    magma_int_t n,
                                    magmaDoubleComplex *a, magma_int_t lda,
                                    magmaDoubleComplex *b, magma_int_t ldb,
                                    double vl, double vu, magma_int_t il, magma_int_t iu,
                                    magma_int_t *m, double *w,
                                    magmaDoubleComplex *work, magma_int_t lwork,
                                    #ifdef COMPLEX
                                    double *rwork, magma_int_t lrwork,
                                    #endif
                                    magma_int_t *iwork, magma_int_t liwork,
                                    magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA function definitions / Data on GPU
*/
magma_int_t magma_zgegqr_gpu( magma_int_t ikind, magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA,   magma_int_t ldda,
                              magmaDoubleComplex *dwork, magmaDoubleComplex *work,
                              magma_int_t *info );

magma_int_t magma_zgels_gpu(  magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              magmaDoubleComplex *dA,    magma_int_t ldda,
                              magmaDoubleComplex *dB,    magma_int_t lddb,
                              magmaDoubleComplex *hwork, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_zgels3_gpu( magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              magmaDoubleComplex *dA,    magma_int_t ldda,
                              magmaDoubleComplex *dB,    magma_int_t lddb,
                              magmaDoubleComplex *hwork, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_zgelqf_gpu( magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA,    magma_int_t ldda,   magmaDoubleComplex *tau,
                              magmaDoubleComplex *work, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_zgeqr2x_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex *dtau,
    magmaDoubleComplex *dT, magmaDoubleComplex *ddA,
    double *dwork, magma_int_t *info);

magma_int_t magma_zgeqr2x2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex *dtau,
    magmaDoubleComplex *dT, magmaDoubleComplex *ddA,
    double *dwork, magma_int_t *info);

magma_int_t magma_zgeqr2x3_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex *dtau,
    magmaDoubleComplex *dT, magmaDoubleComplex *ddA,
    double *dwork, magma_int_t *info);

magma_int_t magma_zgeqr2x4_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magmaDoubleComplex *dtau,
    magmaDoubleComplex *dT, magmaDoubleComplex *ddA,
    double *dwork, magma_int_t *info, magma_queue_t stream);

magma_int_t magma_zgeqrf_gpu( magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA,  magma_int_t ldda,
                              magmaDoubleComplex *tau, magmaDoubleComplex *dT,
                              magma_int_t *info);

magma_int_t magma_zgeqrf2_gpu(magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA,  magma_int_t ldda,
                              magmaDoubleComplex *tau, magma_int_t *info);

magma_int_t magma_zgeqrf2_mgpu(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                               magmaDoubleComplex **dlA, magma_int_t ldda,
                               magmaDoubleComplex *tau, magma_int_t *info );

magma_int_t magma_zgeqrf3_gpu(magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA,  magma_int_t ldda,
                              magmaDoubleComplex *tau, magmaDoubleComplex *dT,
                              magma_int_t *info);

magma_int_t magma_zgeqr2_gpu( magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA,  magma_int_t lda,
                              magmaDoubleComplex *tau, double *work,
                              magma_int_t *info);

magma_int_t magma_zgeqrs_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              magmaDoubleComplex *dA,     magma_int_t ldda,
                              magmaDoubleComplex *tau,   magmaDoubleComplex *dT,
                              magmaDoubleComplex *dB,    magma_int_t lddb,
                              magmaDoubleComplex *hwork, magma_int_t lhwork,
                              magma_int_t *info);

magma_int_t magma_zgeqrs3_gpu( magma_int_t m, magma_int_t n, magma_int_t nrhs,
                              magmaDoubleComplex *dA,     magma_int_t ldda,
                              magmaDoubleComplex *tau,   magmaDoubleComplex *dT,
                              magmaDoubleComplex *dB,    magma_int_t lddb,
                              magmaDoubleComplex *hwork, magma_int_t lhwork,
                              magma_int_t *info);

magma_int_t magma_zgessm_gpu( magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t ib,
                              magma_int_t *ipiv,
                              magmaDoubleComplex *dL1, magma_int_t lddl1,
                              magmaDoubleComplex *dL,  magma_int_t lddl,
                              magmaDoubleComplex *dA,  magma_int_t ldda,
                              magma_int_t *info);

magma_int_t magma_zgesv_gpu(  magma_int_t n, magma_int_t nrhs,
                              magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *ipiv,
                              magmaDoubleComplex *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_zgetf2_gpu( magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t lda, magma_int_t *ipiv,
                              magma_int_t* info );

magma_int_t magma_zgetrf_incpiv_gpu( magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t ib,
                              magmaDoubleComplex *hA, magma_int_t ldha, magmaDoubleComplex *dA, magma_int_t ldda,
                              magmaDoubleComplex *hL, magma_int_t ldhl, magmaDoubleComplex *dL, magma_int_t lddl,
                              magma_int_t *ipiv,
                              magmaDoubleComplex *dwork, magma_int_t lddwork,
                              magma_int_t *info);

magma_int_t magma_zgetrf_gpu( magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t ldda,
                              magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_zgetrf_mgpu(magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                              magmaDoubleComplex **d_lA, magma_int_t ldda,
                              magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_zgetrf_m(magma_int_t num_gpus0, magma_int_t m, magma_int_t n, magmaDoubleComplex *a, magma_int_t lda,
                           magma_int_t *ipiv, magma_int_t *info);

magma_int_t magma_zgetrf_piv(magma_int_t m, magma_int_t n, magma_int_t NB,
                             magmaDoubleComplex *a, magma_int_t lda, magma_int_t *ipiv,
                             magma_int_t *info);

magma_int_t magma_zgetrf2_mgpu(magma_int_t num_gpus,
                               magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
                               magmaDoubleComplex *d_lAT[], magma_int_t lddat, magma_int_t *ipiv,
                               magmaDoubleComplex *d_lAP[], magmaDoubleComplex *a, magma_int_t lda,
                               magma_queue_t streaml[][2], magma_int_t *info);

magma_int_t magma_zgetrf_nopiv_gpu( magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t ldda,
                              magma_int_t *info);

magma_int_t magma_zgetri_gpu( magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *ipiv,
                              magmaDoubleComplex *dwork, magma_int_t lwork, magma_int_t *info);

magma_int_t magma_zgetrs_gpu( magma_trans_t trans, magma_int_t n, magma_int_t nrhs,
                              magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *ipiv,
                              magmaDoubleComplex *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_zlabrd_gpu( magma_int_t m, magma_int_t n, magma_int_t nb,
                              magmaDoubleComplex *a, magma_int_t lda, magmaDoubleComplex *da, magma_int_t ldda,
                              double *d, double *e, magmaDoubleComplex *tauq, magmaDoubleComplex *taup,
                              magmaDoubleComplex *x, magma_int_t ldx, magmaDoubleComplex *dx, magma_int_t lddx,
                              magmaDoubleComplex *y, magma_int_t ldy, magmaDoubleComplex *dy, magma_int_t lddy);

magma_int_t magma_zlaqps_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex *A,  magma_int_t lda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    double *vn1, double *vn2,
    magmaDoubleComplex *auxv,
    magmaDoubleComplex *dF, magma_int_t lddf);

magma_int_t magma_zlaqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex *A,  magma_int_t lda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    double *vn1, double *vn2,
    magmaDoubleComplex *auxv,
    magmaDoubleComplex *dF, magma_int_t lddf);

magma_int_t magma_zlaqps3_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex *A,  magma_int_t lda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    double *vn1, double *vn2,
    magmaDoubleComplex *auxv,
    magmaDoubleComplex *dF, magma_int_t lddf);

magma_int_t magma_zlarf_gpu(
    magma_int_t m,  magma_int_t n,
    const magmaDoubleComplex *dv, const magmaDoubleComplex *dtau,
    magmaDoubleComplex *dc,  magma_int_t lddc);

magma_int_t magma_zlarfb_gpu( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              const magmaDoubleComplex *dv, magma_int_t ldv,
                              const magmaDoubleComplex *dt, magma_int_t ldt,
                              magmaDoubleComplex *dc,       magma_int_t ldc,
                              magmaDoubleComplex *dwork,    magma_int_t ldwork );

magma_int_t magma_zlarfb2_gpu(magma_int_t m, magma_int_t n, magma_int_t k,
                              const magmaDoubleComplex *dV,    magma_int_t ldv,
                              const magmaDoubleComplex *dT,    magma_int_t ldt,
                              magmaDoubleComplex *dC,          magma_int_t ldc,
                              magmaDoubleComplex *dwork,       magma_int_t ldwork );

magma_int_t magma_zlarfb_gpu_gemm( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              const magmaDoubleComplex *dv, magma_int_t ldv,
                              const magmaDoubleComplex *dt, magma_int_t ldt,
                              magmaDoubleComplex *dc,       magma_int_t ldc,
                              magmaDoubleComplex *dwork,    magma_int_t ldwork,
                              magmaDoubleComplex *dworkvt,  magma_int_t ldworkvt);

magma_int_t magma_zposv_gpu(  magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                              magmaDoubleComplex *dA, magma_int_t ldda,
                              magmaDoubleComplex *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_zpotf2_gpu( magma_uplo_t uplo, magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t lda,
                              magma_int_t *info );

magma_int_t magma_zpotrf_gpu( magma_uplo_t uplo,  magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_zpotrf_mgpu(magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n,
                              magmaDoubleComplex **d_lA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_zpotrf3_mgpu(magma_int_t num_gpus, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
                               magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                               magmaDoubleComplex *d_lA[],  magma_int_t ldda,
                               magmaDoubleComplex *d_lP[],  magma_int_t lddp,
                               magmaDoubleComplex *a,      magma_int_t lda,   magma_int_t h,
                               magma_queue_t stream[][3], magma_event_t event[][5],
                               magma_int_t *info );

magma_int_t magma_zpotri_gpu( magma_uplo_t uplo,  magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_zlauum_gpu( magma_uplo_t uplo,  magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_ztrtri_gpu( magma_uplo_t uplo,  magma_diag_t diag, magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *info);

magma_int_t magma_zhetrd_gpu( magma_uplo_t uplo, magma_int_t n,
                              magmaDoubleComplex *da, magma_int_t ldda,
                              double *d, double *e, magmaDoubleComplex *tau,
                              magmaDoubleComplex *wa,  magma_int_t ldwa,
                              magmaDoubleComplex *work, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_zhetrd2_gpu(magma_uplo_t uplo, magma_int_t n,
                              magmaDoubleComplex *da, magma_int_t ldda,
                              double *d, double *e, magmaDoubleComplex *tau,
                              magmaDoubleComplex *wa,  magma_int_t ldwa,
                              magmaDoubleComplex *work, magma_int_t lwork,
                              magmaDoubleComplex *dwork, magma_int_t ldwork,
                              magma_int_t *info);

magma_int_t magma_zlatrd_mgpu(
    magma_int_t num_gpus, magma_uplo_t uplo,
    magma_int_t n0, magma_int_t n, magma_int_t nb, magma_int_t nb0,
    magmaDoubleComplex *a,  magma_int_t lda,
    double *e, magmaDoubleComplex *tau,
    magmaDoubleComplex *w,   magma_int_t ldw,
    magmaDoubleComplex **da, magma_int_t ldda, magma_int_t offset,
    magmaDoubleComplex **dw, magma_int_t lddw,
    magmaDoubleComplex *dwork[MagmaMaxGPUs], magma_int_t ldwork,
    magma_int_t k,
    magmaDoubleComplex  *dx[MagmaMaxGPUs], magmaDoubleComplex *dy[MagmaMaxGPUs],
    magmaDoubleComplex *work,
    magma_queue_t stream[][10],
    double *times );

magma_int_t magma_zhetrd_mgpu(magma_int_t num_gpus, magma_int_t k, magma_uplo_t uplo, magma_int_t n,
                              magmaDoubleComplex *a, magma_int_t lda,
                              double *d, double *e, magmaDoubleComplex *tau,
                              magmaDoubleComplex *work, magma_int_t lwork,
                              magma_int_t *info);

magma_int_t magma_zhetrd_hb2st(magma_uplo_t uplo,
                              magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
                              magmaDoubleComplex *A, magma_int_t lda,
                              double *D, double *E,
                              magmaDoubleComplex *V, magma_int_t ldv,
                              magmaDoubleComplex *TAU, magma_int_t compT,
                              magmaDoubleComplex *T, magma_int_t ldt);

magma_int_t magma_zhetrd_he2hb(magma_uplo_t uplo, magma_int_t n, magma_int_t NB,
                              magmaDoubleComplex *a, magma_int_t lda,
                              magmaDoubleComplex *tau, magmaDoubleComplex *work, magma_int_t lwork,
                              magmaDoubleComplex *dT,
                              magma_int_t *info);

magma_int_t magma_zhetrd_he2hb_mgpu( magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                              magmaDoubleComplex *a, magma_int_t lda,
                              magmaDoubleComplex *tau,
                              magmaDoubleComplex *work, magma_int_t lwork,
                              magmaDoubleComplex *dAmgpu[], magma_int_t ldda,
                              magmaDoubleComplex *dTmgpu[], magma_int_t lddt,
                              magma_int_t ngpu, magma_int_t distblk,
                              magma_queue_t streams[][20], magma_int_t nstream,
                              magma_int_t *info);

magma_int_t magma_zhetrd_he2hb_mgpu_spec( magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
                              magmaDoubleComplex *a, magma_int_t lda,
                              magmaDoubleComplex *tau,
                              magmaDoubleComplex *work, magma_int_t lwork,
                              magmaDoubleComplex *dAmgpu[], magma_int_t ldda,
                              magmaDoubleComplex *dTmgpu[], magma_int_t lddt,
                              magma_int_t ngpu, magma_int_t distblk,
                              magma_queue_t streams[][20], magma_int_t nstream,
                              magma_int_t *info);

magma_int_t magma_zpotrs_gpu( magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs,
                              magmaDoubleComplex *dA, magma_int_t ldda,
                              magmaDoubleComplex *dB, magma_int_t lddb, magma_int_t *info);

magma_int_t magma_zssssm_gpu( magma_order_t order, magma_int_t m1, magma_int_t n1,
                              magma_int_t m2, magma_int_t n2, magma_int_t k, magma_int_t ib,
                              magmaDoubleComplex *dA1, magma_int_t ldda1,
                              magmaDoubleComplex *dA2, magma_int_t ldda2,
                              magmaDoubleComplex *dL1, magma_int_t lddl1,
                              magmaDoubleComplex *dL2, magma_int_t lddl2,
                              magma_int_t *IPIV, magma_int_t *info);

magma_int_t magma_ztstrf_gpu( magma_order_t order, magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb,
                              magmaDoubleComplex *hU, magma_int_t ldhu, magmaDoubleComplex *dU, magma_int_t lddu,
                              magmaDoubleComplex *hA, magma_int_t ldha, magmaDoubleComplex *dA, magma_int_t ldda,
                              magmaDoubleComplex *hL, magma_int_t ldhl, magmaDoubleComplex *dL, magma_int_t lddl,
                              magma_int_t *ipiv,
                              magmaDoubleComplex *hwork, magma_int_t ldhwork,
                              magmaDoubleComplex *dwork, magma_int_t lddwork,
                              magma_int_t *info);

magma_int_t magma_zungqr_gpu( magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaDoubleComplex *da, magma_int_t ldda,
                              magmaDoubleComplex *tau, magmaDoubleComplex *dwork,
                              magma_int_t nb, magma_int_t *info );

magma_int_t magma_zunmql2_gpu(magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaDoubleComplex *da, magma_int_t ldda,
                              magmaDoubleComplex *tau,
                              magmaDoubleComplex *dc, magma_int_t lddc,
                              magmaDoubleComplex *wa, magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_zunmqr_gpu( magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaDoubleComplex *dA,    magma_int_t ldda, magmaDoubleComplex *tau,
                              magmaDoubleComplex *dC,    magma_int_t lddc,
                              magmaDoubleComplex *hwork, magma_int_t lwork,
                              magmaDoubleComplex *dT,    magma_int_t nb, magma_int_t *info);

magma_int_t magma_zunmqr2_gpu(magma_side_t side, magma_trans_t trans,
                              magma_int_t m, magma_int_t n, magma_int_t k,
                              magmaDoubleComplex *da,   magma_int_t ldda,
                              magmaDoubleComplex *tau,
                              magmaDoubleComplex *dc,    magma_int_t lddc,
                              magmaDoubleComplex *wa,    magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_zunmtr_gpu( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                              magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *da,    magma_int_t ldda,
                              magmaDoubleComplex *tau,
                              magmaDoubleComplex *dc,    magma_int_t lddc,
                              magmaDoubleComplex *wa,    magma_int_t ldwa,
                              magma_int_t *info);

magma_int_t magma_zgeqp3_gpu( magma_int_t m, magma_int_t n,
                              magmaDoubleComplex *dA, magma_int_t ldda,
                              magma_int_t *jpvt, magmaDoubleComplex *tau,
                              magmaDoubleComplex *dwork, magma_int_t lwork,
                              #ifdef COMPLEX
                              double *rwork,
                              #endif
                              magma_int_t *info );

magma_int_t magma_zheevd_gpu( magma_vec_t jobz, magma_uplo_t uplo,
                              magma_int_t n,
                              magmaDoubleComplex *da, magma_int_t ldda,
                              double *w,
                              magmaDoubleComplex *wa,  magma_int_t ldwa,
                              magmaDoubleComplex *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              double *rwork, magma_int_t lrwork,
                              #endif
                              magma_int_t *iwork, magma_int_t liwork,
                              magma_int_t *info);

magma_int_t magma_zheevdx_gpu(magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
                              magma_int_t n, magmaDoubleComplex *da,
                              magma_int_t ldda, double vl, double vu,
                              magma_int_t il, magma_int_t iu,
                              magma_int_t *m, double *w,
                              magmaDoubleComplex *wa,  magma_int_t ldwa,
                              magmaDoubleComplex *work, magma_int_t lwork,
                              #ifdef COMPLEX
                              double *rwork, magma_int_t lrwork,
                              #endif
                              magma_int_t *iwork, magma_int_t liwork,
                              magma_int_t *info);

#ifdef COMPLEX
// no real [sd] precisions available
magma_int_t magma_zheevx_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                              magmaDoubleComplex *da, magma_int_t ldda, double vl,
                              double vu, magma_int_t il, magma_int_t iu,
                              double abstol, magma_int_t *m,
                              double *w, magmaDoubleComplex *dz, magma_int_t lddz,
                              magmaDoubleComplex *wa, magma_int_t ldwa,
                              magmaDoubleComplex *wz, magma_int_t ldwz,
                              magmaDoubleComplex *work, magma_int_t lwork,
                              double *rwork, magma_int_t *iwork,
                              magma_int_t *ifail, magma_int_t *info);

magma_int_t magma_zheevr_gpu( magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
                              magmaDoubleComplex *da, magma_int_t ldda, double vl, double vu,
                              magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
                              double *w, magmaDoubleComplex *dz, magma_int_t lddz,
                              magma_int_t *isuppz,
                              magmaDoubleComplex *wa, magma_int_t ldwa,
                              magmaDoubleComplex *wz, magma_int_t ldwz,
                              magmaDoubleComplex *work, magma_int_t lwork,
                              double *rwork, magma_int_t lrwork, magma_int_t *iwork,
                              magma_int_t liwork, magma_int_t *info);
#endif  // COMPLEX

magma_int_t magma_zhegst_gpu(magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
                             magmaDoubleComplex *da, magma_int_t ldda,
                             magmaDoubleComplex *db, magma_int_t lddb, magma_int_t *info);


/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA utility function definitions
*/

void magma_zprint    ( magma_int_t m, magma_int_t n, const magmaDoubleComplex  *A, magma_int_t lda  );
void magma_zprint_gpu( magma_int_t m, magma_int_t n, const magmaDoubleComplex *dA, magma_int_t ldda );

void zpanel_to_q( magma_uplo_t uplo, magma_int_t ib, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *work );
void zq_to_panel( magma_uplo_t uplo, magma_int_t ib, magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *work );

magma_int_t magma_znan_inf(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf );

magma_int_t magma_znan_inf_gpu(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *dA, magma_int_t ldda,
    magma_int_t *cnt_nan,
    magma_int_t *cnt_inf );

extern const magmaDoubleComplex MAGMA_Z_NAN;
extern const magmaDoubleComplex MAGMA_Z_INF;

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif /* MAGMA_Z_H */
