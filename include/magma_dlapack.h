/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_zlapack.h normal z -> d, Fri Jan 30 19:00:06 2015
*/

#ifndef MAGMA_DLAPACK_H
#define MAGMA_DLAPACK_H

#include "magma_types.h"
#include "magma_mangling.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- BLAS and LAPACK functions (alphabetical order)
*/
#define blasf77_idamax     FORTRAN_NAME( idamax, IDAMAX )
#define blasf77_daxpy      FORTRAN_NAME( daxpy,  DAXPY  )
#define blasf77_dcopy      FORTRAN_NAME( dcopy,  DCOPY  )
#define blasf77_dgemm      FORTRAN_NAME( dgemm,  DGEMM  )
#define blasf77_dgemv      FORTRAN_NAME( dgemv,  DGEMV  )
#define blasf77_dger      FORTRAN_NAME( dger,  DGER  )
#define blasf77_dger      FORTRAN_NAME( dger,  DGER  )
#define blasf77_dsymm      FORTRAN_NAME( dsymm,  DSYMM  )
#define blasf77_dsymv      FORTRAN_NAME( dsymv,  DSYMV  )
#define blasf77_dsyr       FORTRAN_NAME( dsyr,   DSYR   )
#define blasf77_dsyr2      FORTRAN_NAME( dsyr2,  DSYR2  )
#define blasf77_dsyr2k     FORTRAN_NAME( dsyr2k, DSYR2K )
#define blasf77_dsyrk      FORTRAN_NAME( dsyrk,  DSYRK  )
#define blasf77_dscal      FORTRAN_NAME( dscal,  DSCAL  )
#define blasf77_dscal     FORTRAN_NAME( dscal, DSCAL )
#define blasf77_dswap      FORTRAN_NAME( dswap,  DSWAP  )
#define blasf77_dsymm      FORTRAN_NAME( dsymm,  DSYMM  )
#define blasf77_dsyr2k     FORTRAN_NAME( dsyr2k, DSYR2K )
#define blasf77_dsyrk      FORTRAN_NAME( dsyrk,  DSYRK  )
#define blasf77_drotg      FORTRAN_NAME( drotg,  DROTG  )
#define blasf77_drot       FORTRAN_NAME( drot,   DROT   )
#define blasf77_drot      FORTRAN_NAME( drot,  DROT  )
#define blasf77_dtrmm      FORTRAN_NAME( dtrmm,  DTRMM  )
#define blasf77_dtrmv      FORTRAN_NAME( dtrmv,  DTRMV  )
#define blasf77_dtrsm      FORTRAN_NAME( dtrsm,  DTRSM  )
#define blasf77_dtrsv      FORTRAN_NAME( dtrsv,  DTRSV  )

#define lapackf77_dlaed2   FORTRAN_NAME( dlaed2, DLAED2 )
#define lapackf77_dlaed4   FORTRAN_NAME( dlaed4, DLAED4 )
#define lapackf77_dlaln2   FORTRAN_NAME( dlaln2, DLALN2 )
#define lapackf77_dlamc3   FORTRAN_NAME( dlamc3, DLAMC3 )
#define lapackf77_dlamrg   FORTRAN_NAME( dlamrg, DLAMRG )
#define lapackf77_dlasrt   FORTRAN_NAME( dlasrt, DLASRT )
#define lapackf77_dstebz   FORTRAN_NAME( dstebz, DSTEBZ )

#define lapackf77_dbdsdc   FORTRAN_NAME( dbdsdc, DBDSDC )
#define lapackf77_dbdsqr   FORTRAN_NAME( dbdsqr, DBDSQR )
#define lapackf77_dgebak   FORTRAN_NAME( dgebak, DGEBAK )
#define lapackf77_dgebal   FORTRAN_NAME( dgebal, DGEBAL )
#define lapackf77_dgebd2   FORTRAN_NAME( dgebd2, DGEBD2 )
#define lapackf77_dgebrd   FORTRAN_NAME( dgebrd, DGEBRD )
#define lapackf77_dgbbrd   FORTRAN_NAME( dgbbrd, DGBBRD )
#define lapackf77_dgeev    FORTRAN_NAME( dgeev,  DGEEV  )
#define lapackf77_dgehd2   FORTRAN_NAME( dgehd2, DGEHD2 )
#define lapackf77_dgehrd   FORTRAN_NAME( dgehrd, DGEHRD )
#define lapackf77_dgelqf   FORTRAN_NAME( dgelqf, DGELQF )
#define lapackf77_dgels    FORTRAN_NAME( dgels,  DGELS  )
#define lapackf77_dgeqlf   FORTRAN_NAME( dgeqlf, DGEQLF )
#define lapackf77_dgeqp3   FORTRAN_NAME( dgeqp3, DGEQP3 )
#define lapackf77_dgeqrf   FORTRAN_NAME( dgeqrf, DGEQRF )
#define lapackf77_dgesdd   FORTRAN_NAME( dgesdd, DGESDD )
#define lapackf77_dgesv    FORTRAN_NAME( dgesv,  DGESV  )
#define lapackf77_dgesvd   FORTRAN_NAME( dgesvd, DGESVD )
#define lapackf77_dgetrf   FORTRAN_NAME( dgetrf, DGETRF )
#define lapackf77_dgetri   FORTRAN_NAME( dgetri, DGETRI )
#define lapackf77_dgetrs   FORTRAN_NAME( dgetrs, DGETRS )
#define lapackf77_dsytf2   FORTRAN_NAME( dsytf2, DSYTF2 )
#define lapackf77_dsytrs   FORTRAN_NAME( dsytrs, DSYTRS )
#define lapackf77_dsbtrd   FORTRAN_NAME( dsbtrd, DSBTRD )
#define lapackf77_dsyev    FORTRAN_NAME( dsyev,  DSYEV  )
#define lapackf77_dsyevd   FORTRAN_NAME( dsyevd, DSYEVD )
#define lapackf77_dsyevr   FORTRAN_NAME( dsyevr, DSYEVR )
#define lapackf77_dsyevx   FORTRAN_NAME( dsyevx, DSYEVX )
#define lapackf77_dsygs2   FORTRAN_NAME( dsygs2, DSYGS2 )
#define lapackf77_dsygst   FORTRAN_NAME( dsygst, DSYGST )
#define lapackf77_dsygvd   FORTRAN_NAME( dsygvd, DSYGVD )
#define lapackf77_dsysv    FORTRAN_NAME( dsysv,  DSYSV  )
#define lapackf77_dsytd2   FORTRAN_NAME( dsytd2, DSYTD2 )
#define lapackf77_dsytrd   FORTRAN_NAME( dsytrd, DSYTRD )
#define lapackf77_dsytrf   FORTRAN_NAME( dsytrf, DSYTRF )
#define lapackf77_dhseqr   FORTRAN_NAME( dhseqr, DHSEQR )
#define lapackf77_dlabrd   FORTRAN_NAME( dlabrd, DLABRD )
#define lapackf77_dlacgv   FORTRAN_NAME( dlacgv, DLACGV )
#define lapackf77_dlacp2   FORTRAN_NAME( dlacp2, DLACP2 )
#define lapackf77_dlacpy   FORTRAN_NAME( dlacpy, DLACPY )
#define lapackf77_dlacrm   FORTRAN_NAME( dlacrm, DLACRM )
#define lapackf77_dladiv   FORTRAN_NAME( dladiv, DLADIV )
#define lapackf77_dlasyf   FORTRAN_NAME( dlasyf, DLASYF )
#define lapackf77_dlange   FORTRAN_NAME( dlange, DLANGE )
#define lapackf77_dlansy   FORTRAN_NAME( dlansy, DLANSY )
#define lapackf77_dlanst   FORTRAN_NAME( dlanst, DLANST )
#define lapackf77_dlansy   FORTRAN_NAME( dlansy, DLANSY )
#define lapackf77_dlantr   FORTRAN_NAME( dlantr, DLANTR )
#define lapackf77_dlapy3   FORTRAN_NAME( dlapy3, DLAPY3 )
#define lapackf77_dlaqp2   FORTRAN_NAME( dlaqp2, DLAQP2 )
#define lapackf77_dlarcm   FORTRAN_NAME( dlarcm, DLARCM )
#define lapackf77_dlarf    FORTRAN_NAME( dlarf,  DLARF  )
#define lapackf77_dlarfb   FORTRAN_NAME( dlarfb, DLARFB )
#define lapackf77_dlarfg   FORTRAN_NAME( dlarfg, DLARFG )
#define lapackf77_dlarft   FORTRAN_NAME( dlarft, DLARFT )
#define lapackf77_dlarnv   FORTRAN_NAME( dlarnv, DLARNV )
#define lapackf77_dlartg   FORTRAN_NAME( dlartg, DLARTG )
#define lapackf77_dlascl   FORTRAN_NAME( dlascl, DLASCL )
#define lapackf77_dlaset   FORTRAN_NAME( dlaset, DLASET )
#define lapackf77_dlaswp   FORTRAN_NAME( dlaswp, DLASWP )
#define lapackf77_dlatrd   FORTRAN_NAME( dlatrd, DLATRD )
#define lapackf77_dlatrs   FORTRAN_NAME( dlatrs, DLATRS )
#define lapackf77_dlauum   FORTRAN_NAME( dlauum, DLAUUM )
#define lapackf77_dlavsy   FORTRAN_NAME( dlavsy, DLAVSY )
#define lapackf77_dposv    FORTRAN_NAME( dposv,  DPOSV  )
#define lapackf77_dpotrf   FORTRAN_NAME( dpotrf, DPOTRF )
#define lapackf77_dpotri   FORTRAN_NAME( dpotri, DPOTRI )
#define lapackf77_dpotrs   FORTRAN_NAME( dpotrs, DPOTRS )
#define lapackf77_dstedc   FORTRAN_NAME( dstedc, DSTEDC )
#define lapackf77_dstein   FORTRAN_NAME( dstein, DSTEIN )
#define lapackf77_dstemr   FORTRAN_NAME( dstemr, DSTEMR )
#define lapackf77_dsteqr   FORTRAN_NAME( dsteqr, DSTEQR )
#define lapackf77_dsymv    FORTRAN_NAME( dsymv,  DSYMV  )
#define lapackf77_dtrevc   FORTRAN_NAME( dtrevc, DTREVC )
#define lapackf77_dtrevc3  FORTRAN_NAME( dtrevc3, DTREVC3 )
#define lapackf77_dtrtri   FORTRAN_NAME( dtrtri, DTRTRI )
#define lapackf77_dorg2r   FORTRAN_NAME( dorg2r, DORG2R )
#define lapackf77_dorgbr   FORTRAN_NAME( dorgbr, DORGBR )
#define lapackf77_dorghr   FORTRAN_NAME( dorghr, DORGHR )
#define lapackf77_dorglq   FORTRAN_NAME( dorglq, DORGLQ )
#define lapackf77_dorgql   FORTRAN_NAME( dorgql, DORGQL )
#define lapackf77_dorgqr   FORTRAN_NAME( dorgqr, DORGQR )
#define lapackf77_dorgtr   FORTRAN_NAME( dorgtr, DORGTR )
#define lapackf77_dorm2r   FORTRAN_NAME( dorm2r, DORM2R )
#define lapackf77_dormbr   FORTRAN_NAME( dormbr, DORMBR )
#define lapackf77_dormlq   FORTRAN_NAME( dormlq, DORMLQ )
#define lapackf77_dormql   FORTRAN_NAME( dormql, DORMQL )
#define lapackf77_dormqr   FORTRAN_NAME( dormqr, DORMQR )
#define lapackf77_dormtr   FORTRAN_NAME( dormtr, DORMTR )

/* testing functions (alphabetical order) */
#define lapackf77_dbdt01   FORTRAN_NAME( dbdt01, DBDT01 )
#define lapackf77_dget22   FORTRAN_NAME( dget22, DGET22 )
#define lapackf77_dsyt21   FORTRAN_NAME( dsyt21, DSYT21 )
#define lapackf77_dhst01   FORTRAN_NAME( dhst01, DHST01 )
#define lapackf77_dlarfx   FORTRAN_NAME( dlarfx, DLARFX )
#define lapackf77_dlarfy   FORTRAN_NAME( dlarfy, DLARFY )
#define lapackf77_dlatms   FORTRAN_NAME( dlatms, DLATMS )
#define lapackf77_dqpt01   FORTRAN_NAME( dqpt01, DQPT01 )
#define lapackf77_dqrt02   FORTRAN_NAME( dqrt02, DQRT02 )
#define lapackf77_dstt21   FORTRAN_NAME( dstt21, DSTT21 )
#define lapackf77_dort01   FORTRAN_NAME( dort01, DORT01 )

/*
 * BLAS functions (alphabetical order)
 */
magma_int_t blasf77_idamax(
                     const magma_int_t *n,
                     const double *x, const magma_int_t *incx );

void blasf77_daxpy(  const magma_int_t *n,
                     const double *alpha,
                     const double *x, const magma_int_t *incx,
                           double *y, const magma_int_t *incy );

void blasf77_dcopy(  const magma_int_t *n,
                     const double *x, const magma_int_t *incx,
                           double *y, const magma_int_t *incy );

void blasf77_dgemm(  const char *transa, const char *transb,
                     const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                     const double *B, const magma_int_t *ldb,
                     const double *beta,
                           double *C, const magma_int_t *ldc );

void blasf77_dgemv(  const char *transa,
                     const magma_int_t *m, const magma_int_t *n,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                     const double *x, const magma_int_t *incx,
                     const double *beta,
                           double *y, const magma_int_t *incy );

void blasf77_dger(  const magma_int_t *m, const magma_int_t *n,
                     const double *alpha,
                     const double *x, const magma_int_t *incx,
                     const double *y, const magma_int_t *incy,
                           double *A, const magma_int_t *lda );

void blasf77_dger(  const magma_int_t *m, const magma_int_t *n,
                     const double *alpha,
                     const double *x, const magma_int_t *incx,
                     const double *y, const magma_int_t *incy,
                           double *A, const magma_int_t *lda );

void blasf77_dsymm(  const char *side, const char *uplo,
                     const magma_int_t *m, const magma_int_t *n,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                     const double *B, const magma_int_t *ldb,
                     const double *beta,
                           double *C, const magma_int_t *ldc );

void blasf77_dsymv(  const char *uplo,
                     const magma_int_t *n,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                     const double *x, const magma_int_t *incx,
                     const double *beta,
                           double *y, const magma_int_t *incy );

void blasf77_dsyr(   const char *uplo,
                     const magma_int_t *n,
                     const double *alpha,
                     const double *x, const magma_int_t *incx,
                           double *A, const magma_int_t *lda );

void blasf77_dsyr2(  const char *uplo,
                     const magma_int_t *n,
                     const double *alpha,
                     const double *x, const magma_int_t *incx,
                     const double *y, const magma_int_t *incy,
                           double *A, const magma_int_t *lda );

void blasf77_dsyr2k( const char *uplo, const char *trans,
                     const magma_int_t *n, const magma_int_t *k,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                     const double *B, const magma_int_t *ldb,
                     const double *beta,
                           double *C, const magma_int_t *ldc );

void blasf77_dsyrk(  const char *uplo, const char *trans,
                     const magma_int_t *n, const magma_int_t *k,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                     const double *beta,
                           double *C, const magma_int_t *ldc );

void blasf77_dscal(  const magma_int_t *n,
                     const double *alpha,
                           double *x, const magma_int_t *incx );

void blasf77_dscal( const magma_int_t *n,
                     const double *alpha,
                           double *x, const magma_int_t *incx );

void blasf77_dswap(  const magma_int_t *n,
                     double *x, const magma_int_t *incx,
                     double *y, const magma_int_t *incy );

/* real-symmetric (non-symmetric) routines */
void blasf77_dsymm(  const char *side, const char *uplo,
                     const magma_int_t *m, const magma_int_t *n,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                     const double *B, const magma_int_t *ldb,
                     const double *beta,
                           double *C, const magma_int_t *ldc );

void blasf77_dsyr2k( const char *uplo, const char *trans,
                     const magma_int_t *n, const magma_int_t *k,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                     const double *B, const magma_int_t *ldb,
                     const double *beta,
                           double *C, const magma_int_t *ldc );

void blasf77_dsyrk(  const char *uplo, const char *trans,
                     const magma_int_t *n, const magma_int_t *k,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                     const double *beta,
                           double *C, const magma_int_t *ldc );

void blasf77_drotg(  double *ca, const double *cb,
                     double *c, double *s );
                     
void blasf77_drot(   const magma_int_t *n,
                     double *x, const magma_int_t *incx,
                     double *y, const magma_int_t *incy,
                     const double *c, const double *s );
                     
void blasf77_drot(  const magma_int_t *n,
                     double *x, const magma_int_t *incx,
                     double *y, const magma_int_t *incy,
                     const double *c, const double *s );

void blasf77_dtrmm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const magma_int_t *m, const magma_int_t *n,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                           double *B, const magma_int_t *ldb );

void blasf77_dtrmv(  const char *uplo, const char *transa, const char *diag,
                     const magma_int_t *n,
                     const double *A, const magma_int_t *lda,
                           double *x, const magma_int_t *incx );

void blasf77_dtrsm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const magma_int_t *m, const magma_int_t *n,
                     const double *alpha,
                     const double *A, const magma_int_t *lda,
                           double *B, const magma_int_t *ldb );

void blasf77_dtrsv(  const char *uplo, const char *transa, const char *diag,
                     const magma_int_t *n,
                     const double *A, const magma_int_t *lda,
                           double *x, const magma_int_t *incx );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA wrappers around BLAS functions (alphabetical order)
    The Fortran interface for these is not portable, so we
    provide a C interface identical to the Fortran interface.
*/

double magma_cblas_dasum(
    magma_int_t n,
    const double *x, magma_int_t incx );

double magma_cblas_dnrm2(
    magma_int_t n,
    const double *x, magma_int_t incx );

double magma_cblas_ddot(
    magma_int_t n,
    const double *x, magma_int_t incx,
    const double *y, magma_int_t incy );

double magma_cblas_ddot(
    magma_int_t n,
    const double *x, magma_int_t incx,
    const double *y, magma_int_t incy );


/*
 * LAPACK functions (alphabetical order)
 */
#ifdef REAL
void   lapackf77_dbdsdc( const char *uplo, const char *compq,
                         const magma_int_t *n,
                         double *d, double *e,
                         double *U,  const magma_int_t *ldu,
                         double *VT, const magma_int_t *ldvt,
                         double *Q, magma_int_t *IQ,
                         double *work, magma_int_t *iwork,
                         magma_int_t *info );
#endif

void   lapackf77_dbdsqr( const char *uplo,
                         const magma_int_t *n, const magma_int_t *ncvt, const magma_int_t *nru,  const magma_int_t *ncc,
                         double *d, double *e,
                         double *Vt, const magma_int_t *ldvt,
                         double *U, const magma_int_t *ldu,
                         double *C, const magma_int_t *ldc,
                         double *work,
                         magma_int_t *info );

void   lapackf77_dgebak( const char *job, const char *side,
                         const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         const double *scale, const magma_int_t *m,
                         double *V, const magma_int_t *ldv,
                         magma_int_t *info );

void   lapackf77_dgebal( const char *job,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         magma_int_t *ilo, magma_int_t *ihi,
                         double *scale,
                         magma_int_t *info );

void   lapackf77_dgebd2( const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *d, double *e,
                         double *tauq,
                         double *taup,
                         double *work,
                         magma_int_t *info );

void   lapackf77_dgebrd( const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *d, double *e,
                         double *tauq,
                         double *taup,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dgbbrd( const char *vect, const magma_int_t *m,
                         const magma_int_t *n, const magma_int_t *ncc,
                         const magma_int_t *kl, const magma_int_t *ku,
                         double *Ab, const magma_int_t *ldab,
                         double *d, double *e,
                         double *Q, const magma_int_t *ldq,
                         double *PT, const magma_int_t *ldpt,
                         double *C, const magma_int_t *ldc,
                         double *work,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_dgeev(  const char *jobvl, const char *jobvr,
                         const magma_int_t *n,
                         double *A,    const magma_int_t *lda,
                         #ifdef COMPLEX
                         double *w,
                         #else
                         double *wr, double *wi,
                         #endif
                         double *Vl,   const magma_int_t *ldvl,
                         double *Vr,   const magma_int_t *ldvr,
                         double *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_dgehd2( const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         double *A, const magma_int_t *lda,
                         double *tau,
                         double *work,
                         magma_int_t *info );

void   lapackf77_dgehrd( const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         double *A, const magma_int_t *lda,
                         double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dgelqf( const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dgels(  const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *nrhs,
                         double *A, const magma_int_t *lda,
                         double *B, const magma_int_t *ldb,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dgeqlf( const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dgeqp3( const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         magma_int_t *jpvt,
                         double *tau,
                         double *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_dgeqrf( const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dgesdd( const char *jobz,
                         const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *s,
                         double *U,  const magma_int_t *ldu,
                         double *Vt, const magma_int_t *ldvt,
                         double *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         magma_int_t *iwork, magma_int_t *info );

void   lapackf77_dgesv(  const magma_int_t *n, const magma_int_t *nrhs,
                         double *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         double *B,  const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_dgesvd( const char *jobu, const char *jobvt,
                         const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *s,
                         double *U,  const magma_int_t *ldu,
                         double *Vt, const magma_int_t *ldvt,
                         double *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_dgetrf( const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         magma_int_t *info );

void   lapackf77_dgetri( const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         const magma_int_t *ipiv,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dgetrs( const char *trans,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         const double *A, const magma_int_t *lda,
                         const magma_int_t *ipiv,
                         double *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_dsytf2( const char*, magma_int_t*, 
                         double*, magma_int_t*, magma_int_t*, magma_int_t* );

void   lapackf77_dsytrs( const char *uplo,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         const double *A, const magma_int_t *lda,
                         const magma_int_t *ipiv,
                         double *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_dsbtrd( const char *vect, const char *uplo,
                         const magma_int_t *n, const magma_int_t *kd,
                         double *Ab, const magma_int_t *ldab,
                         double *d, double *e,
                         double *Q, const magma_int_t *ldq,
                         double *work,
                         magma_int_t *info );

void   lapackf77_dsyev(  const char *jobz, const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *w,
                         double *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_dsyevd( const char *jobz, const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *w,
                         double *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork, const magma_int_t *lrwork,
                         #endif
                         magma_int_t *iwork, const magma_int_t *liwork,
                         magma_int_t *info );

void   lapackf77_dsyevr( const char *jobz, const char *range, const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *vl, double *vu, 
                         magma_int_t *il, magma_int_t *iu, double *abstol, 
                         magma_int_t *m, double *w, double *z__, 
                         magma_int_t *ldz, magma_int_t *isuppz, 
                         double *work, magma_int_t *lwork, 
                         double *rwork, magma_int_t *lrwork, 
                         magma_int_t *iwork, magma_int_t *liwork, magma_int_t *info);

void   lapackf77_dsyevx( const char *jobz, const char *range, const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *vl, double *vu,
                         magma_int_t *il, magma_int_t *iu, double *abstol,
                         magma_int_t *m, double *w, double *z__,
                         magma_int_t *ldz, double *work, magma_int_t *lwork,
                         double *rwork, magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info);

void   lapackf77_dsygs2( const magma_int_t *itype, const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_dsygst( const magma_int_t *itype, const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_dsygvd( const magma_int_t *itype, const char *jobz, const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *B, const magma_int_t *ldb,
                         double *w,
                         double *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork, const magma_int_t *lrwork,
                         #endif
                         magma_int_t *iwork, const magma_int_t *liwork,
                         magma_int_t *info );

void   lapackf77_dsysv( const char *uplo, 
                        const magma_int_t *n, const magma_int_t *nrhs,
                        double *A, const magma_int_t *lda, magma_int_t *ipiv,
                        double *B, const magma_int_t *ldb,
                        double *work, const magma_int_t *lwork,
                        magma_int_t *info );

void   lapackf77_dsytd2( const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *d, double *e,
                         double *tau,
                         magma_int_t *info );

void   lapackf77_dsytrd( const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *d, double *e,
                         double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dsytrf( const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dhseqr( const char *job, const char *compz,
                         const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         double *H, const magma_int_t *ldh,
                         #ifdef COMPLEX
                         double *w,
                         #else
                         double *wr, double *wi,
                         #endif
                         double *Z, const magma_int_t *ldz,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dlabrd( const magma_int_t *m, const magma_int_t *n, const magma_int_t *nb,
                         double *A, const magma_int_t *lda,
                         double *d, double *e,
                         double *tauq,
                         double *taup,
                         double *X, const magma_int_t *ldx,
                         double *Y, const magma_int_t *ldy );

#ifdef COMPLEX
void   lapackf77_dlacgv( const magma_int_t *n,
                         double *x, const magma_int_t *incx );
#endif

#ifdef COMPLEX
void   lapackf77_dlacp2( const char *uplo,
                         const magma_int_t *m, const magma_int_t *n,
                         const double *A, const magma_int_t *lda,
                         double *B, const magma_int_t *ldb );
#endif

void   lapackf77_dlacpy( const char *uplo,
                         const magma_int_t *m, const magma_int_t *n,
                         const double *A, const magma_int_t *lda,
                         double *B, const magma_int_t *ldb );

#ifdef COMPLEX
void   lapackf77_dlacrm( const magma_int_t *m, const magma_int_t *n,
                         const double *A, const magma_int_t *lda,
                         const double             *B, const magma_int_t *ldb,
                         double       *C, const magma_int_t *ldc,
                         double *rwork );
#endif

#ifdef COMPLEX
void   lapackf77_dladiv( double *ret_val,
                         double *x,
                         double *y );
#else // REAL
void   lapackf77_dladiv( const double *a, const double *b,
                         const double *c, const double *d,
                         double *p, double *q );
#endif

void   lapackf77_dlasyf( const char *uplo,
                         const magma_int_t *n, const magma_int_t *kn,
                         magma_int_t *kb,
                         double *A, const magma_int_t lda,
                         magma_int_t *ipiv,
                         double *work, const magma_int_t *ldwork,
                         magma_int_t *info );

double lapackf77_dlange( const char *norm,
                         const magma_int_t *m, const magma_int_t *n,
                         const double *A, const magma_int_t *lda,
                         double *work );

double lapackf77_dlansy( const char *norm, const char *uplo,
                         const magma_int_t *n,
                         const double *A, const magma_int_t *lda,
                         double *work );

double lapackf77_dlanst( const char *norm, const magma_int_t *n,
                         const double *d, const double *e );

double lapackf77_dlansy( const char *norm, const char *uplo,
                         const magma_int_t *n,
                         const double *A, const magma_int_t *lda,
                         double *work );

double lapackf77_dlantr( const char *norm, const char *uplo, const char *diag,
                         const magma_int_t *m, const magma_int_t *n,
                         const double *A, const magma_int_t *lda,
                         double *work );

void   lapackf77_dlaqp2( magma_int_t *m, magma_int_t *n, magma_int_t *offset,
                         double *a, magma_int_t *lda, magma_int_t *jpvt,
                         double *tau,
                         double *vn1, double *vn2,
                         double *work );

#ifdef COMPLEX
void   lapackf77_dlarcm( const magma_int_t *m, const magma_int_t *n,
                         const double             *A, const magma_int_t *lda,
                         const double *B, const magma_int_t *ldb,
                         double       *C, const magma_int_t *ldc,
                         double *rwork );
#endif

void   lapackf77_dlarf(  const char *side, const magma_int_t *m, const magma_int_t *n,
                         double *v, const magma_int_t *incv,
                         double *tau,
                         double *C, const magma_int_t *ldc,
                         double *work );

void   lapackf77_dlarfb( const char *side, const char *trans, const char *direct, const char *storev,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const double *V, const magma_int_t *ldv,
                         const double *T, const magma_int_t *ldt,
                         double *C, const magma_int_t *ldc,
                         double *work, const magma_int_t *ldwork );

void   lapackf77_dlarfg( const magma_int_t *n,
                         double *alpha,
                         double *x, const magma_int_t *incx,
                         double *tau );

void   lapackf77_dlarft( const char *direct, const char *storev,
                         const magma_int_t *n, const magma_int_t *k,
                         double *V, const magma_int_t *ldv,
                         const double *tau,
                         double *T, const magma_int_t *ldt );

void   lapackf77_dlarnv( const magma_int_t *idist, magma_int_t *iseed, const magma_int_t *n,
                         double *x );

void   lapackf77_dlartg( double *F,
                         double *G,
                         double *cs,
                         double *SN,
                         double *R );

void   lapackf77_dlascl( const char *type,
                         const magma_int_t *kl, const magma_int_t *ku,
                         double *cfrom,
                         double *cto,
                         const magma_int_t *m, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_dlaset( const char *uplo,
                         const magma_int_t *m, const magma_int_t *n,
                         const double *alpha,
                         const double *beta,
                         double *A, const magma_int_t *lda );

void   lapackf77_dlaswp( const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         const magma_int_t *k1, const magma_int_t *k2,
                         magma_int_t *ipiv,
                         const magma_int_t *incx );

void   lapackf77_dlatrd( const char *uplo,
                         const magma_int_t *n, const magma_int_t *nb,
                         double *A, const magma_int_t *lda,
                         double *e,
                         double *tau,
                         double *work, const magma_int_t *ldwork );

void   lapackf77_dlatrs( const char *uplo, const char *trans, const char *diag,
                         const char *normin,
                         const magma_int_t *n,
                         const double *A, const magma_int_t *lda,
                         double *x, double *scale,
                         double *cnorm, magma_int_t *info );

void   lapackf77_dlauum( const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_dlavsy( const char *uplo, const char *trans, const char *diag,
                         magma_int_t *n, magma_int_t *nrhs,
                         double *A, magma_int_t *lda,
                         magma_int_t *ipiv,
                         double *B, magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_dposv(  const char *uplo,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         double *A, const magma_int_t *lda,
                         double *B,  const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_dpotrf( const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_dpotri( const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_dpotrs( const char *uplo,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         const double *A, const magma_int_t *lda,
                         double *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_dstedc( const char *compz,
                         const magma_int_t *n,
                         double *d, double *e,
                         double *Z, const magma_int_t *ldz,
                         double *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         double *rwork, const magma_int_t *lrwork,
                         #endif
                         magma_int_t *iwork, const magma_int_t *liwork,
                         magma_int_t *info );

void   lapackf77_dstein( const magma_int_t *n,
                         const double *d, const double *e,
                         const magma_int_t *m,
                         const double *w,
                         const magma_int_t *iblock,
                         const magma_int_t *isplit,
                         double *Z, const magma_int_t *ldz,
                         double *work, magma_int_t *iwork, magma_int_t *ifailv,
                         magma_int_t *info );

void   lapackf77_dstemr( const char *jobz, const char *range,
                         const magma_int_t *n,
                         double *d, double *e,
                         const double *vl, const double *vu,
                         const magma_int_t *il, const magma_int_t *iu,
                         magma_int_t *m,
                         double *w,
                         double *Z, const magma_int_t *ldz,
                         const magma_int_t *nzc, magma_int_t *isuppz, magma_int_t *tryrac,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *iwork, const magma_int_t *liwork,
                         magma_int_t *info );

void   lapackf77_dsteqr( const char *compz,
                         const magma_int_t *n,
                         double *d, double *e,
                         double *Z, const magma_int_t *ldz,
                         double *work,
                         magma_int_t *info );

#ifdef COMPLEX
void   lapackf77_dsymv(  const char *uplo,
                         const magma_int_t *n,
                         const double *alpha,
                         const double *A, const magma_int_t *lda,
                         const double *x, const magma_int_t *incx,
                         const double *beta,
                               double *y, const magma_int_t *incy );
#endif

void   lapackf77_dtrevc( const char *side, const char *howmny,
                         magma_int_t *select, const magma_int_t *n,
                         double *T,  const magma_int_t *ldt,
                         double *Vl, const magma_int_t *ldvl,
                         double *Vr, const magma_int_t *ldvr,
                         const magma_int_t *mm, magma_int_t *m,
                         double *work,
                         #ifdef COMPLEX
                         double *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_dtrevc3( const char *side, const char *howmny,
                          magma_int_t *select, const magma_int_t *n,
                          double *T,  const magma_int_t *ldt,
                          double *VL, const magma_int_t *ldvl, 
                          double *VR, const magma_int_t *ldvr,
                          const magma_int_t *mm,
                          const magma_int_t *mout,
                          double *work, const magma_int_t *lwork,
                          #ifdef COMPLEX
                          double *rwork,
                          #endif
                          magma_int_t *info );

void   lapackf77_dtrtri( const char *uplo, const char *diag,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_dorg2r( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         double *A, const magma_int_t *lda,
                         const double *tau,
                         double *work,
                         magma_int_t *info );

void   lapackf77_dorgbr( const char *vect,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         double *A, const magma_int_t *lda,
                         const double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dorghr( const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         double *A, const magma_int_t *lda,
                         const double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dorglq( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         double *A, const magma_int_t *lda,
                         const double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dorgql( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         double *A, const magma_int_t *lda,
                         const double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dorgqr( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         double *A, const magma_int_t *lda,
                         const double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dorgtr( const char *uplo,
                         const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         const double *tau,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dorm2r( const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const double *A, const magma_int_t *lda,
                         const double *tau,
                         double *C, const magma_int_t *ldc,
                         double *work,
                         magma_int_t *info );

void   lapackf77_dormbr( const char *vect, const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const double *A, const magma_int_t *lda,
                         const double *tau,
                         double *C, const magma_int_t *ldc,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dormlq( const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const double *A, const magma_int_t *lda,
                         const double *tau,
                         double *C, const magma_int_t *ldc,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dormql( const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const double *A, const magma_int_t *lda,
                         const double *tau,
                         double *C, const magma_int_t *ldc,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dormqr( const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const double *A, const magma_int_t *lda,
                         const double *tau,
                         double *C, const magma_int_t *ldc,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_dormtr( const char *side, const char *uplo, const char *trans,
                         const magma_int_t *m, const magma_int_t *n,
                         const double *A, const magma_int_t *lda,
                         const double *tau,
                         double *C, const magma_int_t *ldc,
                         double *work, const magma_int_t *lwork,
                         magma_int_t *info );

/*
 * Real precision extras
 */
void   lapackf77_dstebz( const char *range, const char *order,
                         const magma_int_t *n,
                         double *vl, double *vu,
                         magma_int_t *il, magma_int_t *iu,
                         double *abstol,
                         double *d, double *e,
                         const magma_int_t *m, const magma_int_t *nsplit,
                         double *w,
                         magma_int_t *iblock, magma_int_t *isplit,
                         double *work,
                         magma_int_t *iwork,
                         magma_int_t *info );

void   lapackf77_dlaln2( const magma_int_t *ltrans,
                         const magma_int_t *na, const magma_int_t *nw,
                         const double *smin, const double *ca,
                         const double *a,  const magma_int_t *lda,
                         const double *d1, const double *d2,
                         const double *b,  const magma_int_t *ldb,
                         const double *wr, const double *wi,
                         double *x, const magma_int_t *ldx,
                         double *scale, double *xnorm, magma_int_t *info );

double lapackf77_dlamc3( double *a, double *b );

void   lapackf77_dlamrg( magma_int_t *n1, magma_int_t *n2,
                         double *a,
                         magma_int_t *dtrd1, magma_int_t *dtrd2, magma_int_t *index );

double lapackf77_dlapy3( double *x, double *y, double *z );

void   lapackf77_dlaed2( magma_int_t *k, magma_int_t *n, magma_int_t *cutpnt,
                         double *d, double *q, magma_int_t *ldq, magma_int_t *indxq,
                         double *rho, double *z,
                         double *dlmda, double *w, double *q2,
                         magma_int_t *indx, magma_int_t *indxc, magma_int_t *indxp,
                         magma_int_t *coltyp, magma_int_t *info);

void   lapackf77_dlaed4( magma_int_t *n, magma_int_t *i,
                         double *d,
                         double *z,
                         double *delta,
                         double *rho,
                         double *dlam, magma_int_t *info );

void   lapackf77_dlasrt( const char *id, const magma_int_t *n, double *d, magma_int_t *info );

/*
 * Testing functions
 */
#ifdef COMPLEX
void   lapackf77_dbdt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *kd,
                         double *A, const magma_int_t *lda,
                         double *Q, const magma_int_t *ldq,
                         double *d, double *e,
                         double *Pt, const magma_int_t *ldpt,
                         double *work,
                         double *rwork,
                         double *resid );

void   lapackf77_dget22( const char *transa, const char *transe, const char *transw, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *E, const magma_int_t *lde,
                         double *w,
                         double *work,
                         double *rwork,
                         double *result );

void   lapackf77_dsyt21( const magma_int_t *itype, const char *uplo,
                         const magma_int_t *n, const magma_int_t *kband,
                         double *A, const magma_int_t *lda,
                         double *d, double *e,
                         double *U, const magma_int_t *ldu,
                         double *V, const magma_int_t *ldv,
                         double *tau,
                         double *work,
                         double *rwork,
                         double *result );

void   lapackf77_dhst01( const magma_int_t *n, const magma_int_t *ilo, const magma_int_t *ihi,
                         double *A, const magma_int_t *lda,
                         double *H, const magma_int_t *ldh,
                         double *Q, const magma_int_t *ldq,
                         double *work, const magma_int_t *lwork,
                         double *rwork,
                         double *result );

void   lapackf77_dstt21( const magma_int_t *n, const magma_int_t *kband,
                         double *AD,
                         double *AE,
                         double *SD,
                         double *SE,
                         double *U, const magma_int_t *ldu,
                         double *work,
                         double *rwork,
                         double *result );

void   lapackf77_dort01( const char *rowcol, const magma_int_t *m, const magma_int_t *n,
                         double *U, const magma_int_t *ldu,
                         double *work, const magma_int_t *lwork,
                         double *rwork,
                         double *resid );
#else
void   lapackf77_dbdt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *kd,
                         double *A, const magma_int_t *lda,
                         double *Q, const magma_int_t *ldq,
                         double *d, double *e,
                         double *Pt, const magma_int_t *ldpt,
                         double *work,
                         double *resid );

void   lapackf77_dget22( const char *transa, const char *transe, const char *transw, const magma_int_t *n,
                         double *A, const magma_int_t *lda,
                         double *E, const magma_int_t *lde,
                         double *wr,
                         double *wi,
                         double *work,
                         double *result );

void   lapackf77_dsyt21( magma_int_t *itype, const char *uplo, const magma_int_t *n, const magma_int_t *kband,
                         double *A, const magma_int_t *lda,
                         double *d, double *e,
                         double *U, const magma_int_t *ldu,
                         double *V, const magma_int_t *ldv,
                         double *tau,
                         double *work,
                         double *result );

void   lapackf77_dhst01( const magma_int_t *n, const magma_int_t *ilo, const magma_int_t *ihi,
                         double *A, const magma_int_t *lda,
                         double *H, const magma_int_t *ldh,
                         double *Q, const magma_int_t *ldq,
                         double *work, const magma_int_t *lwork,
                         double *result );

void   lapackf77_dstt21( const magma_int_t *n, const magma_int_t *kband,
                         double *AD,
                         double *AE,
                         double *SD,
                         double *SE,
                         double *U, const magma_int_t *ldu,
                         double *work,
                         double *result );

void   lapackf77_dort01( const char *rowcol, const magma_int_t *m, const magma_int_t *n,
                         double *U, const magma_int_t *ldu,
                         double *work, const magma_int_t *lwork,
                         double *resid );
#endif

void   lapackf77_dlarfy( const char *uplo, const magma_int_t *n,
                         double *V, const magma_int_t *incv,
                         double *tau,
                         double *C, const magma_int_t *ldc,
                         double *work );

void   lapackf77_dlarfx( const char *side, const magma_int_t *m, const magma_int_t *n,
                         double *V,
                         double *tau,
                         double *C, const magma_int_t *ldc,
                         double *work );

double lapackf77_dqpt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         double *A,
                         double *Af, const magma_int_t *lda,
                         double *tau, magma_int_t *jpvt,
                         double *work, const magma_int_t *lwork );

void   lapackf77_dqrt02( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         double *A,
                         double *AF,
                         double *Q,
                         double *R, const magma_int_t *lda,
                         double *tau,
                         double *work, const magma_int_t *lwork,
                         double *rwork,
                         double *result );

void   lapackf77_dlatms( magma_int_t *m, magma_int_t *n,
                         const char *dist, magma_int_t *iseed, const char *sym, double *d,
                         magma_int_t *mode, const double *cond, const double *dmax,
                         magma_int_t *kl, magma_int_t *ku, const char *pack,
                         double *a, magma_int_t *lda, double *work, magma_int_t *info );

#ifdef __cplusplus
}
#endif

#undef REAL

#endif /* MAGMA_DLAPACK_H */
