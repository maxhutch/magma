/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_zlapack.h normal z -> s, Fri Jan 30 19:00:06 2015
*/

#ifndef MAGMA_SLAPACK_H
#define MAGMA_SLAPACK_H

#include "magma_types.h"
#include "magma_mangling.h"

#define REAL

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- BLAS and LAPACK functions (alphabetical order)
*/
#define blasf77_isamax     FORTRAN_NAME( isamax, ISAMAX )
#define blasf77_saxpy      FORTRAN_NAME( saxpy,  SAXPY  )
#define blasf77_scopy      FORTRAN_NAME( scopy,  SCOPY  )
#define blasf77_sgemm      FORTRAN_NAME( sgemm,  SGEMM  )
#define blasf77_sgemv      FORTRAN_NAME( sgemv,  SGEMV  )
#define blasf77_sger      FORTRAN_NAME( sger,  SGER  )
#define blasf77_sger      FORTRAN_NAME( sger,  SGER  )
#define blasf77_ssymm      FORTRAN_NAME( ssymm,  SSYMM  )
#define blasf77_ssymv      FORTRAN_NAME( ssymv,  SSYMV  )
#define blasf77_ssyr       FORTRAN_NAME( ssyr,   SSYR   )
#define blasf77_ssyr2      FORTRAN_NAME( ssyr2,  SSYR2  )
#define blasf77_ssyr2k     FORTRAN_NAME( ssyr2k, SSYR2K )
#define blasf77_ssyrk      FORTRAN_NAME( ssyrk,  SSYRK  )
#define blasf77_sscal      FORTRAN_NAME( sscal,  SSCAL  )
#define blasf77_sscal     FORTRAN_NAME( sscal, SSCAL )
#define blasf77_sswap      FORTRAN_NAME( sswap,  SSWAP  )
#define blasf77_ssymm      FORTRAN_NAME( ssymm,  SSYMM  )
#define blasf77_ssyr2k     FORTRAN_NAME( ssyr2k, SSYR2K )
#define blasf77_ssyrk      FORTRAN_NAME( ssyrk,  SSYRK  )
#define blasf77_srotg      FORTRAN_NAME( srotg,  SROTG  )
#define blasf77_srot       FORTRAN_NAME( srot,   SROT   )
#define blasf77_srot      FORTRAN_NAME( srot,  SROT  )
#define blasf77_strmm      FORTRAN_NAME( strmm,  STRMM  )
#define blasf77_strmv      FORTRAN_NAME( strmv,  STRMV  )
#define blasf77_strsm      FORTRAN_NAME( strsm,  STRSM  )
#define blasf77_strsv      FORTRAN_NAME( strsv,  STRSV  )

#define lapackf77_slaed2   FORTRAN_NAME( slaed2, SLAED2 )
#define lapackf77_slaed4   FORTRAN_NAME( slaed4, SLAED4 )
#define lapackf77_slaln2   FORTRAN_NAME( slaln2, SLALN2 )
#define lapackf77_slamc3   FORTRAN_NAME( slamc3, SLAMC3 )
#define lapackf77_slamrg   FORTRAN_NAME( slamrg, SLAMRG )
#define lapackf77_slasrt   FORTRAN_NAME( slasrt, SLASRT )
#define lapackf77_sstebz   FORTRAN_NAME( sstebz, SSTEBZ )

#define lapackf77_sbdsdc   FORTRAN_NAME( sbdsdc, SBDSDC )
#define lapackf77_sbdsqr   FORTRAN_NAME( sbdsqr, SBDSQR )
#define lapackf77_sgebak   FORTRAN_NAME( sgebak, SGEBAK )
#define lapackf77_sgebal   FORTRAN_NAME( sgebal, SGEBAL )
#define lapackf77_sgebd2   FORTRAN_NAME( sgebd2, SGEBD2 )
#define lapackf77_sgebrd   FORTRAN_NAME( sgebrd, SGEBRD )
#define lapackf77_sgbbrd   FORTRAN_NAME( sgbbrd, SGBBRD )
#define lapackf77_sgeev    FORTRAN_NAME( sgeev,  SGEEV  )
#define lapackf77_sgehd2   FORTRAN_NAME( sgehd2, SGEHD2 )
#define lapackf77_sgehrd   FORTRAN_NAME( sgehrd, SGEHRD )
#define lapackf77_sgelqf   FORTRAN_NAME( sgelqf, SGELQF )
#define lapackf77_sgels    FORTRAN_NAME( sgels,  SGELS  )
#define lapackf77_sgeqlf   FORTRAN_NAME( sgeqlf, SGEQLF )
#define lapackf77_sgeqp3   FORTRAN_NAME( sgeqp3, SGEQP3 )
#define lapackf77_sgeqrf   FORTRAN_NAME( sgeqrf, SGEQRF )
#define lapackf77_sgesdd   FORTRAN_NAME( sgesdd, SGESDD )
#define lapackf77_sgesv    FORTRAN_NAME( sgesv,  SGESV  )
#define lapackf77_sgesvd   FORTRAN_NAME( sgesvd, SGESVD )
#define lapackf77_sgetrf   FORTRAN_NAME( sgetrf, SGETRF )
#define lapackf77_sgetri   FORTRAN_NAME( sgetri, SGETRI )
#define lapackf77_sgetrs   FORTRAN_NAME( sgetrs, SGETRS )
#define lapackf77_ssytf2   FORTRAN_NAME( ssytf2, SSYTF2 )
#define lapackf77_ssytrs   FORTRAN_NAME( ssytrs, SSYTRS )
#define lapackf77_ssbtrd   FORTRAN_NAME( ssbtrd, SSBTRD )
#define lapackf77_ssyev    FORTRAN_NAME( ssyev,  SSYEV  )
#define lapackf77_ssyevd   FORTRAN_NAME( ssyevd, SSYEVD )
#define lapackf77_ssyevr   FORTRAN_NAME( ssyevr, SSYEVR )
#define lapackf77_ssyevx   FORTRAN_NAME( ssyevx, SSYEVX )
#define lapackf77_ssygs2   FORTRAN_NAME( ssygs2, SSYGS2 )
#define lapackf77_ssygst   FORTRAN_NAME( ssygst, SSYGST )
#define lapackf77_ssygvd   FORTRAN_NAME( ssygvd, SSYGVD )
#define lapackf77_ssysv    FORTRAN_NAME( ssysv,  SSYSV  )
#define lapackf77_ssytd2   FORTRAN_NAME( ssytd2, SSYTD2 )
#define lapackf77_ssytrd   FORTRAN_NAME( ssytrd, SSYTRD )
#define lapackf77_ssytrf   FORTRAN_NAME( ssytrf, SSYTRF )
#define lapackf77_shseqr   FORTRAN_NAME( shseqr, SHSEQR )
#define lapackf77_slabrd   FORTRAN_NAME( slabrd, SLABRD )
#define lapackf77_slacgv   FORTRAN_NAME( slacgv, SLACGV )
#define lapackf77_slacp2   FORTRAN_NAME( slacp2, SLACP2 )
#define lapackf77_slacpy   FORTRAN_NAME( slacpy, SLACPY )
#define lapackf77_slacrm   FORTRAN_NAME( slacrm, SLACRM )
#define lapackf77_sladiv   FORTRAN_NAME( sladiv, SLADIV )
#define lapackf77_slasyf   FORTRAN_NAME( slasyf, SLASYF )
#define lapackf77_slange   FORTRAN_NAME( slange, SLANGE )
#define lapackf77_slansy   FORTRAN_NAME( slansy, SLANSY )
#define lapackf77_slanst   FORTRAN_NAME( slanst, SLANST )
#define lapackf77_slansy   FORTRAN_NAME( slansy, SLANSY )
#define lapackf77_slantr   FORTRAN_NAME( slantr, SLANTR )
#define lapackf77_slapy3   FORTRAN_NAME( slapy3, SLAPY3 )
#define lapackf77_slaqp2   FORTRAN_NAME( slaqp2, SLAQP2 )
#define lapackf77_slarcm   FORTRAN_NAME( slarcm, SLARCM )
#define lapackf77_slarf    FORTRAN_NAME( slarf,  SLARF  )
#define lapackf77_slarfb   FORTRAN_NAME( slarfb, SLARFB )
#define lapackf77_slarfg   FORTRAN_NAME( slarfg, SLARFG )
#define lapackf77_slarft   FORTRAN_NAME( slarft, SLARFT )
#define lapackf77_slarnv   FORTRAN_NAME( slarnv, SLARNV )
#define lapackf77_slartg   FORTRAN_NAME( slartg, SLARTG )
#define lapackf77_slascl   FORTRAN_NAME( slascl, SLASCL )
#define lapackf77_slaset   FORTRAN_NAME( slaset, SLASET )
#define lapackf77_slaswp   FORTRAN_NAME( slaswp, SLASWP )
#define lapackf77_slatrd   FORTRAN_NAME( slatrd, SLATRD )
#define lapackf77_slatrs   FORTRAN_NAME( slatrs, SLATRS )
#define lapackf77_slauum   FORTRAN_NAME( slauum, SLAUUM )
#define lapackf77_slavsy   FORTRAN_NAME( slavsy, SLAVSY )
#define lapackf77_sposv    FORTRAN_NAME( sposv,  SPOSV  )
#define lapackf77_spotrf   FORTRAN_NAME( spotrf, SPOTRF )
#define lapackf77_spotri   FORTRAN_NAME( spotri, SPOTRI )
#define lapackf77_spotrs   FORTRAN_NAME( spotrs, SPOTRS )
#define lapackf77_sstedc   FORTRAN_NAME( sstedc, SSTEDC )
#define lapackf77_sstein   FORTRAN_NAME( sstein, SSTEIN )
#define lapackf77_sstemr   FORTRAN_NAME( sstemr, SSTEMR )
#define lapackf77_ssteqr   FORTRAN_NAME( ssteqr, SSTEQR )
#define lapackf77_ssymv    FORTRAN_NAME( ssymv,  SSYMV  )
#define lapackf77_strevc   FORTRAN_NAME( strevc, STREVC )
#define lapackf77_strevc3  FORTRAN_NAME( strevc3, STREVC3 )
#define lapackf77_strtri   FORTRAN_NAME( strtri, STRTRI )
#define lapackf77_sorg2r   FORTRAN_NAME( sorg2r, SORG2R )
#define lapackf77_sorgbr   FORTRAN_NAME( sorgbr, SORGBR )
#define lapackf77_sorghr   FORTRAN_NAME( sorghr, SORGHR )
#define lapackf77_sorglq   FORTRAN_NAME( sorglq, SORGLQ )
#define lapackf77_sorgql   FORTRAN_NAME( sorgql, SORGQL )
#define lapackf77_sorgqr   FORTRAN_NAME( sorgqr, SORGQR )
#define lapackf77_sorgtr   FORTRAN_NAME( sorgtr, SORGTR )
#define lapackf77_sorm2r   FORTRAN_NAME( sorm2r, SORM2R )
#define lapackf77_sormbr   FORTRAN_NAME( sormbr, SORMBR )
#define lapackf77_sormlq   FORTRAN_NAME( sormlq, SORMLQ )
#define lapackf77_sormql   FORTRAN_NAME( sormql, SORMQL )
#define lapackf77_sormqr   FORTRAN_NAME( sormqr, SORMQR )
#define lapackf77_sormtr   FORTRAN_NAME( sormtr, SORMTR )

/* testing functions (alphabetical order) */
#define lapackf77_sbdt01   FORTRAN_NAME( sbdt01, SBDT01 )
#define lapackf77_sget22   FORTRAN_NAME( sget22, SGET22 )
#define lapackf77_ssyt21   FORTRAN_NAME( ssyt21, SSYT21 )
#define lapackf77_shst01   FORTRAN_NAME( shst01, SHST01 )
#define lapackf77_slarfx   FORTRAN_NAME( slarfx, SLARFX )
#define lapackf77_slarfy   FORTRAN_NAME( slarfy, SLARFY )
#define lapackf77_slatms   FORTRAN_NAME( slatms, SLATMS )
#define lapackf77_sqpt01   FORTRAN_NAME( sqpt01, SQPT01 )
#define lapackf77_sqrt02   FORTRAN_NAME( sqrt02, SQRT02 )
#define lapackf77_sstt21   FORTRAN_NAME( sstt21, SSTT21 )
#define lapackf77_sort01   FORTRAN_NAME( sort01, SORT01 )

/*
 * BLAS functions (alphabetical order)
 */
magma_int_t blasf77_isamax(
                     const magma_int_t *n,
                     const float *x, const magma_int_t *incx );

void blasf77_saxpy(  const magma_int_t *n,
                     const float *alpha,
                     const float *x, const magma_int_t *incx,
                           float *y, const magma_int_t *incy );

void blasf77_scopy(  const magma_int_t *n,
                     const float *x, const magma_int_t *incx,
                           float *y, const magma_int_t *incy );

void blasf77_sgemm(  const char *transa, const char *transb,
                     const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                     const float *B, const magma_int_t *ldb,
                     const float *beta,
                           float *C, const magma_int_t *ldc );

void blasf77_sgemv(  const char *transa,
                     const magma_int_t *m, const magma_int_t *n,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                     const float *x, const magma_int_t *incx,
                     const float *beta,
                           float *y, const magma_int_t *incy );

void blasf77_sger(  const magma_int_t *m, const magma_int_t *n,
                     const float *alpha,
                     const float *x, const magma_int_t *incx,
                     const float *y, const magma_int_t *incy,
                           float *A, const magma_int_t *lda );

void blasf77_sger(  const magma_int_t *m, const magma_int_t *n,
                     const float *alpha,
                     const float *x, const magma_int_t *incx,
                     const float *y, const magma_int_t *incy,
                           float *A, const magma_int_t *lda );

void blasf77_ssymm(  const char *side, const char *uplo,
                     const magma_int_t *m, const magma_int_t *n,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                     const float *B, const magma_int_t *ldb,
                     const float *beta,
                           float *C, const magma_int_t *ldc );

void blasf77_ssymv(  const char *uplo,
                     const magma_int_t *n,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                     const float *x, const magma_int_t *incx,
                     const float *beta,
                           float *y, const magma_int_t *incy );

void blasf77_ssyr(   const char *uplo,
                     const magma_int_t *n,
                     const float *alpha,
                     const float *x, const magma_int_t *incx,
                           float *A, const magma_int_t *lda );

void blasf77_ssyr2(  const char *uplo,
                     const magma_int_t *n,
                     const float *alpha,
                     const float *x, const magma_int_t *incx,
                     const float *y, const magma_int_t *incy,
                           float *A, const magma_int_t *lda );

void blasf77_ssyr2k( const char *uplo, const char *trans,
                     const magma_int_t *n, const magma_int_t *k,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                     const float *B, const magma_int_t *ldb,
                     const float *beta,
                           float *C, const magma_int_t *ldc );

void blasf77_ssyrk(  const char *uplo, const char *trans,
                     const magma_int_t *n, const magma_int_t *k,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                     const float *beta,
                           float *C, const magma_int_t *ldc );

void blasf77_sscal(  const magma_int_t *n,
                     const float *alpha,
                           float *x, const magma_int_t *incx );

void blasf77_sscal( const magma_int_t *n,
                     const float *alpha,
                           float *x, const magma_int_t *incx );

void blasf77_sswap(  const magma_int_t *n,
                     float *x, const magma_int_t *incx,
                     float *y, const magma_int_t *incy );

/* real-symmetric (non-symmetric) routines */
void blasf77_ssymm(  const char *side, const char *uplo,
                     const magma_int_t *m, const magma_int_t *n,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                     const float *B, const magma_int_t *ldb,
                     const float *beta,
                           float *C, const magma_int_t *ldc );

void blasf77_ssyr2k( const char *uplo, const char *trans,
                     const magma_int_t *n, const magma_int_t *k,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                     const float *B, const magma_int_t *ldb,
                     const float *beta,
                           float *C, const magma_int_t *ldc );

void blasf77_ssyrk(  const char *uplo, const char *trans,
                     const magma_int_t *n, const magma_int_t *k,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                     const float *beta,
                           float *C, const magma_int_t *ldc );

void blasf77_srotg(  float *ca, const float *cb,
                     float *c, float *s );
                     
void blasf77_srot(   const magma_int_t *n,
                     float *x, const magma_int_t *incx,
                     float *y, const magma_int_t *incy,
                     const float *c, const float *s );
                     
void blasf77_srot(  const magma_int_t *n,
                     float *x, const magma_int_t *incx,
                     float *y, const magma_int_t *incy,
                     const float *c, const float *s );

void blasf77_strmm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const magma_int_t *m, const magma_int_t *n,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                           float *B, const magma_int_t *ldb );

void blasf77_strmv(  const char *uplo, const char *transa, const char *diag,
                     const magma_int_t *n,
                     const float *A, const magma_int_t *lda,
                           float *x, const magma_int_t *incx );

void blasf77_strsm(  const char *side, const char *uplo, const char *transa, const char *diag,
                     const magma_int_t *m, const magma_int_t *n,
                     const float *alpha,
                     const float *A, const magma_int_t *lda,
                           float *B, const magma_int_t *ldb );

void blasf77_strsv(  const char *uplo, const char *transa, const char *diag,
                     const magma_int_t *n,
                     const float *A, const magma_int_t *lda,
                           float *x, const magma_int_t *incx );

/* ////////////////////////////////////////////////////////////////////////////
 -- MAGMA wrappers around BLAS functions (alphabetical order)
    The Fortran interface for these is not portable, so we
    provide a C interface identical to the Fortran interface.
*/

float magma_cblas_sasum(
    magma_int_t n,
    const float *x, magma_int_t incx );

float magma_cblas_snrm2(
    magma_int_t n,
    const float *x, magma_int_t incx );

float magma_cblas_sdot(
    magma_int_t n,
    const float *x, magma_int_t incx,
    const float *y, magma_int_t incy );

float magma_cblas_sdot(
    magma_int_t n,
    const float *x, magma_int_t incx,
    const float *y, magma_int_t incy );


/*
 * LAPACK functions (alphabetical order)
 */
#ifdef REAL
void   lapackf77_sbdsdc( const char *uplo, const char *compq,
                         const magma_int_t *n,
                         float *d, float *e,
                         float *U,  const magma_int_t *ldu,
                         float *VT, const magma_int_t *ldvt,
                         float *Q, magma_int_t *IQ,
                         float *work, magma_int_t *iwork,
                         magma_int_t *info );
#endif

void   lapackf77_sbdsqr( const char *uplo,
                         const magma_int_t *n, const magma_int_t *ncvt, const magma_int_t *nru,  const magma_int_t *ncc,
                         float *d, float *e,
                         float *Vt, const magma_int_t *ldvt,
                         float *U, const magma_int_t *ldu,
                         float *C, const magma_int_t *ldc,
                         float *work,
                         magma_int_t *info );

void   lapackf77_sgebak( const char *job, const char *side,
                         const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         const float *scale, const magma_int_t *m,
                         float *V, const magma_int_t *ldv,
                         magma_int_t *info );

void   lapackf77_sgebal( const char *job,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         magma_int_t *ilo, magma_int_t *ihi,
                         float *scale,
                         magma_int_t *info );

void   lapackf77_sgebd2( const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *d, float *e,
                         float *tauq,
                         float *taup,
                         float *work,
                         magma_int_t *info );

void   lapackf77_sgebrd( const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *d, float *e,
                         float *tauq,
                         float *taup,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sgbbrd( const char *vect, const magma_int_t *m,
                         const magma_int_t *n, const magma_int_t *ncc,
                         const magma_int_t *kl, const magma_int_t *ku,
                         float *Ab, const magma_int_t *ldab,
                         float *d, float *e,
                         float *Q, const magma_int_t *ldq,
                         float *PT, const magma_int_t *ldpt,
                         float *C, const magma_int_t *ldc,
                         float *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_sgeev(  const char *jobvl, const char *jobvr,
                         const magma_int_t *n,
                         float *A,    const magma_int_t *lda,
                         #ifdef COMPLEX
                         float *w,
                         #else
                         float *wr, float *wi,
                         #endif
                         float *Vl,   const magma_int_t *ldvl,
                         float *Vr,   const magma_int_t *ldvr,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_sgehd2( const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         float *A, const magma_int_t *lda,
                         float *tau,
                         float *work,
                         magma_int_t *info );

void   lapackf77_sgehrd( const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         float *A, const magma_int_t *lda,
                         float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sgelqf( const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sgels(  const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *nrhs,
                         float *A, const magma_int_t *lda,
                         float *B, const magma_int_t *ldb,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sgeqlf( const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sgeqp3( const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         magma_int_t *jpvt,
                         float *tau,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_sgeqrf( const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sgesdd( const char *jobz,
                         const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *s,
                         float *U,  const magma_int_t *ldu,
                         float *Vt, const magma_int_t *ldvt,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         magma_int_t *iwork, magma_int_t *info );

void   lapackf77_sgesv(  const magma_int_t *n, const magma_int_t *nrhs,
                         float *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         float *B,  const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_sgesvd( const char *jobu, const char *jobvt,
                         const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *s,
                         float *U,  const magma_int_t *ldu,
                         float *Vt, const magma_int_t *ldvt,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_sgetrf( const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         magma_int_t *info );

void   lapackf77_sgetri( const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         const magma_int_t *ipiv,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sgetrs( const char *trans,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         const float *A, const magma_int_t *lda,
                         const magma_int_t *ipiv,
                         float *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_ssytf2( const char*, magma_int_t*, 
                         float*, magma_int_t*, magma_int_t*, magma_int_t* );

void   lapackf77_ssytrs( const char *uplo,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         const float *A, const magma_int_t *lda,
                         const magma_int_t *ipiv,
                         float *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_ssbtrd( const char *vect, const char *uplo,
                         const magma_int_t *n, const magma_int_t *kd,
                         float *Ab, const magma_int_t *ldab,
                         float *d, float *e,
                         float *Q, const magma_int_t *ldq,
                         float *work,
                         magma_int_t *info );

void   lapackf77_ssyev(  const char *jobz, const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *w,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_ssyevd( const char *jobz, const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *w,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork, const magma_int_t *lrwork,
                         #endif
                         magma_int_t *iwork, const magma_int_t *liwork,
                         magma_int_t *info );

void   lapackf77_ssyevr( const char *jobz, const char *range, const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *vl, float *vu, 
                         magma_int_t *il, magma_int_t *iu, float *abstol, 
                         magma_int_t *m, float *w, float *z__, 
                         magma_int_t *ldz, magma_int_t *isuppz, 
                         float *work, magma_int_t *lwork, 
                         float *rwork, magma_int_t *lrwork, 
                         magma_int_t *iwork, magma_int_t *liwork, magma_int_t *info);

void   lapackf77_ssyevx( const char *jobz, const char *range, const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *vl, float *vu,
                         magma_int_t *il, magma_int_t *iu, float *abstol,
                         magma_int_t *m, float *w, float *z__,
                         magma_int_t *ldz, float *work, magma_int_t *lwork,
                         float *rwork, magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info);

void   lapackf77_ssygs2( const magma_int_t *itype, const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_ssygst( const magma_int_t *itype, const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_ssygvd( const magma_int_t *itype, const char *jobz, const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *B, const magma_int_t *ldb,
                         float *w,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork, const magma_int_t *lrwork,
                         #endif
                         magma_int_t *iwork, const magma_int_t *liwork,
                         magma_int_t *info );

void   lapackf77_ssysv( const char *uplo, 
                        const magma_int_t *n, const magma_int_t *nrhs,
                        float *A, const magma_int_t *lda, magma_int_t *ipiv,
                        float *B, const magma_int_t *ldb,
                        float *work, const magma_int_t *lwork,
                        magma_int_t *info );

void   lapackf77_ssytd2( const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *d, float *e,
                         float *tau,
                         magma_int_t *info );

void   lapackf77_ssytrd( const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *d, float *e,
                         float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_ssytrf( const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         magma_int_t *ipiv,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_shseqr( const char *job, const char *compz,
                         const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         float *H, const magma_int_t *ldh,
                         #ifdef COMPLEX
                         float *w,
                         #else
                         float *wr, float *wi,
                         #endif
                         float *Z, const magma_int_t *ldz,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_slabrd( const magma_int_t *m, const magma_int_t *n, const magma_int_t *nb,
                         float *A, const magma_int_t *lda,
                         float *d, float *e,
                         float *tauq,
                         float *taup,
                         float *X, const magma_int_t *ldx,
                         float *Y, const magma_int_t *ldy );

#ifdef COMPLEX
void   lapackf77_slacgv( const magma_int_t *n,
                         float *x, const magma_int_t *incx );
#endif

#ifdef COMPLEX
void   lapackf77_slacp2( const char *uplo,
                         const magma_int_t *m, const magma_int_t *n,
                         const float *A, const magma_int_t *lda,
                         float *B, const magma_int_t *ldb );
#endif

void   lapackf77_slacpy( const char *uplo,
                         const magma_int_t *m, const magma_int_t *n,
                         const float *A, const magma_int_t *lda,
                         float *B, const magma_int_t *ldb );

#ifdef COMPLEX
void   lapackf77_slacrm( const magma_int_t *m, const magma_int_t *n,
                         const float *A, const magma_int_t *lda,
                         const float             *B, const magma_int_t *ldb,
                         float       *C, const magma_int_t *ldc,
                         float *rwork );
#endif

#ifdef COMPLEX
void   lapackf77_sladiv( float *ret_val,
                         float *x,
                         float *y );
#else // REAL
void   lapackf77_sladiv( const float *a, const float *b,
                         const float *c, const float *d,
                         float *p, float *q );
#endif

void   lapackf77_slasyf( const char *uplo,
                         const magma_int_t *n, const magma_int_t *kn,
                         magma_int_t *kb,
                         float *A, const magma_int_t lda,
                         magma_int_t *ipiv,
                         float *work, const magma_int_t *ldwork,
                         magma_int_t *info );

float lapackf77_slange( const char *norm,
                         const magma_int_t *m, const magma_int_t *n,
                         const float *A, const magma_int_t *lda,
                         float *work );

float lapackf77_slansy( const char *norm, const char *uplo,
                         const magma_int_t *n,
                         const float *A, const magma_int_t *lda,
                         float *work );

float lapackf77_slanst( const char *norm, const magma_int_t *n,
                         const float *d, const float *e );

float lapackf77_slansy( const char *norm, const char *uplo,
                         const magma_int_t *n,
                         const float *A, const magma_int_t *lda,
                         float *work );

float lapackf77_slantr( const char *norm, const char *uplo, const char *diag,
                         const magma_int_t *m, const magma_int_t *n,
                         const float *A, const magma_int_t *lda,
                         float *work );

void   lapackf77_slaqp2( magma_int_t *m, magma_int_t *n, magma_int_t *offset,
                         float *a, magma_int_t *lda, magma_int_t *jpvt,
                         float *tau,
                         float *vn1, float *vn2,
                         float *work );

#ifdef COMPLEX
void   lapackf77_slarcm( const magma_int_t *m, const magma_int_t *n,
                         const float             *A, const magma_int_t *lda,
                         const float *B, const magma_int_t *ldb,
                         float       *C, const magma_int_t *ldc,
                         float *rwork );
#endif

void   lapackf77_slarf(  const char *side, const magma_int_t *m, const magma_int_t *n,
                         float *v, const magma_int_t *incv,
                         float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work );

void   lapackf77_slarfb( const char *side, const char *trans, const char *direct, const char *storev,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const float *V, const magma_int_t *ldv,
                         const float *T, const magma_int_t *ldt,
                         float *C, const magma_int_t *ldc,
                         float *work, const magma_int_t *ldwork );

void   lapackf77_slarfg( const magma_int_t *n,
                         float *alpha,
                         float *x, const magma_int_t *incx,
                         float *tau );

void   lapackf77_slarft( const char *direct, const char *storev,
                         const magma_int_t *n, const magma_int_t *k,
                         float *V, const magma_int_t *ldv,
                         const float *tau,
                         float *T, const magma_int_t *ldt );

void   lapackf77_slarnv( const magma_int_t *idist, magma_int_t *iseed, const magma_int_t *n,
                         float *x );

void   lapackf77_slartg( float *F,
                         float *G,
                         float *cs,
                         float *SN,
                         float *R );

void   lapackf77_slascl( const char *type,
                         const magma_int_t *kl, const magma_int_t *ku,
                         float *cfrom,
                         float *cto,
                         const magma_int_t *m, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_slaset( const char *uplo,
                         const magma_int_t *m, const magma_int_t *n,
                         const float *alpha,
                         const float *beta,
                         float *A, const magma_int_t *lda );

void   lapackf77_slaswp( const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         const magma_int_t *k1, const magma_int_t *k2,
                         magma_int_t *ipiv,
                         const magma_int_t *incx );

void   lapackf77_slatrd( const char *uplo,
                         const magma_int_t *n, const magma_int_t *nb,
                         float *A, const magma_int_t *lda,
                         float *e,
                         float *tau,
                         float *work, const magma_int_t *ldwork );

void   lapackf77_slatrs( const char *uplo, const char *trans, const char *diag,
                         const char *normin,
                         const magma_int_t *n,
                         const float *A, const magma_int_t *lda,
                         float *x, float *scale,
                         float *cnorm, magma_int_t *info );

void   lapackf77_slauum( const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_slavsy( const char *uplo, const char *trans, const char *diag,
                         magma_int_t *n, magma_int_t *nrhs,
                         float *A, magma_int_t *lda,
                         magma_int_t *ipiv,
                         float *B, magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_sposv(  const char *uplo,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         float *A, const magma_int_t *lda,
                         float *B,  const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_spotrf( const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_spotri( const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_spotrs( const char *uplo,
                         const magma_int_t *n, const magma_int_t *nrhs,
                         const float *A, const magma_int_t *lda,
                         float *B, const magma_int_t *ldb,
                         magma_int_t *info );

void   lapackf77_sstedc( const char *compz,
                         const magma_int_t *n,
                         float *d, float *e,
                         float *Z, const magma_int_t *ldz,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork, const magma_int_t *lrwork,
                         #endif
                         magma_int_t *iwork, const magma_int_t *liwork,
                         magma_int_t *info );

void   lapackf77_sstein( const magma_int_t *n,
                         const float *d, const float *e,
                         const magma_int_t *m,
                         const float *w,
                         const magma_int_t *iblock,
                         const magma_int_t *isplit,
                         float *Z, const magma_int_t *ldz,
                         float *work, magma_int_t *iwork, magma_int_t *ifailv,
                         magma_int_t *info );

void   lapackf77_sstemr( const char *jobz, const char *range,
                         const magma_int_t *n,
                         float *d, float *e,
                         const float *vl, const float *vu,
                         const magma_int_t *il, const magma_int_t *iu,
                         magma_int_t *m,
                         float *w,
                         float *Z, const magma_int_t *ldz,
                         const magma_int_t *nzc, magma_int_t *isuppz, magma_int_t *tryrac,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *iwork, const magma_int_t *liwork,
                         magma_int_t *info );

void   lapackf77_ssteqr( const char *compz,
                         const magma_int_t *n,
                         float *d, float *e,
                         float *Z, const magma_int_t *ldz,
                         float *work,
                         magma_int_t *info );

#ifdef COMPLEX
void   lapackf77_ssymv(  const char *uplo,
                         const magma_int_t *n,
                         const float *alpha,
                         const float *A, const magma_int_t *lda,
                         const float *x, const magma_int_t *incx,
                         const float *beta,
                               float *y, const magma_int_t *incy );
#endif

void   lapackf77_strevc( const char *side, const char *howmny,
                         magma_int_t *select, const magma_int_t *n,
                         float *T,  const magma_int_t *ldt,
                         float *Vl, const magma_int_t *ldvl,
                         float *Vr, const magma_int_t *ldvr,
                         const magma_int_t *mm, magma_int_t *m,
                         float *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         magma_int_t *info );

void   lapackf77_strevc3( const char *side, const char *howmny,
                          magma_int_t *select, const magma_int_t *n,
                          float *T,  const magma_int_t *ldt,
                          float *VL, const magma_int_t *ldvl, 
                          float *VR, const magma_int_t *ldvr,
                          const magma_int_t *mm,
                          const magma_int_t *mout,
                          float *work, const magma_int_t *lwork,
                          #ifdef COMPLEX
                          float *rwork,
                          #endif
                          magma_int_t *info );

void   lapackf77_strtri( const char *uplo, const char *diag,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         magma_int_t *info );

void   lapackf77_sorg2r( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         float *A, const magma_int_t *lda,
                         const float *tau,
                         float *work,
                         magma_int_t *info );

void   lapackf77_sorgbr( const char *vect,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         float *A, const magma_int_t *lda,
                         const float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sorghr( const magma_int_t *n,
                         const magma_int_t *ilo, const magma_int_t *ihi,
                         float *A, const magma_int_t *lda,
                         const float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sorglq( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         float *A, const magma_int_t *lda,
                         const float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sorgql( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         float *A, const magma_int_t *lda,
                         const float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sorgqr( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         float *A, const magma_int_t *lda,
                         const float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sorgtr( const char *uplo,
                         const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         const float *tau,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sorm2r( const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const float *A, const magma_int_t *lda,
                         const float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work,
                         magma_int_t *info );

void   lapackf77_sormbr( const char *vect, const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const float *A, const magma_int_t *lda,
                         const float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sormlq( const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const float *A, const magma_int_t *lda,
                         const float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sormql( const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const float *A, const magma_int_t *lda,
                         const float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sormqr( const char *side, const char *trans,
                         const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         const float *A, const magma_int_t *lda,
                         const float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

void   lapackf77_sormtr( const char *side, const char *uplo, const char *trans,
                         const magma_int_t *m, const magma_int_t *n,
                         const float *A, const magma_int_t *lda,
                         const float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work, const magma_int_t *lwork,
                         magma_int_t *info );

/*
 * Real precision extras
 */
void   lapackf77_sstebz( const char *range, const char *order,
                         const magma_int_t *n,
                         float *vl, float *vu,
                         magma_int_t *il, magma_int_t *iu,
                         float *abstol,
                         float *d, float *e,
                         const magma_int_t *m, const magma_int_t *nsplit,
                         float *w,
                         magma_int_t *iblock, magma_int_t *isplit,
                         float *work,
                         magma_int_t *iwork,
                         magma_int_t *info );

void   lapackf77_slaln2( const magma_int_t *ltrans,
                         const magma_int_t *na, const magma_int_t *nw,
                         const float *smin, const float *ca,
                         const float *a,  const magma_int_t *lda,
                         const float *d1, const float *d2,
                         const float *b,  const magma_int_t *ldb,
                         const float *wr, const float *wi,
                         float *x, const magma_int_t *ldx,
                         float *scale, float *xnorm, magma_int_t *info );

float lapackf77_slamc3( float *a, float *b );

void   lapackf77_slamrg( magma_int_t *n1, magma_int_t *n2,
                         float *a,
                         magma_int_t *dtrd1, magma_int_t *dtrd2, magma_int_t *index );

float lapackf77_slapy3( float *x, float *y, float *z );

void   lapackf77_slaed2( magma_int_t *k, magma_int_t *n, magma_int_t *cutpnt,
                         float *d, float *q, magma_int_t *ldq, magma_int_t *indxq,
                         float *rho, float *z,
                         float *dlmda, float *w, float *q2,
                         magma_int_t *indx, magma_int_t *indxc, magma_int_t *indxp,
                         magma_int_t *coltyp, magma_int_t *info);

void   lapackf77_slaed4( magma_int_t *n, magma_int_t *i,
                         float *d,
                         float *z,
                         float *delta,
                         float *rho,
                         float *dlam, magma_int_t *info );

void   lapackf77_slasrt( const char *id, const magma_int_t *n, float *d, magma_int_t *info );

/*
 * Testing functions
 */
#ifdef COMPLEX
void   lapackf77_sbdt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *kd,
                         float *A, const magma_int_t *lda,
                         float *Q, const magma_int_t *ldq,
                         float *d, float *e,
                         float *Pt, const magma_int_t *ldpt,
                         float *work,
                         float *rwork,
                         float *resid );

void   lapackf77_sget22( const char *transa, const char *transe, const char *transw, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *E, const magma_int_t *lde,
                         float *w,
                         float *work,
                         float *rwork,
                         float *result );

void   lapackf77_ssyt21( const magma_int_t *itype, const char *uplo,
                         const magma_int_t *n, const magma_int_t *kband,
                         float *A, const magma_int_t *lda,
                         float *d, float *e,
                         float *U, const magma_int_t *ldu,
                         float *V, const magma_int_t *ldv,
                         float *tau,
                         float *work,
                         float *rwork,
                         float *result );

void   lapackf77_shst01( const magma_int_t *n, const magma_int_t *ilo, const magma_int_t *ihi,
                         float *A, const magma_int_t *lda,
                         float *H, const magma_int_t *ldh,
                         float *Q, const magma_int_t *ldq,
                         float *work, const magma_int_t *lwork,
                         float *rwork,
                         float *result );

void   lapackf77_sstt21( const magma_int_t *n, const magma_int_t *kband,
                         float *AD,
                         float *AE,
                         float *SD,
                         float *SE,
                         float *U, const magma_int_t *ldu,
                         float *work,
                         float *rwork,
                         float *result );

void   lapackf77_sort01( const char *rowcol, const magma_int_t *m, const magma_int_t *n,
                         float *U, const magma_int_t *ldu,
                         float *work, const magma_int_t *lwork,
                         float *rwork,
                         float *resid );
#else
void   lapackf77_sbdt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *kd,
                         float *A, const magma_int_t *lda,
                         float *Q, const magma_int_t *ldq,
                         float *d, float *e,
                         float *Pt, const magma_int_t *ldpt,
                         float *work,
                         float *resid );

void   lapackf77_sget22( const char *transa, const char *transe, const char *transw, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *E, const magma_int_t *lde,
                         float *wr,
                         float *wi,
                         float *work,
                         float *result );

void   lapackf77_ssyt21( magma_int_t *itype, const char *uplo, const magma_int_t *n, const magma_int_t *kband,
                         float *A, const magma_int_t *lda,
                         float *d, float *e,
                         float *U, const magma_int_t *ldu,
                         float *V, const magma_int_t *ldv,
                         float *tau,
                         float *work,
                         float *result );

void   lapackf77_shst01( const magma_int_t *n, const magma_int_t *ilo, const magma_int_t *ihi,
                         float *A, const magma_int_t *lda,
                         float *H, const magma_int_t *ldh,
                         float *Q, const magma_int_t *ldq,
                         float *work, const magma_int_t *lwork,
                         float *result );

void   lapackf77_sstt21( const magma_int_t *n, const magma_int_t *kband,
                         float *AD,
                         float *AE,
                         float *SD,
                         float *SE,
                         float *U, const magma_int_t *ldu,
                         float *work,
                         float *result );

void   lapackf77_sort01( const char *rowcol, const magma_int_t *m, const magma_int_t *n,
                         float *U, const magma_int_t *ldu,
                         float *work, const magma_int_t *lwork,
                         float *resid );
#endif

void   lapackf77_slarfy( const char *uplo, const magma_int_t *n,
                         float *V, const magma_int_t *incv,
                         float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work );

void   lapackf77_slarfx( const char *side, const magma_int_t *m, const magma_int_t *n,
                         float *V,
                         float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work );

float lapackf77_sqpt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         float *A,
                         float *Af, const magma_int_t *lda,
                         float *tau, magma_int_t *jpvt,
                         float *work, const magma_int_t *lwork );

void   lapackf77_sqrt02( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         float *A,
                         float *AF,
                         float *Q,
                         float *R, const magma_int_t *lda,
                         float *tau,
                         float *work, const magma_int_t *lwork,
                         float *rwork,
                         float *result );

void   lapackf77_slatms( magma_int_t *m, magma_int_t *n,
                         const char *dist, magma_int_t *iseed, const char *sym, float *d,
                         magma_int_t *mode, const float *cond, const float *dmax,
                         magma_int_t *kl, magma_int_t *ku, const char *pack,
                         float *a, magma_int_t *lda, float *work, magma_int_t *info );

#ifdef __cplusplus
}
#endif

#undef REAL

#endif /* MAGMA_SLAPACK_H */
