/*
 *   -- MAGMA (version 1.6.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      @date January 2015
 *
 * @precisions normal z -> s d c
 */

#ifndef MAGMA_ZLAPACK_H
#define MAGMA_ZLAPACK_H

#define PRECISION_z
#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- LAPACK Externs used in MAGMA
*/

#define blasf77_zaxpy      FORTRAN_NAME( zaxpy,  ZAXPY  )
#define blasf77_zcopy      FORTRAN_NAME( zcopy,  ZCOPY  )

/* complex versions use C wrapper to return value; no name mangling. */
#if  defined(PRECISION_z) || defined(PRECISION_c)    
#define blasf77_zdotc      zdotc
#else
#define blasf77_zdotc      FORTRAN_NAME( zdotc,  ZDOTC  )
#endif

#define blasf77_zgemm      FORTRAN_NAME( zgemm,  ZGEMM  )
#define blasf77_zgemv      FORTRAN_NAME( zgemv,  ZGEMV  )
#define blasf77_zhemm      FORTRAN_NAME( zhemm,  ZHEMM  )
#define blasf77_zhemv      FORTRAN_NAME( zhemv,  ZHEMV  )
#define blasf77_zher2k     FORTRAN_NAME( zher2k, ZHER2K )
#define blasf77_zherk      FORTRAN_NAME( zherk,  ZHERK  )
#define blasf77_zscal      FORTRAN_NAME( zscal,  ZSCAL  )
#define blasf77_zdscal     FORTRAN_NAME( zdscal, ZDSCAL ) 
#define blasf77_zsymm      FORTRAN_NAME( zsymm,  ZSYMM  )
#define blasf77_zsyr2k     FORTRAN_NAME( zsyr2k, ZSYR2K )
#define blasf77_zsyrk      FORTRAN_NAME( zsyrk,  ZSYRK  )
#define blasf77_zswap      FORTRAN_NAME( zswap,  ZSWAP  )
#define blasf77_ztrmm      FORTRAN_NAME( ztrmm,  ZTRMM  )
#define blasf77_ztrmv      FORTRAN_NAME( ztrmv,  ZTRMV  )
#define blasf77_ztrsm      FORTRAN_NAME( ztrsm,  ZTRSM  )
#define blasf77_ztrsv      FORTRAN_NAME( ztrsv,  ZTRSV  )
#define blasf77_zgeru      FORTRAN_NAME( zgeru,  ZGERU  )

#define lapackf77_zbdsqr   FORTRAN_NAME( zbdsqr, ZBDSQR )
#define lapackf77_zgebak   FORTRAN_NAME( zgebak, ZGEBAK )
#define lapackf77_zgebal   FORTRAN_NAME( zgebal, ZGEBAL )
#define lapackf77_zgebd2   FORTRAN_NAME( zgebd2, ZGEBD2 )
#define lapackf77_zgebrd   FORTRAN_NAME( zgebrd, ZGEBRD )
#define lapackf77_zgeev    FORTRAN_NAME( zgeev,  ZGEEV  )
#define lapackf77_zgehd2   FORTRAN_NAME( zgehd2, ZGEHD2 )
#define lapackf77_zgehrd   FORTRAN_NAME( zgehrd, ZGEHRD )
#define lapackf77_zgelqf   FORTRAN_NAME( zgelqf, ZGELQF )
#define lapackf77_zgels    FORTRAN_NAME( zgels,  ZGELS  )
#define lapackf77_zgeqlf   FORTRAN_NAME( zgeqlf, ZGEQLF )
#define lapackf77_zgeqrf   FORTRAN_NAME( zgeqrf, ZGEQRF )
#define lapackf77_zgesvd   FORTRAN_NAME( zgesvd, ZGESVD )
#define lapackf77_zgetrf   FORTRAN_NAME( zgetrf, ZGETRF )
#define lapackf77_zgetrs   FORTRAN_NAME( zgetrs, ZGETRS )
#define lapackf77_zheev    FORTRAN_NAME( zheev,  ZHEEV  )
#define lapackf77_zheevd   FORTRAN_NAME( zheevd, ZHEEVD )
#define lapackf77_zhegs2   FORTRAN_NAME( zhegs2, ZHEGS2 )
#define lapackf77_zhegvd   FORTRAN_NAME( zhegvd, ZHEGVD )
#define lapackf77_zhetd2   FORTRAN_NAME( zhetd2, ZHETD2 )
#define lapackf77_zhetrd   FORTRAN_NAME( zhetrd, ZHETRD )
#define lapackf77_zhseqr   FORTRAN_NAME( zhseqr, ZHSEQR )
#define lapackf77_zlacpy   FORTRAN_NAME( zlacpy, ZLACPY )
#define lapackf77_zlacgv   FORTRAN_NAME( zlacgv, ZLACGV )
#define lapackf77_zlange   FORTRAN_NAME( zlange, ZLANGE )
#define lapackf77_zlanhe   FORTRAN_NAME( zlanhe, ZLANHE )
#define lapackf77_zlansy   FORTRAN_NAME( zlansy, ZLANSY )
#define lapackf77_zlarfb   FORTRAN_NAME( zlarfb, ZLARFB )
#define lapackf77_zlarfg   FORTRAN_NAME( zlarfg, ZLARFG )
#define lapackf77_zlarft   FORTRAN_NAME( zlarft, ZLARFT )
#define lapackf77_zlarnv   FORTRAN_NAME( zlarnv, ZLARNV )
#define lapackf77_zlartg   FORTRAN_NAME( zlartg, ZLARTG )
#define lapackf77_zlascl   FORTRAN_NAME( zlascl, ZLASCL )
#define lapackf77_zlaset   FORTRAN_NAME( zlaset, ZLASET )
#define lapackf77_zlaswp   FORTRAN_NAME( zlaswp, ZLASWP )
#define lapackf77_zlatrd   FORTRAN_NAME( zlatrd, ZLATRD )
#define lapackf77_zlabrd   FORTRAN_NAME( zlabrd, ZLABRD )
#define lapackf77_zlauum   FORTRAN_NAME( zlauum, ZLAUUM )
#define lapackf77_zpotrf   FORTRAN_NAME( zpotrf, ZPOTRF )
#define lapackf77_zpotrs   FORTRAN_NAME( zpotrs, ZPOTRS )
#define lapackf77_zpotri   FORTRAN_NAME( zpotri, ZPOTRI )
#define lapackf77_ztrevc   FORTRAN_NAME( ztrevc, ZTREVC )
#define lapackf77_ztrtri   FORTRAN_NAME( ztrtri, ZTRTRI )
#define lapackf77_zsteqr   FORTRAN_NAME( zsteqr, ZSTEQR )
#define lapackf77_zstedc   FORTRAN_NAME( zstedc, ZSTEDC )
#define lapackf77_zsymv    FORTRAN_NAME( zsymv,  ZSYMV  )
#define lapackf77_zung2r   FORTRAN_NAME( zung2r, ZUNG2R )
#define lapackf77_zungbr   FORTRAN_NAME( zungbr, ZUNGBR )
#define lapackf77_zunghr   FORTRAN_NAME( zunghr, ZUNGHR )
#define lapackf77_zunglq   FORTRAN_NAME( zunglq, ZUNGLQ )
#define lapackf77_zungqr   FORTRAN_NAME( zungqr, ZUNGQR )
#define lapackf77_zungtr   FORTRAN_NAME( zungtr, ZUNGTR )
#define lapackf77_zunm2r   FORTRAN_NAME( zunm2r, ZUNM2R )
#define lapackf77_zunmbr   FORTRAN_NAME( zunmbr, ZUNMBR )
#define lapackf77_zunmlq   FORTRAN_NAME( zunmlq, ZUNMLQ )
#define lapackf77_zunmql   FORTRAN_NAME( zunmql, ZUNMQL )
#define lapackf77_zunmqr   FORTRAN_NAME( zunmqr, ZUNMQR )
#define lapackf77_zunmtr   FORTRAN_NAME( zunmtr, ZUNMTR )

/* testing functions */
#define lapackf77_zbdt01   FORTRAN_NAME( zbdt01, ZBDT01 )
#define lapackf77_zget22   FORTRAN_NAME( zget22, ZGET22 )
#define lapackf77_zhet21   FORTRAN_NAME( zhet21, ZHET21 )
#define lapackf77_zhst01   FORTRAN_NAME( zhst01, ZHST01 )
#define lapackf77_zqrt02   FORTRAN_NAME( zqrt02, ZQRT02 )
#define lapackf77_zunt01   FORTRAN_NAME( zunt01, ZUNT01 )
#define lapackf77_zlarfy   FORTRAN_NAME( zlarfy, ZLARFY )
#define lapackf77_zstt21   FORTRAN_NAME( zstt21, ZSTT21 )


#if defined(PRECISION_z) || defined(PRECISION_c)
#define DWORKFORZ        double *rwork,
#define DWORKFORZ_AND_LD double *rwork, magma_int_t *ldrwork,
#define WSPLIT           cuDoubleComplex *w
#else
#define DWORKFORZ 
#define DWORKFORZ_AND_LD
#define WSPLIT           double *wr, double *wi
#endif

  /*
   * BLAS functions (Alphabetical order)
   */
void     blasf77_zaxpy(const int *, cuDoubleComplex *, cuDoubleComplex *, 
               const int *, cuDoubleComplex *, const int *);
void     blasf77_zcopy(const int *, cuDoubleComplex *, const int *,
               cuDoubleComplex *, const int *);
#if defined(PRECISION_z) || defined(PRECISION_c)
void     blasf77_zdotc(cuDoubleComplex *, int *, cuDoubleComplex *, int *, 
                       cuDoubleComplex *, int *);
#endif
void     blasf77_zgemm(const char *, const char *, const int *, const int *, const int *,
               cuDoubleComplex *, cuDoubleComplex *, const int *, 
               cuDoubleComplex *, const int *, cuDoubleComplex *,
               cuDoubleComplex *, const int *);
void     blasf77_zgemv(const char *, const int  *, const int *, cuDoubleComplex *, 
               cuDoubleComplex *, const int *, cuDoubleComplex *, const int *, 
               cuDoubleComplex *, cuDoubleComplex *, const int *);
void     blasf77_zgeru(int *, int *, cuDoubleComplex *, cuDoubleComplex *, int *, 
               cuDoubleComplex *, int *, cuDoubleComplex *, int *);
void     blasf77_zhemm(const char *, const char *, const int *, const int *, 
               cuDoubleComplex *, cuDoubleComplex *, const int *, 
               cuDoubleComplex *, const int *, cuDoubleComplex *,
               cuDoubleComplex *, const int *);
void     blasf77_zhemv(const char *, const int  *, cuDoubleComplex *, cuDoubleComplex *,
               const int *, cuDoubleComplex *, const int *, cuDoubleComplex *,
               cuDoubleComplex *, const int *);
void    blasf77_zher2k(const char *, const char *, const int *, const int *, 
               cuDoubleComplex *, cuDoubleComplex *, const int *, 
               cuDoubleComplex *, const int *, double *, 
               cuDoubleComplex *, const int *);
void    blasf77_zherk( const char *, const char *, const int *, const int *, double *, 
               cuDoubleComplex *, const int *, double *, cuDoubleComplex *, 
               const int *);
void    blasf77_zscal( const int *, cuDoubleComplex *, cuDoubleComplex *, const int *);
#if defined(PRECISION_z) || defined(PRECISION_c)
void    blasf77_zdscal( const int *, double *, cuDoubleComplex *, const int *);
#endif
void    blasf77_zsymm( const char *, const char *, const int *, const int *, 
               cuDoubleComplex *, cuDoubleComplex *, const int *, 
               cuDoubleComplex *, const int *, cuDoubleComplex *,
               cuDoubleComplex *, const int *);
void    blasf77_zsyr2k(const char *, const char *, const int *, const int *, 
               cuDoubleComplex *, cuDoubleComplex *, const int *, 
               cuDoubleComplex *, const int *, cuDoubleComplex *, 
               cuDoubleComplex *, const int *);
void    blasf77_zsyrk( const char *, const char *, const int *, const int *, 
               cuDoubleComplex *, cuDoubleComplex *, const int *, 
               cuDoubleComplex *, cuDoubleComplex *, const int *);
void    blasf77_zswap( int *, cuDoubleComplex *, int *, cuDoubleComplex *, int *);
void    blasf77_ztrmm( const char *, const char *, const char *, const char *, 
               const int *, const int *, cuDoubleComplex *,
               cuDoubleComplex *, const int *, cuDoubleComplex *,const int *);
void    blasf77_ztrmv( const char *, const char *, const char *, const int *, 
               cuDoubleComplex*,  const int *, cuDoubleComplex *, const int*);
void    blasf77_ztrsm( const char *, const char *, const char *, const char *, 
               const int *, const int *, cuDoubleComplex *, 
               cuDoubleComplex *, const int *, cuDoubleComplex *,const int*);
void    blasf77_ztrsv( const char *, const char *, const char *, const int *, 
               cuDoubleComplex *, const int *, cuDoubleComplex *, const int*);

  /*
   * Lapack functions (Alphabetical order)
   */
void    lapackf77_zbdsqr(const char *uplo, magma_int_t *n, magma_int_t *nvct, 
             magma_int_t *nru,  magma_int_t *ncc, double *D, double *E, 
             cuDoubleComplex *VT, magma_int_t *ldvt, 
             cuDoubleComplex *U, magma_int_t *ldu, 
                         cuDoubleComplex *C, magma_int_t *ldc, 
             double *work, magma_int_t *info);
void    lapackf77_zgebak(const char *job, const char *side, magma_int_t *n, 
             magma_int_t *ilo, magma_int_t *ihi, 
                         double *scale, magma_int_t *m,
             cuDoubleComplex *v, magma_int_t *ldv, magma_int_t *info);
void    lapackf77_zgebal(const char *job, magma_int_t *n, cuDoubleComplex *A, magma_int_t *lda, 
                         magma_int_t *ilo, magma_int_t *ihi, double *scale, magma_int_t *info);
void    lapackf77_zgebd2(magma_int_t *m, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, double *d, double *e,
             cuDoubleComplex *tauq, cuDoubleComplex *taup,
             cuDoubleComplex *work, magma_int_t *info);
void    lapackf77_zgebrd(magma_int_t *m, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, double *d, double *e,
             cuDoubleComplex *tauq, cuDoubleComplex *taup, 
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void     lapackf77_zgeev(char *jobl, char *jobr, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, WSPLIT, 
             cuDoubleComplex *vl, magma_int_t *ldvl, 
             cuDoubleComplex *vr, magma_int_t *ldvr, 
             cuDoubleComplex *work, magma_int_t *lwork, 
             DWORKFORZ magma_int_t *info);
void    lapackf77_zgehd2(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
             cuDoubleComplex *a, magma_int_t *lda, cuDoubleComplex *tau, 
             cuDoubleComplex *work, magma_int_t *info);
void    lapackf77_zgehrd(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
             cuDoubleComplex *a, magma_int_t *lda, cuDoubleComplex *tau,
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zgelqf(magma_int_t *m, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, cuDoubleComplex *tau,
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void     lapackf77_zgels(const char *trans, 
             magma_int_t *m, magma_int_t *n, magma_int_t *nrhs, 
             cuDoubleComplex *a, magma_int_t *lda, 
             cuDoubleComplex *b, magma_int_t *ldb,
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zgeqlf(magma_int_t *m, magma_int_t *n,
             cuDoubleComplex *a, magma_int_t *lda, cuDoubleComplex *tau, 
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zgeqrf(magma_int_t *m, magma_int_t *n,
             cuDoubleComplex *a, magma_int_t *lda, cuDoubleComplex *tau,
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zgetrf(magma_int_t *m, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, 
             magma_int_t *ipiv, magma_int_t *info);
void    lapackf77_zgetrs(const char* trans,
                         magma_int_t *n, magma_int_t *nrhs,
                         cuDoubleComplex *a, magma_int_t *lda, magma_int_t *ipiv,
                         cuDoubleComplex *b, magma_int_t *ldb, magma_int_t *info);
void    lapackf77_zgesvd(const char *jobu, const char *jobvt, 
             magma_int_t *m, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, 
             double *s, cuDoubleComplex *u, magma_int_t *ldu, 
                         cuDoubleComplex *vt, magma_int_t *ldvt, 
             cuDoubleComplex *work, magma_int_t *lwork, 
             DWORKFORZ magma_int_t *info );
void    lapackf77_zheev(const char *jobz, const char *uplo, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, double *w, 
                         cuDoubleComplex *work, magma_int_t *lwork,
             DWORKFORZ_AND_LD magma_int_t *info);
void    lapackf77_zheevd(const char *jobz, const char *uplo, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, double *w, 
                         cuDoubleComplex *work, magma_int_t *lwork,
             DWORKFORZ_AND_LD magma_int_t *iwork, 
             magma_int_t *liwork, magma_int_t *info);
void    lapackf77_zhegs2(int *itype, char *uplo, int *n, 
             cuDoubleComplex *a, int *lda, 
             cuDoubleComplex *b, int *ldb, int *info);
void    lapackf77_zhegvd(magma_int_t *itype, const char *jobz, const char *uplo, 
             magma_int_t *n, cuDoubleComplex *a, magma_int_t *lda,
             cuDoubleComplex *b, magma_int_t *ldb, double *w,
             cuDoubleComplex *work, magma_int_t *lwork, 
             DWORKFORZ_AND_LD magma_int_t *iwork, magma_int_t *liwork,
             magma_int_t *info);
void    lapackf77_zhetd2(const char *uplo, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, 
             double *d, double *e, cuDoubleComplex *tau, magma_int_t *info);
void    lapackf77_zhetrd(const char *uplo, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, 
             double *d, double *e, cuDoubleComplex *tau, 
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zhseqr(const char *job, const char *compz, magma_int_t *n, 
             magma_int_t *ilo, magma_int_t *ihi, 
                         cuDoubleComplex *H, magma_int_t *ldh, WSPLIT, 
                         cuDoubleComplex *Z, magma_int_t *ldz, 
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zlacpy(const char *uplo, magma_int_t *m, magma_int_t *n, 
             const cuDoubleComplex *a, magma_int_t *lda, 
             cuDoubleComplex *b, magma_int_t *ldb);
void    lapackf77_zlacgv(magma_int_t *n, cuDoubleComplex *x, magma_int_t *incx);
double  lapackf77_zlange(const char *norm, magma_int_t *m, magma_int_t *n, 
             const cuDoubleComplex *a, magma_int_t *lda, double *work);
double  lapackf77_zlanhe(const char *norm, const char *uplo, magma_int_t *n, 
             const cuDoubleComplex *a, magma_int_t *lda, double * work);
double  lapackf77_zlansy(const char *norm, const char *uplo, magma_int_t *n, 
             const cuDoubleComplex *a, magma_int_t *lda, double * work);
void    lapackf77_zlarfb(const char *side, const char *trans, const char *direct, 
             const char *storev, magma_int_t *m, magma_int_t *n, magma_int_t *k, 
             const cuDoubleComplex *v, magma_int_t *ldv, 
             const cuDoubleComplex *t, magma_int_t *ldt, 
             cuDoubleComplex *c, magma_int_t *ldc, 
             cuDoubleComplex *work, magma_int_t *ldwork);
void    lapackf77_zlarfg(magma_int_t *n, cuDoubleComplex *alpha, 
             cuDoubleComplex *x, magma_int_t *incx, cuDoubleComplex *tau);
void    lapackf77_zlarft(const char *direct, const char *storev, magma_int_t *n, magma_int_t *k, 
             cuDoubleComplex *v, magma_int_t *ldv, const cuDoubleComplex *tau, 
             cuDoubleComplex *t, magma_int_t *ldt);
void    lapackf77_zlarnv(magma_int_t *idist, magma_int_t *iseed, magma_int_t *n, 
             cuDoubleComplex *x);
void    lapackf77_zlartg(cuDoubleComplex *F, cuDoubleComplex *G, double *cs, 
             cuDoubleComplex *SN, cuDoubleComplex *R);
void    lapackf77_zlascl(const char *type, magma_int_t *kl, magma_int_t *ku, 
             double *cfrom, double *cto, 
                         magma_int_t *m, magma_int_t *n, 
             cuDoubleComplex *A, magma_int_t *lda, magma_int_t *info);
void    lapackf77_zlaset(const char *uplo, magma_int_t *m, magma_int_t *n, 
             cuDoubleComplex *alpha, cuDoubleComplex *beta,
             cuDoubleComplex *A, magma_int_t *lda);
void    lapackf77_zlaswp(magma_int_t *n, cuDoubleComplex *a, magma_int_t *lda, 
             magma_int_t *k1, magma_int_t *k2, magma_int_t *ipiv,
             magma_int_t *incx);
void    lapackf77_zlatrd(const char *uplo, magma_int_t *n, magma_int_t *nb, 
             cuDoubleComplex *a, magma_int_t *lda, double *e,
             cuDoubleComplex *tau, cuDoubleComplex *work, magma_int_t *ldwork);
void    lapackf77_zlabrd(magma_int_t *m, magma_int_t *n, magma_int_t *nb, 
             cuDoubleComplex *a, magma_int_t *lda, double *d__, double *e, 
             cuDoubleComplex *tauq, cuDoubleComplex *taup,
             cuDoubleComplex *x, magma_int_t *ldx,
             cuDoubleComplex *y, magma_int_t *ldy);
void    lapackf77_zpotrf(const char *uplo, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, magma_int_t *info);
void    lapackf77_zpotrs(const char *uplo, magma_int_t *n, magma_int_t *nrhs,
             cuDoubleComplex *a, magma_int_t *lda,
             cuDoubleComplex *b, magma_int_t *ldb, magma_int_t *info);
void    lapackf77_zpotri(const char *uplo, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, magma_int_t *info);
void    lapackf77_zlauum(const char *uplo, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, magma_int_t *info);
void    lapackf77_ztrevc(const char *side, const char *howmny, magma_int_t *select, magma_int_t *n, 
                         cuDoubleComplex *T,  magma_int_t *ldt,  cuDoubleComplex *VL, magma_int_t *ldvl,
                         cuDoubleComplex *VR, magma_int_t *ldvr, magma_int_t *MM, magma_int_t *M, 
                         cuDoubleComplex *work, DWORKFORZ magma_int_t *info);
void    lapackf77_zsteqr(const char *compz, magma_int_t *n, double *D, double *E, 
                         cuDoubleComplex *Z, magma_int_t *ldz, 
                         double *work, magma_int_t *info);
void    lapackf77_zstedc(const char *compz, magma_int_t *n, double *D, double *E, 
             cuDoubleComplex *Z, magma_int_t *ldz, 
                         cuDoubleComplex *work, magma_int_t *ldwork, 
             DWORKFORZ_AND_LD magma_int_t *iwork, magma_int_t *liwork,
             magma_int_t *info);
void    lapackf77_ztrtri(const char *uplo, const char *diag, magma_int_t *n,
             cuDoubleComplex *a, magma_int_t *lda, magma_int_t *info);
#if defined(PRECISION_z) || defined(PRECISION_c)
void    lapackf77_zsymv(const char *uplo, const magma_int_t *N, const cuDoubleComplex *alpha, 
            const cuDoubleComplex *A, const magma_int_t *lda, 
            const cuDoubleComplex *X, const magma_int_t *incX,
            const cuDoubleComplex *beta, 
            cuDoubleComplex *Y, const magma_int_t *incY);
#endif
void    lapackf77_zung2r(magma_int_t *m, magma_int_t *n, magma_int_t *k, 
             cuDoubleComplex *a, magma_int_t *lda,
             const cuDoubleComplex *tau, cuDoubleComplex *work,
             magma_int_t *info);
void    lapackf77_zungbr(const char *vect, magma_int_t *m, magma_int_t *n, magma_int_t *k,
             cuDoubleComplex *a, magma_int_t *lda, const cuDoubleComplex *tau,
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunghr(magma_int_t *n, magma_int_t *ilo, magma_int_t *ihi, 
             cuDoubleComplex *a, magma_int_t *lda, const cuDoubleComplex *tau,
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunglq(magma_int_t *m, magma_int_t *n, magma_int_t *k, 
             cuDoubleComplex *a, magma_int_t *lda, const cuDoubleComplex *tau, 
             cuDoubleComplex *work, magma_int_t *ldwork, magma_int_t *info);
void    lapackf77_zungqr(magma_int_t *m, magma_int_t *n, magma_int_t *k, 
             cuDoubleComplex *a, magma_int_t *lda, const cuDoubleComplex *tau, 
             cuDoubleComplex *work, magma_int_t *ldwork, magma_int_t *info);
void    lapackf77_zungtr(const char *uplo, magma_int_t *n, 
             cuDoubleComplex *a, magma_int_t *lda, const cuDoubleComplex *tau, 
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunm2r(const char *side, const char *trans, 
             magma_int_t *m, magma_int_t *n, magma_int_t *k, 
             const cuDoubleComplex *a, magma_int_t *lda, 
             const cuDoubleComplex *tau, cuDoubleComplex *c, magma_int_t *ldc,
             cuDoubleComplex *work, magma_int_t *info);
void    lapackf77_zunmbr(const char *vect, const char *side, const char *trans,
             magma_int_t *M, magma_int_t *N, magma_int_t *K, 
             cuDoubleComplex *A, magma_int_t *lda, cuDoubleComplex *Tau,
                         cuDoubleComplex *C, magma_int_t *ldc, 
             cuDoubleComplex *work, magma_int_t *ldwork, magma_int_t *info);
void    lapackf77_zunmlq(const char *side, const char *trans, 
             magma_int_t *m, magma_int_t *n, magma_int_t *k,
             const cuDoubleComplex *a, magma_int_t *lda, 
             const cuDoubleComplex *tau, cuDoubleComplex *c, magma_int_t *ldc, 
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunmql(const char *side, const char *trans, 
             magma_int_t *m, magma_int_t *n, magma_int_t *k,
             const cuDoubleComplex *a, magma_int_t *lda, 
             const cuDoubleComplex *tau, cuDoubleComplex *c, magma_int_t *ldc,
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunmqr(const char *side, const char *trans, 
             magma_int_t *m, magma_int_t *n, magma_int_t *k, 
             const cuDoubleComplex *a, magma_int_t *lda, 
             const cuDoubleComplex *tau, cuDoubleComplex *c, magma_int_t *ldc, 
             cuDoubleComplex *work, magma_int_t *lwork, magma_int_t *info);
void    lapackf77_zunmtr(const char *side, const char *uplo, const char *trans,
             magma_int_t *M, magma_int_t *N,
             cuDoubleComplex *A, magma_int_t *lda, cuDoubleComplex *Tau,
                         cuDoubleComplex *C, magma_int_t *ldc, 
             cuDoubleComplex *work, magma_int_t *ldwork, magma_int_t *info);


  /*
   * Testing functions
   */

#if defined(PRECISION_z) || defined(PRECISION_c)

void    lapackf77_zbdt01(int *m, int *n, int *kd, cuDoubleComplex *A, int *lda, 
             cuDoubleComplex *Q, int *ldq, double *D, double *E, 
             cuDoubleComplex *PT, int *ldpt, cuDoubleComplex *work, 
             double *rwork, double *resid);
void    lapackf77_zget22(char *transa, char *transe, char *transw, int *n,
             cuDoubleComplex *a, int *lda, cuDoubleComplex *e, int *lde,
             cuDoubleComplex *w, cuDoubleComplex *work,
             double *rwork, double *result);
void    lapackf77_zhet21(int *itype, const char *uplo, int *n, int *kband, 
             cuDoubleComplex *A, int *lda, double *D, double *E, 
             cuDoubleComplex *U, int *ldu, cuDoubleComplex *V, int *ldv, 
             cuDoubleComplex *TAU, cuDoubleComplex *work,
             double *rwork, double *result);
void    lapackf77_zhst01(int *n, int *ilo, int *ihi, cuDoubleComplex *A, int *lda, 
             cuDoubleComplex *H, int *ldh, cuDoubleComplex *Q, int *ldq,
             cuDoubleComplex *work, int *lwork, double *rwork, double *result);
void    lapackf77_zstt21(int *n, int *kband, double *AD, double *AE, double *SD,
             double *SE, cuDoubleComplex *U, int *ldu, 
             cuDoubleComplex *work, double *rwork, double *result);
void    lapackf77_zunt01(const char *rowcol, int *m, int *n, cuDoubleComplex *U, int *ldu,
             cuDoubleComplex *work, int *lwork, double *rwork, double *resid);

#else

void    lapackf77_zbdt01(int *m, int *n, int *kd, cuDoubleComplex *A, int *lda, 
             cuDoubleComplex *Q, int *ldq, double *D, double *E, 
             cuDoubleComplex *PT, int *ldpt, 
             cuDoubleComplex *work, double *resid);
void    lapackf77_zget22(char *transa, char *transe, char *transw, int *n,
             cuDoubleComplex *a, int *lda, cuDoubleComplex *e, int *lde,
             cuDoubleComplex *wr, cuDoubleComplex *wi, 
             double *work, double *result);
void    lapackf77_zhet21(int *itype, const char *uplo, int *n, int *kband, 
             cuDoubleComplex *A, int *lda, double *D, double *E,
             cuDoubleComplex *U, int *ldu, cuDoubleComplex *V, int *ldv, 
             cuDoubleComplex *TAU, cuDoubleComplex *work, double *result);
void    lapackf77_zhst01(int *n, int *ilo, int *ihi, cuDoubleComplex *A, int *lda, 
             cuDoubleComplex *H, int *ldh, cuDoubleComplex *Q, int *ldq, 
             cuDoubleComplex *work, int *lwork, double *result);
void    lapackf77_zstt21(int *n, int *kband, double *AD, double *AE, double *SD, 
             double *SE, cuDoubleComplex *U, int *ldu, 
             cuDoubleComplex *work, double *result);
void    lapackf77_zunt01(const char *rowcol, int *m, int *n, cuDoubleComplex *U, int *ldu,
             cuDoubleComplex *work, int *lwork, double *resid);
#endif

void    lapackf77_zlarfy(const char *uplo, int *N, cuDoubleComplex *V, int *incv, 
             cuDoubleComplex *tau, cuDoubleComplex *C, int *ldc, 
             cuDoubleComplex *work);
void    lapackf77_zqrt02(int *m, int *n, int *k, cuDoubleComplex *A, cuDoubleComplex *AF,
             cuDoubleComplex *Q, cuDoubleComplex *R, int *lda, 
             cuDoubleComplex *TAU, cuDoubleComplex *work, int *lwork,
             double *rwork, double *result);

#ifdef __cplusplus
}
#endif

#undef DWORKFORZ 
#undef DWORKFORZ_AND_LD
#undef WSPLIT
#undef PRECISION_z
#endif /* MAGMA ZLAPACK */
