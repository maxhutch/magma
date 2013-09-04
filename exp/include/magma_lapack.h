#ifndef MAGMA_LAPACK_H
#define MAGMA_LAPACK_H

#ifndef FORTRAN_NAME
#if defined(ADD_)
#define FORTRAN_NAME(lcname, UCNAME)  lcname##_
#elif defined(NOCHANGE)               
#define FORTRAN_NAME(lcname, UCNAME)  lcname
#elif defined(UPCASE)                 
#define FORTRAN_NAME(lcname, UCNAME)  UCNAME
#endif
#endif

#include "magma_zlapack.h"
#include "magma_clapack.h"
#include "magma_dlapack.h"
#include "magma_slapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#define lapackf77_lsame  FORTRAN_NAME( lsame,  LSAME  )
#define lapackf77_xerbla FORTRAN_NAME( xerbla, XERBLA )

#define lapackf77_slamch FORTRAN_NAME( slamch, SLAMCH )
#define lapackf77_dlamch FORTRAN_NAME( dlamch, DLAMCH )
#define lapackf77_slabad FORTRAN_NAME( slabad, SLABAD )
#define lapackf77_dlabad FORTRAN_NAME( dlabad, DLABAD )
#define lapackf77_zcgesv FORTRAN_NAME( zcgesv, ZCGESV )
#define lapackf77_dsgesv FORTRAN_NAME( dsgesv, DSGESV )
                                               
#define lapackf77_dsterf FORTRAN_NAME( dsterf, DSTERF )
#define lapackf77_ssterf FORTRAN_NAME( ssterf, SSTERF )
                                               
#define lapackf77_zlag2c FORTRAN_NAME( zlag2c, ZLAG2C )
#define lapackf77_clag2z FORTRAN_NAME( clag2z, CLAG2Z )
#define lapackf77_dlag2s FORTRAN_NAME( dlag2s, DLAG2S )
#define lapackf77_slag2d FORTRAN_NAME( slag2d, SLAG2D )
                                               
#define lapackf77_dlapy2 FORTRAN_NAME( dlapy2, DLAPY2 )
#define lapackf77_slapy2 FORTRAN_NAME( slapy2, SLAPY2 )

long int lapackf77_lsame( const char *ca, const char *cb);
void     lapackf77_xerbla( const char* name, magma_int_t* info, int name_len );

float    lapackf77_slamch(const char *cmach);
double   lapackf77_dlamch(const char *cmach);
void     lapackf77_slabad(float  *small, float  *large);
void     lapackf77_dlabad(double *small, double *large);
void     lapackf77_zcgesv(magma_int_t *n, magma_int_t *nrhs, cuDoubleComplex *A, magma_int_t *lda, magma_int_t *IPIV, cuDoubleComplex *B, magma_int_t *ldb, 
                          cuDoubleComplex *X, magma_int_t *ldx, cuDoubleComplex *work, cuFloatComplex *swork, double *rwork, magma_int_t *iter, magma_int_t *info);
void     lapackf77_dsgesv(magma_int_t *n, magma_int_t *nrhs, double          *A, magma_int_t *lda, magma_int_t *IPIV, double          *B, magma_int_t *ldb, 
                          double          *X, magma_int_t *ldx, double          *work, float          *swork,                magma_int_t *iter, magma_int_t *info);

void     lapackf77_dsterf(magma_int_t *, double *, double *, magma_int_t *);
void     lapackf77_ssterf(magma_int_t *, float *, float *, magma_int_t *);

void     lapackf77_zlag2c( magma_int_t *m, magma_int_t *n, cuDoubleComplex *a,  magma_int_t *lda,  cuFloatComplex  *sa, magma_int_t *ldsa, magma_int_t *info );
void     lapackf77_clag2z( magma_int_t *m, magma_int_t *n, cuFloatComplex  *sa, magma_int_t *ldsa, cuDoubleComplex *a,  magma_int_t *lda,  magma_int_t *info );
void     lapackf77_dlag2s( magma_int_t *m, magma_int_t *n, double          *a,  magma_int_t *lda,  float           *sa, magma_int_t *ldsa, magma_int_t *info );
void     lapackf77_slag2d( magma_int_t *m, magma_int_t *n, float           *sa, magma_int_t *ldsa, double          *a,  magma_int_t *lda,  magma_int_t *info );

double   lapackf77_dlapy2( double *x, double *y  );
float    lapackf77_slapy2( float  *x, float  *y  );

// zdotc has different calling sequence, so define these here
double   blasf77_ddot( magma_int_t *, double *, magma_int_t *, double *, magma_int_t *);
float    blasf77_sdot( magma_int_t *, float *,  magma_int_t *, float *,  magma_int_t *);

#ifdef __cplusplus
}
#endif

#endif /* MAGMA LAPACK */
