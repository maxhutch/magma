/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates
       @generated from testing/lin/magma_z_no_fortran.cpp normal z -> s, Mon May  2 23:31:04 2016
       
       This is simply a copy of part of magma_slapack.h,
       with the { printf(...); } function body added to each function.
*/
#include <stdio.h>

#include "magma_v2.h"
#include "magma_lapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#define REAL

static const char* format = "Cannot check results: %s unavailable, since there was no Fortran compiler.\n";

/*
 * Testing functions
 */
void   lapackf77_sbdt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *kd,
                         float *A, const magma_int_t *lda,
                         float *Q, const magma_int_t *ldq,
                         float *d, float *e,
                         float *Pt, const magma_int_t *ldpt,
                         float *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *resid )
                         { printf( format, __func__ ); }

void   lapackf77_sget22( const char *transa, const char *transe, const char *transw, const magma_int_t *n,
                         float *A, const magma_int_t *lda,
                         float *E, const magma_int_t *lde,
                         #ifdef COMPLEX
                         float *w,
                         #else
                         float *wr,
                         float *wi,
                         #endif
                         float *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_ssyt21( const magma_int_t *itype, const char *uplo,
                         const magma_int_t *n, const magma_int_t *kband,
                         float *A, const magma_int_t *lda,
                         float *d, float *e,
                         float *U, const magma_int_t *ldu,
                         float *V, const magma_int_t *ldv,
                         float *tau,
                         float *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_shst01( const magma_int_t *n, const magma_int_t *ilo, const magma_int_t *ihi,
                         float *A, const magma_int_t *lda,
                         float *H, const magma_int_t *ldh,
                         float *Q, const magma_int_t *ldq,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_sstt21( const magma_int_t *n, const magma_int_t *kband,
                         float *AD,
                         float *AE,
                         float *SD,
                         float *SE,
                         float *U, const magma_int_t *ldu,
                         float *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_sort01( const char *rowcol, const magma_int_t *m, const magma_int_t *n,
                         float *U, const magma_int_t *ldu,
                         float *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *resid )
                         { printf( format, __func__ ); }

void   lapackf77_slarfy( const char *uplo, const magma_int_t *n,
                         float *V, const magma_int_t *incv,
                         float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work )
                         { printf( format, __func__ ); }

void   lapackf77_slarfx( const char *side, const magma_int_t *m, const magma_int_t *n,
                         float *V,
                         float *tau,
                         float *C, const magma_int_t *ldc,
                         float *work )
                         { printf( format, __func__ ); }

float lapackf77_sqpt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         float *A,
                         float *Af, const magma_int_t *lda,
                         float *tau, magma_int_t *jpvt,
                         float *work, const magma_int_t *lwork )
                         { printf( format, __func__ ); return -1; }

void   lapackf77_sqrt02( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         float *A,
                         float *AF,
                         float *Q,
                         float *R, const magma_int_t *lda,
                         float *tau,
                         float *work, const magma_int_t *lwork,
                         float *rwork,
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_slatms( magma_int_t *m, magma_int_t *n,
                         const char *dist, magma_int_t *iseed, const char *sym, float *d,
                         magma_int_t *mode, const float *cond, const float *dmax,
                         magma_int_t *kl, magma_int_t *ku, const char *pack,
                         float *a, magma_int_t *lda, float *work, magma_int_t *info )
                         { printf( format, __func__ ); }

#ifdef __cplusplus
}
#endif
