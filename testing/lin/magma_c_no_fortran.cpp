/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates
       @generated from testing/lin/magma_z_no_fortran.cpp normal z -> c, Mon May  2 23:31:05 2016
       
       This is simply a copy of part of magma_clapack.h,
       with the { printf(...); } function body added to each function.
*/
#include <stdio.h>

#include "magma_v2.h"
#include "magma_lapack.h"

#ifdef __cplusplus
extern "C" {
#endif

#define COMPLEX

static const char* format = "Cannot check results: %s unavailable, since there was no Fortran compiler.\n";

/*
 * Testing functions
 */
void   lapackf77_cbdt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *kd,
                         magmaFloatComplex *A, const magma_int_t *lda,
                         magmaFloatComplex *Q, const magma_int_t *ldq,
                         float *d, float *e,
                         magmaFloatComplex *Pt, const magma_int_t *ldpt,
                         magmaFloatComplex *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *resid )
                         { printf( format, __func__ ); }

void   lapackf77_cget22( const char *transa, const char *transe, const char *transw, const magma_int_t *n,
                         magmaFloatComplex *A, const magma_int_t *lda,
                         magmaFloatComplex *E, const magma_int_t *lde,
                         #ifdef COMPLEX
                         magmaFloatComplex *w,
                         #else
                         magmaFloatComplex *wr,
                         magmaFloatComplex *wi,
                         #endif
                         magmaFloatComplex *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_chet21( const magma_int_t *itype, const char *uplo,
                         const magma_int_t *n, const magma_int_t *kband,
                         magmaFloatComplex *A, const magma_int_t *lda,
                         float *d, float *e,
                         magmaFloatComplex *U, const magma_int_t *ldu,
                         magmaFloatComplex *V, const magma_int_t *ldv,
                         magmaFloatComplex *tau,
                         magmaFloatComplex *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_chst01( const magma_int_t *n, const magma_int_t *ilo, const magma_int_t *ihi,
                         magmaFloatComplex *A, const magma_int_t *lda,
                         magmaFloatComplex *H, const magma_int_t *ldh,
                         magmaFloatComplex *Q, const magma_int_t *ldq,
                         magmaFloatComplex *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_cstt21( const magma_int_t *n, const magma_int_t *kband,
                         float *AD,
                         float *AE,
                         float *SD,
                         float *SE,
                         magmaFloatComplex *U, const magma_int_t *ldu,
                         magmaFloatComplex *work,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_cunt01( const char *rowcol, const magma_int_t *m, const magma_int_t *n,
                         magmaFloatComplex *U, const magma_int_t *ldu,
                         magmaFloatComplex *work, const magma_int_t *lwork,
                         #ifdef COMPLEX
                         float *rwork,
                         #endif
                         float *resid )
                         { printf( format, __func__ ); }

void   lapackf77_clarfy( const char *uplo, const magma_int_t *n,
                         magmaFloatComplex *V, const magma_int_t *incv,
                         magmaFloatComplex *tau,
                         magmaFloatComplex *C, const magma_int_t *ldc,
                         magmaFloatComplex *work )
                         { printf( format, __func__ ); }

void   lapackf77_clarfx( const char *side, const magma_int_t *m, const magma_int_t *n,
                         magmaFloatComplex *V,
                         magmaFloatComplex *tau,
                         magmaFloatComplex *C, const magma_int_t *ldc,
                         magmaFloatComplex *work )
                         { printf( format, __func__ ); }

float lapackf77_cqpt01( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         magmaFloatComplex *A,
                         magmaFloatComplex *Af, const magma_int_t *lda,
                         magmaFloatComplex *tau, magma_int_t *jpvt,
                         magmaFloatComplex *work, const magma_int_t *lwork )
                         { printf( format, __func__ ); return -1; }

void   lapackf77_cqrt02( const magma_int_t *m, const magma_int_t *n, const magma_int_t *k,
                         magmaFloatComplex *A,
                         magmaFloatComplex *AF,
                         magmaFloatComplex *Q,
                         magmaFloatComplex *R, const magma_int_t *lda,
                         magmaFloatComplex *tau,
                         magmaFloatComplex *work, const magma_int_t *lwork,
                         float *rwork,
                         float *result )
                         { printf( format, __func__ ); }

void   lapackf77_clatms( magma_int_t *m, magma_int_t *n,
                         const char *dist, magma_int_t *iseed, const char *sym, float *d,
                         magma_int_t *mode, const float *cond, const float *dmax,
                         magma_int_t *kl, magma_int_t *ku, const char *pack,
                         magmaFloatComplex *a, magma_int_t *lda, magmaFloatComplex *work, magma_int_t *info )
                         { printf( format, __func__ ); }

#ifdef __cplusplus
}
#endif
