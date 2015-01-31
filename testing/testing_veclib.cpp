/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
*/

// Checks vector and matrix norms, for -m32 and -m64, with return as float or as double.
// In MacOS, -m64 must return double for both single and double precision
// functions, e.g., {s,d}dot, {s,d}nrm2, {s,d}lange, {s,d}lansy.
// Oddly, with -m32 both return float and double for single precision functions works.
// This is essentially a bug from an old f2c version of lapack (clapack).
//
// We work around this bug by putting replacement routines in the magma/lib/libblas_fix.a library.
// These correctly return float for the single precision functions that had issues.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "magma_types.h"
#include "magma_mangling.h"

// ------------------------------------------------------------
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ------------------------------------------------------------
//#define LAPACK_RETURN_DOUBLE

#ifdef LAPACK_RETURN_DOUBLE
typedef double RETURN_FLOAT;
#else
typedef float  RETURN_FLOAT;
#endif


// ------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

#define blasf77_sdot     FORTRAN_NAME( sdot,   SDOT   )
#define blasf77_snrm2    FORTRAN_NAME( snrm2,  SNRM2  )
#define lapackf77_slange FORTRAN_NAME( slange, SLANGE )
#define lapackf77_slansy FORTRAN_NAME( slansy, SLANSY )

#define blasf77_ddot     FORTRAN_NAME( ddot,   DDOT   )
#define blasf77_dnrm2    FORTRAN_NAME( dnrm2,  DNRM2  )
#define lapackf77_dlange FORTRAN_NAME( dlange, DLANGE )
#define lapackf77_dlansy FORTRAN_NAME( dlansy, DLANSY )

RETURN_FLOAT
blasf77_sdot(   const magma_int_t *n,
                const float *x, const magma_int_t *incx,
                const float *y, const magma_int_t *incy );
                
RETURN_FLOAT    
blasf77_snrm2(  const magma_int_t *n,
                const float *x, const magma_int_t *incx );

RETURN_FLOAT
lapackf77_slange( const char *norm,
                const magma_int_t *m, const magma_int_t *n,
                const float *A, const magma_int_t *lda,
                float *work );

RETURN_FLOAT
lapackf77_slansy( const char *norm, const char* uplo,
                const magma_int_t *n,
                const float *A, const magma_int_t *lda,
                float *work );

double
blasf77_ddot(   const magma_int_t *n,
                const double *x, const magma_int_t *incx,
                const double *y, const magma_int_t *incy );

double
blasf77_dnrm2(  const magma_int_t *n,
                const double *x, const magma_int_t *incx );

double
lapackf77_dlange( const char *norm,
                const magma_int_t *m, const magma_int_t *n,
                const double *A, const magma_int_t *lda,
                double *work );

double
lapackf77_dlansy( const char *norm, const char* uplo,
                const magma_int_t *n,
                const double *A, const magma_int_t *lda,
                double *work );

#ifdef __cplusplus
}
#endif

// ------------------------------------------------------------
// call matrix norms {s,d}lan{ge,sy}.
// return value, to check that the call stack isn't messed up.
float test( magma_int_t m, magma_int_t n )
{
#define sA(i,j) (sA + (i) + (j)*lda)
#define dA(i,j) (dA + (i) + (j)*lda)
    
    float  *sA, *swork;
    float  snorm_one, snorm_inf, snorm_fro, snorm_max;
    
    double *dA, *dwork;
    double dnorm_one, dnorm_inf, dnorm_fro, dnorm_max;
    
    const magma_int_t ione = 1;
    magma_int_t lda = MAX(m,n);
    
    sA    = (float*)  malloc( lda*n * sizeof(float)  );
    dA    = (double*) malloc( lda*n * sizeof(double) );
    swork = (float*)  malloc( m     * sizeof(float)  );
    dwork = (double*) malloc( m     * sizeof(double) );
    
    for( magma_int_t j = 0; j < n; ++j ) {
    for( magma_int_t i = 0; i < lda; ++i ) {
        double tmp = rand() / (double)(RAND_MAX);
        *sA(i,j) = tmp;
        *dA(i,j) = tmp;
    }}
    
    double error;
    magma_int_t status;
    
    // can repeat multiple times, but shows same results every time
    status = 0;
    for( magma_int_t i=0; i < 1; ++i ) {
        snorm_one = blasf77_sdot(  &m, sA, &ione, sA, &ione );
        dnorm_one = blasf77_ddot(  &m, dA, &ione, dA, &ione );
        snorm_fro = blasf77_snrm2( &m, sA, &ione );
        dnorm_fro = blasf77_dnrm2( &m, dA, &ione );
        printf( "m %d, sdot %12.8f, snrm2 %12.8f\n", (int) m, snorm_one, snorm_fro );
        printf( "m %d, ddot %12.8f, dnrm2 %12.8f\n", (int) m, dnorm_one, dnorm_fro );
        error = fabs(snorm_one - dnorm_one) / dnorm_one;
        status |= ! (error < 1e-6);
    }
    if ( status ) {
        printf( "**** failed ****\n" );
    }
    else {
        printf( "ok\n" );
    }
    printf( "\n" );
    
    status = 0;
    for( magma_int_t i=0; i < 1; ++i ) {
        snorm_one = lapackf77_slange( "one", &m, &n, sA, &lda, swork );
        snorm_inf = lapackf77_slange( "inf", &m, &n, sA, &lda, swork );
        snorm_max = lapackf77_slange( "max", &m, &n, sA, &lda, swork );
        snorm_fro = lapackf77_slange( "fro", &m, &n, sA, &lda, swork );
                                                      
        dnorm_one = lapackf77_dlange( "one", &m, &n, dA, &lda, dwork );
        dnorm_inf = lapackf77_dlange( "inf", &m, &n, dA, &lda, dwork );
        dnorm_max = lapackf77_dlange( "max", &m, &n, dA, &lda, dwork );
        dnorm_fro = lapackf77_dlange( "fro", &m, &n, dA, &lda, dwork );
        
        printf( "m %d, n %d, slange norm one %12.8f,  inf %12.8f,  max %12.8f,  fro %12.8f\n",
                (int) m, (int) n, snorm_one, snorm_inf, snorm_max, snorm_fro );
        
        printf( "m %d, n %d, dlange norm one %12.8f,  inf %12.8f,  max %12.8f,  fro %12.8f\n",
                (int) m, (int) n, dnorm_one, dnorm_inf, dnorm_max, dnorm_fro );
        error = fabs(snorm_one - dnorm_one) / dnorm_one;
        status |= ! (error < 1e-6);
    }
    if ( status ) {
        printf( "**** failed ****\n" );
    }
    else {
        printf( "ok\n" );
    }
    printf( "\n" );
    
    status = 0;
    for( magma_int_t i=0; i < 1; ++i ) {
        snorm_one = lapackf77_slansy( "one", "up", &n, sA, &lda, swork );
        snorm_inf = lapackf77_slansy( "inf", "up", &n, sA, &lda, swork );
        snorm_max = lapackf77_slansy( "max", "up", &n, sA, &lda, swork );
        snorm_fro = lapackf77_slansy( "fro", "up", &n, sA, &lda, swork );
                                                  
        dnorm_one = lapackf77_dlansy( "one", "up", &n, dA, &lda, dwork );
        dnorm_inf = lapackf77_dlansy( "inf", "up", &n, dA, &lda, dwork );
        dnorm_max = lapackf77_dlansy( "max", "up", &n, dA, &lda, dwork );
        dnorm_fro = lapackf77_dlansy( "fro", "up", &n, dA, &lda, dwork );
        
        printf( "m %d, n %d, slansy norm one %12.8f,  inf %12.8f,  max %12.8f,  fro %12.8f\n",
                (int) m, (int) n, snorm_one, snorm_inf, snorm_max, snorm_fro );
        
        printf( "m %d, n %d, dlansy norm one %12.8f,  inf %12.8f,  max %12.8f,  fro %12.8f\n",
                (int) m, (int) n, dnorm_one, dnorm_inf, dnorm_max, dnorm_fro );
        error = fabs(snorm_one - dnorm_one) / dnorm_one;
        status |= ! (error < 1e-6);
    }
    if ( status ) {
        printf( "**** failed ****\n" );
    }
    else {
        printf( "ok\n" );
    }
    printf( "\n" );
    
    return 1.125;
}


// ------------------------------------------------------------
int main( int argc, char** argv )
{
    magma_int_t m = 100;
    magma_int_t n = m;
    if ( argc > 1 ) {
        n = atoi( argv[1] );
    }
    
    float value;
    
    printf( "--------------------\n" );
    printf( "sizeof(void*) %lu, sizeof(RETURN_FLOAT) %lu\n\n",
            sizeof(void*), sizeof(RETURN_FLOAT) );
    
    // can repeat multiple times, but shows same results every time
    for( magma_int_t i=0; i < 1; ++i ) {
        value = test( m, n );
        printf( "value %.4f\n\n", value );
    }
    
    return 0;
}
