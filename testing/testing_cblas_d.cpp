/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver

       @generated from testing_cblas_z.cpp normal z -> d, Fri Jan 30 19:00:24 2015
       @author Mark Gates
       
       These tests ensure that the MAGMA wrappers around (CPU) CBLAS calls are
       correct.
       This is derived from the testing_blas_d.cpp code that checks MAGMA's
       wrappers around CUBLAS.
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cblas.h>

// make sure that asserts are enabled
#undef NDEBUG
#include <assert.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define REAL

#define A(i,j)   &A[  (i) + (j)*ld ]
#define B(i,j)   &B[  (i) + (j)*ld ]


// ----------------------------------------
// These may not be portable to different Fortran implementations,
// hence why MAGMA does not rely on them.

#define blasf77_dasum FORTRAN_NAME( dasum, DASUM )
#define blasf77_dnrm2 FORTRAN_NAME( dnrm2, DNRM2 )
#define blasf77_ddot  FORTRAN_NAME( ddot,  DDOT  )
#define blasf77_ddot  FORTRAN_NAME( ddot,  DDOT  )

#ifdef __cplusplus
extern "C" {
#endif

double blasf77_dasum( const magma_int_t* n,
                       const double* x, const magma_int_t* incx );

double blasf77_dnrm2( const magma_int_t* n,
                       const double* x, const magma_int_t* incx );

double blasf77_ddot( const magma_int_t* n,
                                  const double* x, const magma_int_t* incx,
                                  const double* y, const magma_int_t* incy );

double blasf77_ddot( const magma_int_t* n,
                                  const double* x, const magma_int_t* incx,
                                  const double* y, const magma_int_t* incy );

#ifdef __cplusplus
}  // extern "C"
#endif


// ----------------------------------------
double gTol = 0;

const char* isok( double diff, double error )
{
    if ( diff == 0 && error < gTol ) {
        return "ok";
    }
    else {
        return "failed";
    }
}

void output( const char* routine, double diff, double error )
{
    bool ok = (diff == 0 && error < gTol);
    printf( "%-8s                                            %8.3g   %8.3g   %s\n",
            routine, diff, error, (ok ? "ok" : "failed") );
}



// ----------------------------------------
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    //real_Double_t   t_m, t_c, t_f;
    magma_int_t ione = 1;
    
    double  *A, *B;
    double diff, error;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t m, n, k, size, maxn, ld;
    double x2_m, x2_c;  // real x for magma, cblas/fortran blas respectively
    double x_m, x_c;  // x for magma, cblas/fortran blas respectively
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    opts.tolerance = max( 100., opts.tolerance );
    double tol = opts.tolerance * lapackf77_dlamch("E");
    gTol = tol;
    
    printf( "!! Calling these CBLAS and Fortran BLAS sometimes crashes (segfault), which !!\n"
            "!! is why we use wrappers. It does not necesarily indicate a bug in MAGMA.  !!\n"
            "\n"
            "Diff  compares MAGMA wrapper        to CBLAS and BLAS function; should be exactly 0.\n"
            "Error compares MAGMA implementation to CBLAS and BLAS function; should be ~ machine epsilon.\n"
            "\n" );
    
    double total_diff  = 0.;
    double total_error = 0.;
    int inc[] = { 1 };  //{ -2, -1, 1, 2 };  //{ 1 };  //{ -1, 1 };
    int ninc = sizeof(inc)/sizeof(*inc);
    
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        m = opts.msize[itest];
        n = opts.nsize[itest];
        k = opts.ksize[itest];
        
    for( int iincx = 0; iincx < ninc; ++iincx ) {
        magma_int_t incx = inc[iincx];
        
    for( int iincy = 0; iincy < ninc; ++iincy ) {
        magma_int_t incy = inc[iincy];
        
        printf("=========================================================================\n");
        printf( "m=%d, n=%d, k=%d, incx = %d, incy = %d\n",
                (int) m, (int) n, (int) k, (int) incx, (int) incy );
        printf( "Function              MAGMA     CBLAS     BLAS        Diff      Error\n"
                "                      msec      msec      msec\n" );
        
        // allocate matrices
        // over-allocate so they can be any combination of
        // {m,n,k} * {abs(incx), abs(incy)} by
        // {m,n,k} * {abs(incx), abs(incy)}
        maxn = max( max( m, n ), k ) * max( abs(incx), abs(incy) );
        ld = max( 1, maxn );
        size = ld*maxn;
        magma_dmalloc_pinned( &A,  size );  assert( A   != NULL );
        magma_dmalloc_pinned( &B,  size );  assert( B   != NULL );
        
        // initialize matrices
        lapackf77_dlarnv( &ione, ISEED, &size, A );
        lapackf77_dlarnv( &ione, ISEED, &size, B );
        
        printf( "Level 1 BLAS ----------------------------------------------------------\n" );
        
        
        // ----- test DASUM
        // get one-norm of column j of A
        if ( incx > 0 && incx == incy ) {  // positive, no incy
            diff  = 0;
            error = 0;
            for( int j = 0; j < k; ++j ) {
                x_m = magma_cblas_dasum( m, A(0,j), incx );
                
                x_c = cblas_dasum( m, A(0,j), incx );
                diff += fabs( x_m - x_c );
                
                x_c = blasf77_dasum( &m, A(0,j), &incx );
                error += fabs( (x_m - x_c) / (m*x_c) );
            }
            output( "dasum", diff, error );
            total_diff  += diff;
            total_error += error;
        }
        
        // ----- test DNRM2
        // get two-norm of column j of A
        if ( incx > 0 && incx == incy ) {  // positive, no incy
            diff  = 0;
            error = 0;
            for( int j = 0; j < k; ++j ) {
                x_m = magma_cblas_dnrm2( m, A(0,j), incx );
                
                x_c = cblas_dnrm2( m, A(0,j), incx );
                diff += fabs( x_m - x_c );
                
                x_c = blasf77_dnrm2( &m, A(0,j), &incx );
                error += fabs( (x_m - x_c) / (m*x_c) );
            }
            output( "dnrm2", diff, error );
            total_diff  += diff;
            total_error += error;
        }
        
        // ----- test DDOT
        // dot columns, Aj^H Bj
        diff  = 0;
        error = 0;
        for( int j = 0; j < k; ++j ) {
            // MAGMA implementation, not just wrapper
            x2_m = magma_cblas_ddot( m, A(0,j), incx, B(0,j), incy );
            
            // crashes on MKL 11.1.2, ILP64
            #if ! defined( MAGMA_WITH_MKL )
                #ifdef COMPLEX
                cblas_ddot_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
                #else
                x2_c = cblas_ddot( m, A(0,j), incx, B(0,j), incy );
                #endif
                error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            #endif
            
            // crashes on MacOS 10.9
            #if ! defined( __APPLE__ )
                x2_c = blasf77_ddot( &m, A(0,j), &incx, B(0,j), &incy );
                error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            #endif
        }
        output( "ddot", diff, error );
        total_diff  += diff;
        total_error += error;
        total_error += error;
        
        // ----- test DDOT
        // dot columns, Aj^T * Bj
        diff  = 0;
        error = 0;
        for( int j = 0; j < k; ++j ) {
            // MAGMA implementation, not just wrapper
            x2_m = magma_cblas_ddot( m, A(0,j), incx, B(0,j), incy );
            
            // crashes on MKL 11.1.2, ILP64
            #if ! defined( MAGMA_WITH_MKL )
                #ifdef COMPLEX
                cblas_ddot_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
                #else
                x2_c = cblas_ddot( m, A(0,j), incx, B(0,j), incy );
                #endif
                error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            #endif
            
            // crashes on MacOS 10.9
            #if ! defined( __APPLE__ )
                x2_c = blasf77_ddot( &m, A(0,j), &incx, B(0,j), &incy );
                error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            #endif
        }
        output( "ddot", diff, error );
        total_diff  += diff;
        total_error += error;
        
        // tell user about disabled functions
        #if defined( MAGMA_WITH_MKL )
            printf( "cblas_ddot and cblas_ddot disabled with MKL (segfaults)\n" );
        #endif
        
        #if defined( __APPLE__ )
            printf( "blasf77_ddot and blasf77_ddot disabled on MacOS (segfaults)\n" );
        #endif
            
        // cleanup
        magma_free_pinned( A );
        magma_free_pinned( B );
        fflush( stdout );
    }}}  // itest, incx, incy
    
    // TODO use average error?
    printf( "sum diffs  = %8.2g, MAGMA wrapper        compared to CBLAS and Fortran BLAS; should be exactly 0.\n"
            "sum errors = %8.2e, MAGMA implementation compared to CBLAS and Fortran BLAS; should be ~ machine epsilon.\n\n",
            total_diff, total_error );
    if ( total_diff != 0. ) {
        printf( "some tests failed diff == 0.; see above.\n" );
    }
    else {
        printf( "all tests passed diff == 0.\n" );
    }
    
    TESTING_FINALIZE();
    
    int status = (total_diff != 0.);
    return status;
}
