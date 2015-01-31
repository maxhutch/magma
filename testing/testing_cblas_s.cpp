/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver

       @generated from testing_cblas_z.cpp normal z -> s, Fri Jan 30 19:00:24 2015
       @author Mark Gates
       
       These tests ensure that the MAGMA wrappers around (CPU) CBLAS calls are
       correct.
       This is derived from the testing_blas_s.cpp code that checks MAGMA's
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

#define blasf77_sasum FORTRAN_NAME( sasum, SASUM )
#define blasf77_snrm2 FORTRAN_NAME( snrm2, SNRM2 )
#define blasf77_sdot  FORTRAN_NAME( sdot,  SDOT  )
#define blasf77_sdot  FORTRAN_NAME( sdot,  SDOT  )

#ifdef __cplusplus
extern "C" {
#endif

float blasf77_sasum( const magma_int_t* n,
                       const float* x, const magma_int_t* incx );

float blasf77_snrm2( const magma_int_t* n,
                       const float* x, const magma_int_t* incx );

float blasf77_sdot( const magma_int_t* n,
                                  const float* x, const magma_int_t* incx,
                                  const float* y, const magma_int_t* incy );

float blasf77_sdot( const magma_int_t* n,
                                  const float* x, const magma_int_t* incx,
                                  const float* y, const magma_int_t* incy );

#ifdef __cplusplus
}  // extern "C"
#endif


// ----------------------------------------
float gTol = 0;

const char* isok( float diff, float error )
{
    if ( diff == 0 && error < gTol ) {
        return "ok";
    }
    else {
        return "failed";
    }
}

void output( const char* routine, float diff, float error )
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
    
    float  *A, *B;
    float diff, error;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t m, n, k, size, maxn, ld;
    float x2_m, x2_c;  // real x for magma, cblas/fortran blas respectively
    float x_m, x_c;  // x for magma, cblas/fortran blas respectively
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    opts.tolerance = max( 100., opts.tolerance );
    float tol = opts.tolerance * lapackf77_slamch("E");
    gTol = tol;
    
    printf( "!! Calling these CBLAS and Fortran BLAS sometimes crashes (segfault), which !!\n"
            "!! is why we use wrappers. It does not necesarily indicate a bug in MAGMA.  !!\n"
            "\n"
            "Diff  compares MAGMA wrapper        to CBLAS and BLAS function; should be exactly 0.\n"
            "Error compares MAGMA implementation to CBLAS and BLAS function; should be ~ machine epsilon.\n"
            "\n" );
    
    float total_diff  = 0.;
    float total_error = 0.;
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
        magma_smalloc_pinned( &A,  size );  assert( A   != NULL );
        magma_smalloc_pinned( &B,  size );  assert( B   != NULL );
        
        // initialize matrices
        lapackf77_slarnv( &ione, ISEED, &size, A );
        lapackf77_slarnv( &ione, ISEED, &size, B );
        
        printf( "Level 1 BLAS ----------------------------------------------------------\n" );
        
        
        // ----- test SASUM
        // get one-norm of column j of A
        if ( incx > 0 && incx == incy ) {  // positive, no incy
            diff  = 0;
            error = 0;
            for( int j = 0; j < k; ++j ) {
                x_m = magma_cblas_sasum( m, A(0,j), incx );
                
                x_c = cblas_sasum( m, A(0,j), incx );
                diff += fabs( x_m - x_c );
                
                x_c = blasf77_sasum( &m, A(0,j), &incx );
                error += fabs( (x_m - x_c) / (m*x_c) );
            }
            output( "sasum", diff, error );
            total_diff  += diff;
            total_error += error;
        }
        
        // ----- test SNRM2
        // get two-norm of column j of A
        if ( incx > 0 && incx == incy ) {  // positive, no incy
            diff  = 0;
            error = 0;
            for( int j = 0; j < k; ++j ) {
                x_m = magma_cblas_snrm2( m, A(0,j), incx );
                
                x_c = cblas_snrm2( m, A(0,j), incx );
                diff += fabs( x_m - x_c );
                
                x_c = blasf77_snrm2( &m, A(0,j), &incx );
                error += fabs( (x_m - x_c) / (m*x_c) );
            }
            output( "snrm2", diff, error );
            total_diff  += diff;
            total_error += error;
        }
        
        // ----- test SDOT
        // dot columns, Aj^H Bj
        diff  = 0;
        error = 0;
        for( int j = 0; j < k; ++j ) {
            // MAGMA implementation, not just wrapper
            x2_m = magma_cblas_sdot( m, A(0,j), incx, B(0,j), incy );
            
            // crashes on MKL 11.1.2, ILP64
            #if ! defined( MAGMA_WITH_MKL )
                #ifdef COMPLEX
                cblas_sdot_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
                #else
                x2_c = cblas_sdot( m, A(0,j), incx, B(0,j), incy );
                #endif
                error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            #endif
            
            // crashes on MacOS 10.9
            #if ! defined( __APPLE__ )
                x2_c = blasf77_sdot( &m, A(0,j), &incx, B(0,j), &incy );
                error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            #endif
        }
        output( "sdot", diff, error );
        total_diff  += diff;
        total_error += error;
        total_error += error;
        
        // ----- test SDOT
        // dot columns, Aj^T * Bj
        diff  = 0;
        error = 0;
        for( int j = 0; j < k; ++j ) {
            // MAGMA implementation, not just wrapper
            x2_m = magma_cblas_sdot( m, A(0,j), incx, B(0,j), incy );
            
            // crashes on MKL 11.1.2, ILP64
            #if ! defined( MAGMA_WITH_MKL )
                #ifdef COMPLEX
                cblas_sdot_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
                #else
                x2_c = cblas_sdot( m, A(0,j), incx, B(0,j), incy );
                #endif
                error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            #endif
            
            // crashes on MacOS 10.9
            #if ! defined( __APPLE__ )
                x2_c = blasf77_sdot( &m, A(0,j), &incx, B(0,j), &incy );
                error += fabs( x2_m - x2_c ) / fabs( m*x2_c );
            #endif
        }
        output( "sdot", diff, error );
        total_diff  += diff;
        total_error += error;
        
        // tell user about disabled functions
        #if defined( MAGMA_WITH_MKL )
            printf( "cblas_sdot and cblas_sdot disabled with MKL (segfaults)\n" );
        #endif
        
        #if defined( __APPLE__ )
            printf( "blasf77_sdot and blasf77_sdot disabled on MacOS (segfaults)\n" );
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
