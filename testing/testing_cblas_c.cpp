/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver

       @generated from testing/testing_cblas_z.cpp normal z -> c, Mon May  2 23:31:07 2016
       @author Mark Gates
       
       These tests ensure that the MAGMA implementations of CBLAS routines
       are correct. (We no longer use wrappers.)
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

// make sure that asserts are enabled
#undef NDEBUG
#include <assert.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define COMPLEX

#define A(i,j)   &A[  (i) + (j)*ld ]
#define B(i,j)   &B[  (i) + (j)*ld ]


// ----------------------------------------
// These may not be portable to different Fortran implementations,
// hence why MAGMA does not rely on them.

#define blasf77_scasum FORTRAN_NAME( scasum, SCASUM )
#define blasf77_scnrm2 FORTRAN_NAME( scnrm2, SCNRM2 )
#define blasf77_cdotc  FORTRAN_NAME( cdotc,  CDOTC  )
#define blasf77_cdotu  FORTRAN_NAME( cdotu,  CDOTU  )

#ifdef __cplusplus
extern "C" {
#endif

float blasf77_scasum( const magma_int_t* n,
                       const magmaFloatComplex* x, const magma_int_t* incx );

float blasf77_scnrm2( const magma_int_t* n,
                       const magmaFloatComplex* x, const magma_int_t* incx );

magmaFloatComplex blasf77_cdotc( const magma_int_t* n,
                                  const magmaFloatComplex* x, const magma_int_t* incx,
                                  const magmaFloatComplex* y, const magma_int_t* incy );

magmaFloatComplex blasf77_cdotu( const magma_int_t* n,
                                  const magmaFloatComplex* x, const magma_int_t* incx,
                                  const magmaFloatComplex* y, const magma_int_t* incy );

#ifdef __cplusplus
}  // extern "C"
#endif


// ----------------------------------------
float gTol = 0;
magma_int_t gStatus = 0;

const float SKIPPED_FLAG = -1;

void output(
    const char* routine,
    magma_int_t m, magma_int_t n, magma_int_t k, magma_int_t incx, magma_int_t incy,
    float error_cblas, float error_fblas, float error_inline )
{
    // SKIPPED_FLAG indicates skipped, e.g., cdotc with MKL -- it isn't an error
    bool okay = (error_cblas  == SKIPPED_FLAG || error_cblas  < gTol) &&
                (error_fblas  == SKIPPED_FLAG || error_fblas  < gTol) &&
                (error_inline < gTol);
    gStatus += ! okay;
    
    printf( "%5d %5d %5d %5d %5d   %-8s",
            int(m), int(n), int(k), int(incx), int(incy), routine );
    
    if ( error_cblas == SKIPPED_FLAG )
        printf( "   %8s", "n/a" );
    else
        printf( "   %#8.3g", error_cblas );
    
    if ( error_fblas == SKIPPED_FLAG )
        printf( "       %8s", "n/a" );
    else
        printf( "       %#8.3g", error_fblas );
    
    printf( "       %#8.3g   %s\n", error_inline, (okay ? "ok" : "failed") );
}


// ----------------------------------------
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    //real_Double_t   t_m, t_c, t_f;
    magma_int_t ione = 1;
    
    magmaFloatComplex  *A, *B;
    float error_cblas, error_fblas, error_inline;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t i, j, k, m, n, size, maxn, ld;
    
    // complex x for magma, cblas, fortran, inline blas respectively
    magmaFloatComplex x2_m, x2_c, x2_f, x2_i;
    
    // real    x for magma, cblas, fortran, inline blas respectively
    float x_m, x_c, x_f, x_i;
    
    MAGMA_UNUSED( x_c  );
    MAGMA_UNUSED( x_f  );
    MAGMA_UNUSED( x2_c );
    MAGMA_UNUSED( x2_f );
    MAGMA_UNUSED( x2_m );
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    opts.tolerance = max( 100., opts.tolerance );
    float tol = opts.tolerance * lapackf77_slamch("E");
    gTol = tol;
    
    magma_int_t inc[] = { -2, -1, 1, 2 };  //{ 1 };  //{ -1, 1 };
    magma_int_t ninc = sizeof(inc)/sizeof(*inc);
    magma_int_t maxinc = 0;
    for( i=0; i < ninc; ++i ) {
        maxinc = max( maxinc, abs(inc[i]) );
    }
    
    printf( "!! Calling these CBLAS and Fortran BLAS sometimes crashes (segfaults), which !!\n"
            "!! is why we use wrappers. It does not necesarily indicate a bug in MAGMA.   !!\n"
            "!! If MAGMA_WITH_MKL or __APPLE__ are defined, known failures are skipped.   !!\n"
            "\n" );
    
    // tell user about disabled functions
    #ifndef HAVE_CBLAS
        printf( "n/a: HAVE_CBLAS not defined, so no cblas functions tested.\n\n" );
    #endif
    
    #if defined(MAGMA_WITH_MKL)
        printf( "n/a: cblas_cdotc, cblas_cdotu, blasf77_cdotc, and blasf77_cdotu are disabled with MKL, due to segfaults.\n\n" );
    #endif
    
    #if defined(__APPLE__)
        printf( "n/a: blasf77_cdotc and blasf77_cdotu are disabled on MacOS, due to segfaults.\n\n" );
    #endif
    
    printf( "%%                                          Error w.r.t.   Error w.r.t.   Error w.r.t.\n"
            "%%   M     N     K  incx  incy   Function   CBLAS          Fortran BLAS   inline\n"
            "%%====================================================================================\n" );
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        if ( itest > 0 ) {
            printf( "%%----------------------------------------------------------------------\n" );
        }
        
        m = opts.msize[itest];
        n = opts.nsize[itest];
        k = opts.ksize[itest];
        
        // allocate matrices
        // over-allocate so they can be any combination of
        // {m,n,k} * {abs(incx), abs(incy)} by
        // {m,n,k} * {abs(incx), abs(incy)}
        maxn = max( max( m, n ), k ) * maxinc;
        ld = max( 1, maxn );
        size = ld*maxn;
        TESTING_MALLOC_CPU( A, magmaFloatComplex, size );
        TESTING_MALLOC_CPU( B, magmaFloatComplex, size );
        
        // initialize matrices
        lapackf77_clarnv( &ione, ISEED, &size, A );
        lapackf77_clarnv( &ione, ISEED, &size, B );
        
        // ----- test SCASUM
        for( int iincx = 0; iincx < ninc; ++iincx ) {
            magma_int_t incx = inc[iincx];
            
            for( int iincy = 0; iincy < ninc; ++iincy ) {
                magma_int_t incy = inc[iincy];
                
                // get one-norm of column j of A
                if ( incx > 0 && incx == incy ) {  // positive, no incy
                    error_cblas  = 0;
                    error_fblas  = 0;
                    error_inline = 0;
                    for( j=0; j < k; ++j ) {
                        x_m = magma_cblas_scasum( m, A(0,j), incx );
                        
                        #ifdef HAVE_CBLAS
                            x_c = cblas_scasum( m, A(0,j), incx );
                            error_cblas = max( error_cblas, fabs(x_m - x_c) / fabs(m*x_c) );
                        #else
                            x_c = 0;
                            error_cblas = SKIPPED_FLAG;
                        #endif
                        
                        x_f = blasf77_scasum( &m, A(0,j), &incx );
                        error_fblas = max( error_fblas, fabs(x_m - x_f) / fabs(m*x_f) );
                        
                        // inline implementation
                        x_i = 0;
                        for( i=0; i < m; ++i ) {
                            x_i += MAGMA_C_ABS1( *A(i*incx,j) );  // |real(Aij)| + |imag(Aij)|
                        }
                        error_inline = max( error_inline, fabs(x_m - x_i) / fabs(m*x_i) );
                        
                        //printf( "scasum xm %.8e, xc %.8e, xf %.8e, xi %.8e\n", x_m, x_c, x_f, x_i );
                    }
                    output( "scasum", m, n, k, incx, incy, error_cblas, error_fblas, error_inline );
                }
            }
        }
        printf( "\n" );
        
        // ----- test SCNRM2
        // get two-norm of column j of A
        for( int iincx = 0; iincx < ninc; ++iincx ) {
            magma_int_t incx = inc[iincx];
            
            for( int iincy = 0; iincy < ninc; ++iincy ) {
                magma_int_t incy = inc[iincy];
                
                if ( incx > 0 && incx == incy ) {  // positive, no incy
                    error_cblas  = 0;
                    error_fblas  = 0;
                    error_inline = 0;
                    for( j=0; j < k; ++j ) {
                        x_m = magma_cblas_scnrm2( m, A(0,j), incx );
                        
                        #ifdef HAVE_CBLAS
                            x_c = cblas_scnrm2( m, A(0,j), incx );
                            error_cblas = max( error_cblas, fabs(x_m - x_c) / fabs(m*x_c) );
                        #else
                            x_c = 0;
                            error_cblas = SKIPPED_FLAG;
                        #endif
                        
                        x_f = blasf77_scnrm2( &m, A(0,j), &incx );
                        error_fblas = max( error_fblas, fabs(x_m - x_f) / fabs(m*x_f) );
                        
                        // inline implementation (poor -- doesn't scale)
                        x_i = 0;
                        for( i=0; i < m; ++i ) {
                            x_i += real( *A(i*incx,j) ) * real( *A(i*incx,j) )
                                +  imag( *A(i*incx,j) ) * imag( *A(i*incx,j) );
                            // same: real( conj( *A(i*incx,j) ) * *A(i*incx,j) );
                        }
                        x_i = sqrt( x_i );
                        error_inline = max( error_inline, fabs(x_m - x_i) / fabs(m*x_i) );
                        
                        //printf( "scnrm2 xm %.8e, xc %.8e, xf %.8e, xi %.8e\n", x_m, x_c, x_f, x_i );
                    }
                    output( "scnrm2", m, n, k, incx, incy, error_cblas, error_fblas, error_inline );
                }
            }
        }
        printf( "\n" );
        
        // ----- test CDOTC
        // dot columns, Aj^H Bj
        for( int iincx = 0; iincx < ninc; ++iincx ) {
            magma_int_t incx = inc[iincx];
            
            for( int iincy = 0; iincy < ninc; ++iincy ) {
                magma_int_t incy = inc[iincy];
                
                error_cblas  = 0;
                error_fblas  = 0;
                error_inline = 0;
                for( j=0; j < k; ++j ) {
                    // MAGMA implementation, not just wrapper
                    x2_m = magma_cblas_cdotc( m, A(0,j), incx, B(0,j), incy );
                    
                    // crashes with MKL 11.1.2, ILP64
                    #if defined(HAVE_CBLAS) && ! defined(MAGMA_WITH_MKL)
                        #ifdef COMPLEX
                        cblas_cdotc_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
                        #else
                        x2_c = cblas_cdotc( m, A(0,j), incx, B(0,j), incy );
                        #endif
                        error_cblas = max( error_cblas, fabs(x2_m - x2_c) / fabs(m*x2_c) );
                    #else
                        x2_c = MAGMA_C_ZERO;
                        error_cblas = SKIPPED_FLAG;
                    #endif
                    
                    // crashes with MKL 11.2.3 and MacOS 10.9
                    #if (! defined(COMPLEX) || ! defined(MAGMA_WITH_MKL)) && ! defined(__APPLE__)
                        x2_f = blasf77_cdotc( &m, A(0,j), &incx, B(0,j), &incy );
                        error_fblas = max( error_fblas, fabs(x2_m - x2_f) / fabs(m*x2_f) );
                    #else
                        x2_f = MAGMA_C_ZERO;
                        error_fblas = SKIPPED_FLAG;
                    #endif
                    
                    // inline implementation
                    x2_i = MAGMA_C_ZERO;
                    magma_int_t A_offset = (incx > 0 ? 0 : (-n + 1)*incx);
                    magma_int_t B_offset = (incy > 0 ? 0 : (-n + 1)*incy);
                    for( i=0; i < m; ++i ) {
                        x2_i += conj( *A(A_offset + i*incx,j) ) * *B(B_offset + i*incy,j);
                    }
                    error_inline = max( error_inline, fabs(x2_m - x2_i) / fabs(m*x2_i) );
                    
                    //printf( "cdotc xm %.8e + %.8ei, xc %.8e + %.8ei, xf %.8e + %.8ei, xi %.8e + %.8ei\n",
                    //        real(x2_m), imag(x2_m),
                    //        real(x2_c), imag(x2_c),
                    //        real(x2_f), imag(x2_f),
                    //        real(x2_i), imag(x2_i) );
                }
                output( "cdotc", m, n, k, incx, incy, error_cblas, error_fblas, error_inline );
            }
        }
        printf( "\n" );
        
        // ----- test CDOTU
        // dot columns, Aj^T * Bj
        for( int iincx = 0; iincx < ninc; ++iincx ) {
            magma_int_t incx = inc[iincx];
            
            for( int iincy = 0; iincy < ninc; ++iincy ) {
                magma_int_t incy = inc[iincy];
                
                error_cblas  = 0;
                error_fblas  = 0;
                error_inline = 0;
                for( j=0; j < k; ++j ) {
                    // MAGMA implementation, not just wrapper
                    x2_m = magma_cblas_cdotu( m, A(0,j), incx, B(0,j), incy );
                    
                    // crashes with MKL 11.1.2, ILP64
                    #if defined(HAVE_CBLAS) && ! defined(MAGMA_WITH_MKL)
                        #ifdef COMPLEX
                        cblas_cdotu_sub( m, A(0,j), incx, B(0,j), incy, &x2_c );
                        #else
                        x2_c = cblas_cdotu( m, A(0,j), incx, B(0,j), incy );
                        #endif
                        error_cblas = max( error_cblas, fabs(x2_m - x2_c) / fabs(m*x2_c) );
                    #else
                        x2_c = MAGMA_C_ZERO;
                        error_cblas = SKIPPED_FLAG;
                    #endif
                    
                    // crashes with MKL 11.2.3 and MacOS 10.9
                    #if (! defined(COMPLEX) || ! defined(MAGMA_WITH_MKL)) && ! defined(__APPLE__)
                        x2_f = blasf77_cdotu( &m, A(0,j), &incx, B(0,j), &incy );
                        error_fblas = max( error_fblas, fabs(x2_m - x2_f) / fabs(m*x2_f) );
                    #else
                        x2_f = MAGMA_C_ZERO;
                        error_fblas = SKIPPED_FLAG;
                    #endif
                    
                    // inline implementation
                    x2_i = MAGMA_C_ZERO;
                    magma_int_t A_offset = (incx > 0 ? 0 : (-n + 1)*incx);
                    magma_int_t B_offset = (incy > 0 ? 0 : (-n + 1)*incy);
                    for( i=0; i < m; ++i ) {
                        x2_i += *A(A_offset + i*incx,j) * *B(B_offset + i*incy,j);
                    }
                    error_inline = max( error_inline, fabs(x2_m - x2_i) / fabs(m*x2_i) );
                    
                    //printf( "cdotu xm %.8e + %.8ei, xc %.8e + %.8ei, xf %.8e + %.8ei, xi %.8e + %.8ei\n",
                    //        real(x2_m), imag(x2_m),
                    //        real(x2_c), imag(x2_c),
                    //        real(x2_f), imag(x2_f),
                    //        real(x2_i), imag(x2_i) );
                }
                output( "cdotu", m, n, k, incx, incy, error_cblas, error_fblas, error_inline );
            }
        }
        
        // cleanup
        TESTING_FREE_CPU( A );
        TESTING_FREE_CPU( B );
        fflush( stdout );
    }  // itest, incx, incy
    
    opts.cleanup();
    TESTING_FINALIZE();
    return gStatus;
}
