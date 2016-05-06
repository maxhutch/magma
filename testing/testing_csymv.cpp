/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates
       
       Note: [ds] precisions generated from testing_chemv.cu
       
       @generated from testing/testing_zsymv.cpp normal z -> c, Mon May  2 23:31:05 2016
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


int main(int argc, char **argv)
{
    TESTING_INIT();

    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    const magma_int_t        ione      = 1;
    
    real_Double_t   gflops, magma_perf=0, magma_time=0, cpu_perf, cpu_time;
    float          magma_error=0, work[1];
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t N, lda, ldda, sizeA, sizeX, sizeY, blocks, ldwork;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t nb   = 64;
    magmaFloatComplex alpha = MAGMA_C_MAKE(  1.5, -2.3 );
    magmaFloatComplex beta  = MAGMA_C_MAKE( -0.6,  0.8 );
    magmaFloatComplex *A, *X, *Y, *Ymagma;
    magmaFloatComplex_ptr dA, dX, dY, dwork;
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N   MAGMA Gflop/s (ms)  CPU Gflop/s (ms)  MAGMA error\n");
    printf("%%========================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            sizeA  = N*lda;
            sizeX  = N*incx;
            sizeY  = N*incy;
            gflops = FLOPS_CSYMV( N ) / 1e9;
            
            TESTING_MALLOC_CPU( A,       magmaFloatComplex, sizeA );
            TESTING_MALLOC_CPU( X,       magmaFloatComplex, sizeX );
            TESTING_MALLOC_CPU( Y,       magmaFloatComplex, sizeY );
            TESTING_MALLOC_CPU( Ymagma,  magmaFloatComplex, sizeY );
            
            TESTING_MALLOC_DEV( dA, magmaFloatComplex, ldda*N );
            TESTING_MALLOC_DEV( dX, magmaFloatComplex, sizeX );
            TESTING_MALLOC_DEV( dY, magmaFloatComplex, sizeY );
            
            blocks = magma_ceildiv( N, nb );
            ldwork = ldda*blocks;
            TESTING_MALLOC_DEV( dwork, magmaFloatComplex, ldwork );
            
            magmablas_claset( MagmaFull, ldwork, 1, MAGMA_C_NAN, MAGMA_C_NAN, dwork, ldwork, opts.queue );
            magmablas_claset( MagmaFull, ldda,   N, MAGMA_C_NAN, MAGMA_C_NAN, dA,    ldda,   opts.queue );
            
            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &sizeA, A );
            magma_cmake_hermitian( N, A, lda );
            
            // should not use data from the opposite triangle -- fill with NAN to check
            magma_int_t N1 = N-1;
            if ( opts.uplo == MagmaUpper ) {
                lapackf77_claset( "Lower", &N1, &N1, &MAGMA_C_NAN, &MAGMA_C_NAN, &A[1], &lda );
            }
            else {
                lapackf77_claset( "Upper", &N1, &N1, &MAGMA_C_NAN, &MAGMA_C_NAN, &A[lda], &lda );
            }
            
            lapackf77_clarnv( &ione, ISEED, &sizeX, X );
            lapackf77_clarnv( &ione, ISEED, &sizeY, Y );
            
            /* Note: CUBLAS does not implement csymv */
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_csetmatrix( N, N, A, lda, dA, ldda, opts.queue );
                magma_csetvector( N, X, incx, dX, incx, opts.queue );
                magma_csetvector( N, Y, incy, dY, incy, opts.queue );
                
                magma_time = magma_sync_wtime( opts.queue );
                if ( opts.version == 1 ) {
                    magmablas_csymv_work( opts.uplo, N, alpha, dA, ldda, dX, incx, beta, dY, incy, dwork, ldwork, opts.queue );
                }
                else {
                    // non-work interface (has added overhead)
                    magmablas_csymv( opts.uplo, N, alpha, dA, ldda, dX, incx, beta, dY, incy, opts.queue );
                }
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_cgetvector( N, dY, incy, Ymagma, incy, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_csymv( lapack_uplo_const(opts.uplo), &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            blasf77_caxpy( &N, &c_neg_one, Y, &incy, Ymagma, &incy );
            magma_error = lapackf77_clange( "M", &N, &ione, Ymagma, &N, work ) / N;
            
            printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                   (int) N,
                   magma_perf,  1000.*magma_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, (magma_error < tol ? "ok" : "failed"));
            status += ! (magma_error < tol);
            
            TESTING_FREE_CPU( A );
            TESTING_FREE_CPU( X );
            TESTING_FREE_CPU( Y );
            TESTING_FREE_CPU( Ymagma  );
            
            TESTING_FREE_DEV( dA );
            TESTING_FREE_DEV( dX );
            TESTING_FREE_DEV( dY );
            TESTING_FREE_DEV( dwork );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
