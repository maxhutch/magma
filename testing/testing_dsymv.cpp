/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zhemv.cpp normal z -> d, Mon May  2 23:31:05 2016
       
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


int main(int argc, char **argv)
{
    TESTING_INIT();

    const double c_neg_one = MAGMA_D_NEG_ONE;
    const magma_int_t        ione      = 1;
    
    real_Double_t   atomics_perf=0, atomics_time=0;
    real_Double_t   gflops, magma_perf=0, magma_time=0, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double          magma_error=0, atomics_error=0, cublas_error, work[1];
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t N, lda, ldda, sizeA, sizeX, sizeY, blocks, ldwork;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t nb   = 64;
    double alpha = MAGMA_D_MAKE(  1.5, -2.3 );
    double beta  = MAGMA_D_MAKE( -0.6,  0.8 );
    double *A, *X, *Y, *Yatomics, *Ycublas, *Ymagma;
    magmaDouble_ptr dA, dX, dY, dwork;
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N   MAGMA Gflop/s (ms)    Atomics Gflop/s      CUBLAS Gflop/s       CPU Gflop/s   MAGMA error  Atomics    CUBLAS\n");
    printf("%%=====================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            sizeA  = N*lda;
            sizeX  = N*incx;
            sizeY  = N*incy;
            gflops = FLOPS_DSYMV( N ) / 1e9;
            
            TESTING_MALLOC_CPU( A,        double, sizeA );
            TESTING_MALLOC_CPU( X,        double, sizeX );
            TESTING_MALLOC_CPU( Y,        double, sizeY );
            TESTING_MALLOC_CPU( Yatomics, double, sizeY );
            TESTING_MALLOC_CPU( Ycublas,  double, sizeY );
            TESTING_MALLOC_CPU( Ymagma,   double, sizeY );
            
            TESTING_MALLOC_DEV( dA, double, ldda*N );
            TESTING_MALLOC_DEV( dX, double, sizeX );
            TESTING_MALLOC_DEV( dY, double, sizeY );
            
            blocks = magma_ceildiv( N, nb );
            ldwork = ldda*blocks;
            TESTING_MALLOC_DEV( dwork, double, ldwork );
            
            magmablas_dlaset( MagmaFull, ldwork, 1, MAGMA_D_NAN, MAGMA_D_NAN, dwork, ldwork, opts.queue );
            magmablas_dlaset( MagmaFull, ldda,   N, MAGMA_D_NAN, MAGMA_D_NAN, dA,    ldda,   opts.queue );
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &sizeA, A );
            magma_dmake_symmetric( N, A, lda );
            
            // should not use data from the opposite triangle -- fill with NAN to check
            magma_int_t N1 = N-1;
            if ( opts.uplo == MagmaUpper ) {
                lapackf77_dlaset( "Lower", &N1, &N1, &MAGMA_D_NAN, &MAGMA_D_NAN, &A[1], &lda );
            }
            else {
                lapackf77_dlaset( "Upper", &N1, &N1, &MAGMA_D_NAN, &MAGMA_D_NAN, &A[lda], &lda );
            }
            
            lapackf77_dlarnv( &ione, ISEED, &sizeX, X );
            lapackf77_dlarnv( &ione, ISEED, &sizeY, Y );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_dsetmatrix( N, N, A, lda, dA, ldda, opts.queue );
            magma_dsetvector( N, X, incx, dX, incx, opts.queue );
            magma_dsetvector( N, Y, incy, dY, incy, opts.queue );
            
            cublas_time = magma_sync_wtime( opts.queue );
            #ifdef HAVE_CUBLAS
                cublasDsymv( opts.handle, cublas_uplo_const(opts.uplo),
                             N, &alpha, dA, ldda, dX, incx, &beta, dY, incy );
            #else
                magma_dsymv( opts.uplo, N, alpha, dA, 0, ldda, dX, 0, incx, beta, dY, 0, incy, opts.queue );
            #endif
            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_dgetvector( N, dY, incy, Ycublas, incy, opts.queue );
            
            /* =====================================================================
               Performs operation using CUBLAS - using atomics
               =================================================================== */
            #ifdef HAVE_CUBLAS
                cublasSetAtomicsMode( opts.handle, CUBLAS_ATOMICS_ALLOWED );
                magma_dsetvector( N, Y, incy, dY, incy, opts.queue );
                
                // sync on queue doesn't work -- need device sync or use NULL stream -- bug in CUBLAS?
                atomics_time = magma_sync_wtime( NULL /*opts.queue*/ );
                cublasDsymv( opts.handle, cublas_uplo_const(opts.uplo),
                             N, &alpha, dA, ldda, dX, incx, &beta, dY, incy );
                atomics_time = magma_sync_wtime( NULL /*opts.queue*/ ) - atomics_time;
                atomics_perf = gflops / atomics_time;
                
                magma_dgetvector( N, dY, incy, Yatomics, incy, opts.queue );
                cublasSetAtomicsMode( opts.handle, CUBLAS_ATOMICS_NOT_ALLOWED );
            #endif
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_dsetvector( N, Y, incy, dY, incy, opts.queue );
                
                magma_time = magma_sync_wtime( opts.queue );
                if ( opts.version == 1 ) {
                    magmablas_dsymv_work( opts.uplo, N, alpha, dA, ldda, dX, incx, beta, dY, incy, dwork, ldwork, opts.queue );
                }
                else {
                    // non-work interface (has added overhead)
                    magmablas_dsymv( opts.uplo, N, alpha, dA, ldda, dX, incx, beta, dY, incy, opts.queue );
                }
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_dgetvector( N, dY, incy, Ymagma, incy, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            cpu_time = magma_wtime();
            blasf77_dsymv( lapack_uplo_const(opts.uplo), &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            blasf77_daxpy( &N, &c_neg_one, Y, &incy, Ycublas, &incy );
            cublas_error = lapackf77_dlange( "M", &N, &ione, Ycublas, &N, work ) / N;
            
            #ifdef HAVE_CUBLAS
                blasf77_daxpy( &N, &c_neg_one, Y, &incy, Yatomics, &incy );
                atomics_error = lapackf77_dlange( "M", &N, &ione, Yatomics, &N, work ) / N;
                
                blasf77_daxpy( &N, &c_neg_one, Y, &incy, Ymagma, &incy );
                magma_error = lapackf77_dlange( "M", &N, &ione, Ymagma, &N, work ) / N;
            #endif
            
            bool okay = (magma_error < tol && cublas_error < tol && atomics_error < tol);
            status += ! okay;
            printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e   %s\n",
                   (int) N,
                   magma_perf,   1000.*magma_time,
                   atomics_perf, 1000.*atomics_time,
                   cublas_perf,  1000.*cublas_time,
                   cpu_perf,     1000.*cpu_time,
                   magma_error, cublas_error, atomics_error,
                   (okay ? "ok" : "failed"));
            
            TESTING_FREE_CPU( A );
            TESTING_FREE_CPU( X );
            TESTING_FREE_CPU( Y );
            TESTING_FREE_CPU( Ycublas  );
            TESTING_FREE_CPU( Yatomics );
            TESTING_FREE_CPU( Ymagma   );
            
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
