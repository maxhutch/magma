/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from testing_zhemv.cpp normal z -> s, Fri Jul 18 17:34:22 2014
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_s

int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    float          magma_error, cublas_error, work[1];
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t N, lda, ldda, sizeA, sizeX, sizeY, blocks, ldwork;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t nb   = 64;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  1.5, -2.3 );
    float beta  = MAGMA_S_MAKE( -0.6,  0.8 );
    float *A, *X, *Y, *Ycublas, *Ymagma;
    float *dA, *dX, *dY, *dwork;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    float tol = opts.tolerance * lapackf77_slamch("E");

    printf("uplo = %s\n", lapack_uplo_const(opts.uplo) );
    if ( opts.uplo == MagmaUpper ) {
        printf("** for uplo=MagmaUpper, magmablas_ssymv simply calls cublas_ssymv.\n");
    }
    printf("    N   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  CUBLAS error\n");
    printf("=============================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldda   = ((N + 31)/32)*32;
            sizeA  = N*lda;
            sizeX  = N*incx;
            sizeY  = N*incy;
            gflops = FLOPS_SSYMV( N ) / 1e9;
            
            TESTING_MALLOC_CPU( A,       float, sizeA );
            TESTING_MALLOC_CPU( X,       float, sizeX );
            TESTING_MALLOC_CPU( Y,       float, sizeY );
            TESTING_MALLOC_CPU( Ycublas, float, sizeY );
            TESTING_MALLOC_CPU( Ymagma,  float, sizeY );
            
            TESTING_MALLOC_DEV( dA, float, ldda*N );
            TESTING_MALLOC_DEV( dX, float, sizeX );
            TESTING_MALLOC_DEV( dY, float, sizeY );
            
            blocks = (N + nb - 1) / nb;
            ldwork = ldda*blocks;
            TESTING_MALLOC_DEV( dwork, float, ldwork );
            
            magmablas_slaset( MagmaFull, ldwork, 1, MAGMA_S_NAN, MAGMA_S_NAN, dwork, ldwork );
            magmablas_slaset( MagmaFull, ldda,   N, MAGMA_S_NAN, MAGMA_S_NAN, dA,    ldda   );
            
            /* Initialize the matrix */
            lapackf77_slarnv( &ione, ISEED, &sizeA, A );
            magma_smake_symmetric( N, A, lda );
            lapackf77_slarnv( &ione, ISEED, &sizeX, X );
            lapackf77_slarnv( &ione, ISEED, &sizeY, Y );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_ssetmatrix( N, N, A, lda, dA, ldda );
            magma_ssetvector( N, X, incx, dX, incx );
            magma_ssetvector( N, Y, incy, dY, incy );
            
            cublas_time = magma_sync_wtime( 0 );
            cublasSsymv( handle, cublas_uplo_const(opts.uplo),
                         N, &alpha, dA, ldda, dX, incx, &beta, dY, incy );
            cublas_time = magma_sync_wtime( 0 ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_sgetvector( N, dY, incy, Ycublas, incy );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_ssetvector( N, Y, incy, dY, incy );
            
            //magma_sprint_gpu( ldda, blocks, dwork, ldda );
            
            magma_time = magma_sync_wtime( 0 );
            magmablas_ssymv_work( opts.uplo, N, alpha, dA, ldda, dX, incx, beta, dY, incy, dwork, ldwork );
            // TODO provide option to test non-work interface
            //magmablas_ssymv( opts.uplo, N, alpha, dA, ldda, dX, incx, beta, dY, incy );
            magma_time = magma_sync_wtime( 0 ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_sgetvector( N, dY, incy, Ymagma, incy );
            
            //magma_sprint_gpu( ldda, blocks, dwork, ldda );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            cpu_time = magma_wtime();
            blasf77_ssymv( lapack_uplo_const(opts.uplo), &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            blasf77_saxpy( &N, &c_neg_one, Y, &incy, Ymagma, &incy );
            magma_error = lapackf77_slange( "M", &N, &ione, Ymagma, &N, work ) / N;
            
            blasf77_saxpy( &N, &c_neg_one, Y, &incy, Ycublas, &incy );
            cublas_error = lapackf77_slange( "M", &N, &ione, Ycublas, &N, work ) / N;
            
            printf("%5d   %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e     %8.2e   %s\n",
                   (int) N,
                   magma_perf,  1000.*magma_time,
                   cublas_perf, 1000.*cublas_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, cublas_error,
                   (magma_error < tol && cublas_error < tol ? "ok" : "failed"));
            status += ! (magma_error < tol && cublas_error < tol);
            
            TESTING_FREE_CPU( A );
            TESTING_FREE_CPU( X );
            TESTING_FREE_CPU( Y );
            TESTING_FREE_CPU( Ycublas );
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

    TESTING_FINALIZE();
    return status;
}
