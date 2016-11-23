/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zposv_gpu.cpp, normal z -> c, Sun Nov 20 20:20:35 2016
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cposv_gpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    float          error, Rnorm, Anorm, Xnorm, *work;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex *h_A, *h_B, *h_X;
    magmaFloatComplex_ptr d_A, d_B;
    magma_int_t N, lda, ldb, ldda, lddb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%===============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = ldb = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            lddb = ldda;
            gflops = ( FLOPS_CPOTRF( N ) + FLOPS_CPOTRS( N, opts.nrhs ) ) / 1e9;
            
            TESTING_CHECK( magma_cmalloc_cpu( &h_A, lda*N         ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_B, ldb*opts.nrhs ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_X, ldb*opts.nrhs ));
            TESTING_CHECK( magma_smalloc_cpu( &work, N ));
            
            TESTING_CHECK( magma_cmalloc( &d_A, ldda*N         ));
            TESTING_CHECK( magma_cmalloc( &d_B, lddb*opts.nrhs ));
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            sizeA = lda*N;
            sizeB = ldb*opts.nrhs;
            lapackf77_clarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_clarnv( &ione, ISEED, &sizeB, h_B );
            magma_cmake_hpd( N, h_A, lda );
            
            magma_csetmatrix( N, N,         h_A, lda, d_A, ldda, opts.queue );
            magma_csetmatrix( N, opts.nrhs, h_B, lda, d_B, lddb, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_cposv_gpu( opts.uplo, N, opts.nrhs, d_A, ldda, d_B, lddb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_cpotrf_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            /* =====================================================================
               Residual
               =================================================================== */
            magma_cgetmatrix( N, opts.nrhs, d_B, lddb, h_X, ldb, opts.queue );
            
            Anorm = lapackf77_clange("I", &N, &N,         h_A, &lda, work);
            Xnorm = lapackf77_clange("I", &N, &opts.nrhs, h_X, &ldb, work);
            
            blasf77_cgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &opts.nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_B, &ldb );
            
            Rnorm = lapackf77_clange("I", &N, &opts.nrhs, h_B, &ldb, work);
            error = Rnorm/(N*Anorm*Xnorm);
            status += ! (error < tol);
            
            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_cposv( lapack_uplo_const(opts.uplo), &N, &opts.nrhs, h_A, &lda, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_cposv returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                printf( "%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) opts.nrhs, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (error < tol ? "ok" : "failed"));
            }
            else {
                printf( "%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) opts.nrhs, gpu_perf, gpu_time,
                        error, (error < tol ? "ok" : "failed"));
            }
            
            magma_free_cpu( h_A  );
            magma_free_cpu( h_B  );
            magma_free_cpu( h_X  );
            magma_free_cpu( work );
            
            magma_free( d_A  );
            magma_free( d_B  );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
