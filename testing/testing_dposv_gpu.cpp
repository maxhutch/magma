/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zposv_gpu.cpp, normal z -> d, Sun Nov 20 20:20:34 2016
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
   -- Testing dposv_gpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    double          error, Rnorm, Anorm, Xnorm, *work;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_B, *h_X;
    magmaDouble_ptr d_A, d_B;
    magma_int_t N, lda, ldb, ldda, lddb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%===============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = ldb = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            lddb = ldda;
            gflops = ( FLOPS_DPOTRF( N ) + FLOPS_DPOTRS( N, opts.nrhs ) ) / 1e9;
            
            TESTING_CHECK( magma_dmalloc_cpu( &h_A, lda*N         ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_B, ldb*opts.nrhs ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_X, ldb*opts.nrhs ));
            TESTING_CHECK( magma_dmalloc_cpu( &work, N ));
            
            TESTING_CHECK( magma_dmalloc( &d_A, ldda*N         ));
            TESTING_CHECK( magma_dmalloc( &d_B, lddb*opts.nrhs ));
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            sizeA = lda*N;
            sizeB = ldb*opts.nrhs;
            lapackf77_dlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_dlarnv( &ione, ISEED, &sizeB, h_B );
            magma_dmake_hpd( N, h_A, lda );
            
            magma_dsetmatrix( N, N,         h_A, lda, d_A, ldda, opts.queue );
            magma_dsetmatrix( N, opts.nrhs, h_B, lda, d_B, lddb, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_dposv_gpu( opts.uplo, N, opts.nrhs, d_A, ldda, d_B, lddb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_dpotrf_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            /* =====================================================================
               Residual
               =================================================================== */
            magma_dgetmatrix( N, opts.nrhs, d_B, lddb, h_X, ldb, opts.queue );
            
            Anorm = lapackf77_dlange("I", &N, &N,         h_A, &lda, work);
            Xnorm = lapackf77_dlange("I", &N, &opts.nrhs, h_X, &ldb, work);
            
            blasf77_dgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &opts.nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_B, &ldb );
            
            Rnorm = lapackf77_dlange("I", &N, &opts.nrhs, h_B, &ldb, work);
            error = Rnorm/(N*Anorm*Xnorm);
            status += ! (error < tol);
            
            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_dposv( lapack_uplo_const(opts.uplo), &N, &opts.nrhs, h_A, &lda, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_dposv returned error %lld: %s.\n",
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
