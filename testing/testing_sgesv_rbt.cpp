/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zgesv_rbt.cpp, normal z -> s, Sun Nov 20 20:20:36 2016
       @author Mark Gates
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
   -- Testing sgesv_rbt
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    float          error, Rnorm, Anorm, Xnorm, *work;
    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float *h_A, *h_LU, *h_B, *h_X;
    magma_int_t *ipiv;
    magma_int_t N, nrhs, lda, ldb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    nrhs = opts.nrhs;
    
    printf("%% ngpu %lld\n", (long long) opts.ngpu );
    printf("%%   N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%===============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldb    = lda;
            gflops = ( FLOPS_SGETRF( N, N ) + FLOPS_SGETRS( N, nrhs ) ) / 1e9;
            
            TESTING_CHECK( magma_smalloc_cpu( &h_A,  lda*N    ));
            TESTING_CHECK( magma_smalloc_cpu( &h_LU, lda*N    ));
            TESTING_CHECK( magma_smalloc_cpu( &h_B,  ldb*nrhs ));
            TESTING_CHECK( magma_smalloc_cpu( &h_X,  ldb*nrhs ));
            TESTING_CHECK( magma_smalloc_cpu( &work, N        ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, N        ));
            
            /* Initialize the matrices */
            sizeA = lda*N;
            sizeB = ldb*nrhs;
            lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_slarnv( &ione, ISEED, &sizeB, h_B );
            
            // copy A to LU and B to X; save A and B for residual
            lapackf77_slacpy( "F", &N, &N,    h_A, &lda, h_LU, &lda );
            lapackf77_slacpy( "F", &N, &nrhs, h_B, &ldb, h_X,  &ldb );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_sgesv_rbt(MagmaTrue, N, nrhs, h_LU, lda, h_X, ldb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_sgesv_rbt returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            for (int i = 0; i < N; i++)
                ipiv[i] = i+1;

            //=====================================================================
            // Residual
            //=====================================================================
            Anorm = lapackf77_slange("I", &N, &N,    h_A, &lda, work);
            Xnorm = lapackf77_slange("I", &N, &nrhs, h_X, &ldb, work);
            
            blasf77_sgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldb,
                           &c_neg_one, h_B, &ldb);
            
            Rnorm = lapackf77_slange("I", &N, &nrhs, h_B, &ldb, work);
            error = Rnorm/(N*Anorm*Xnorm);
            status += ! (error < tol);
            
            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_sgesv( &N, &nrhs, h_A, &lda, ipiv, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_sgesv returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                printf( "%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) nrhs, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (error < tol ? "ok" : "failed"));
            }
            else {
                printf( "%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N, (long long) nrhs, gpu_perf, gpu_time,
                        error, (error < tol ? "ok" : "failed"));
            }
            
            magma_free_cpu( h_A  );
            magma_free_cpu( h_LU );
            magma_free_cpu( h_B  );
            magma_free_cpu( h_X  );
            magma_free_cpu( work );
            magma_free_cpu( ipiv );
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
