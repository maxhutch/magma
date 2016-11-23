/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zgegqr_gpu.cpp, normal z -> s, Sun Nov 20 20:20:35 2016
       @author Stan Tomov

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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgegqr
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float           error, e1, e2, e3, e4, e5, *work;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float c_one     = MAGMA_S_ONE;
    float c_zero    = MAGMA_S_ZERO;
    float *h_A, *h_R, *tau, *dtau, *h_work, *h_rwork, tmp[1], unused[1];

    magmaFloat_ptr d_A, dwork;
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn;
    magma_int_t ione     = 1, ldwork;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    // versions 1...4 are valid
    if (opts.version < 1 || opts.version > 4) {
        printf("Unknown version %lld; exiting\n", (long long) opts.version );
        return -1;
    }
    
    float tol = 10. * opts.tolerance * lapackf77_slamch("E");
    
    printf("%% version %lld\n", (long long) opts.version );
    printf("%% M     N     CPU Gflop/s (ms)    GPU Gflop/s (ms)      ||I-Q'Q||_F / M     ||I-Q'Q||_I / M    ||A-Q R||_I\n");
    printf("%%                                                       MAGMA  /  LAPACK    MAGMA  /  LAPACK\n");
    printf("%%=========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];

            if (N > 128) {
                printf("%5lld %5lld   skipping because sgegqr requires N <= 128\n",
                        (long long) M, (long long) N);
                continue;
            }
            if (M < N) {
                printf("%5lld %5lld   skipping because sgegqr requires M >= N\n",
                        (long long) M, (long long) N);
                continue;
            }

            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_SGEQRF( M, N ) / 1e9 +  FLOPS_SORGQR( M, N, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_sgeqrf( &M, &N, unused, &M, unused, tmp, &lwork, &info );
            lwork = (magma_int_t)MAGMA_S_REAL( tmp[0] );
            lwork = max(lwork, 3*N*N);
            
            ldwork = N*N;
            if (opts.version == 2) {
                ldwork = 3*N*N + min_mn + 2;
            }

            TESTING_CHECK( magma_smalloc_pinned( &tau,    min_mn ));
            TESTING_CHECK( magma_smalloc_pinned( &h_work, lwork  ));
            TESTING_CHECK( magma_smalloc_pinned( &h_rwork, lwork  ));

            TESTING_CHECK( magma_smalloc_cpu( &h_A,   n2     ));
            TESTING_CHECK( magma_smalloc_cpu( &h_R,   n2     ));
            TESTING_CHECK( magma_smalloc_cpu( &work,  M      ));
            
            TESTING_CHECK( magma_smalloc( &d_A,   ldda*N ));
            TESTING_CHECK( magma_smalloc( &dtau,  min_mn ));
            TESTING_CHECK( magma_smalloc( &dwork, ldwork ));

            /* Initialize the matrix */
            lapackf77_slarnv( &ione, ISEED, &n2, h_A );

            lapackf77_slacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_ssetmatrix( M, N, h_R, lda, d_A, ldda, opts.queue );
            
            // warmup
            if ( opts.warmup ) {
                magma_sgegqr_gpu( 1, M, N, d_A, ldda, dwork, h_work, &info );
                magma_ssetmatrix( M, N, h_R, lda, d_A, ldda, opts.queue );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( opts.queue );
            magma_sgegqr_gpu( opts.version, M, N, d_A, ldda, dwork, h_rwork, &info );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_sgegqr returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            magma_sgetmatrix( M, N, d_A, ldda, h_R, lda, opts.queue );

            // Regenerate R
            // blasf77_sgemm("t", "n", &N, &N, &M, &c_one, h_R, &lda, h_A, &lda, &c_zero, h_rwork, &N);
            // magma_sprint(N, N, h_work, N);

            blasf77_strmm("r", "u", "n", "n", &M, &N, &c_one, h_rwork, &N, h_R, &lda);
            blasf77_saxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
            e5 = lapackf77_slange("i", &M, &N, h_R, &lda, work) /
                 lapackf77_slange("i", &M, &N, h_A, &lda, work);
            magma_sgetmatrix( M, N, d_A, ldda, h_R, lda, opts.queue );
 
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();

                /* Orthogonalize on the CPU */
                lapackf77_sgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                lapackf77_sorgqr(&M, &N, &N, h_A, &lda, tau, h_work, &lwork, &info );

                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_sorgqr returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                blasf77_sgemm("c", "n", &N, &N, &M, &c_one, h_R, &lda, h_R, &lda, &c_zero, h_work, &N);
                for (magma_int_t ii = 0; ii < N*N; ii += N+1 ) {
                    h_work[ii] = MAGMA_S_SUB(h_work[ii], c_one);
                }
                e1 = lapackf77_slange("f", &N, &N, h_work, &N, work) / N;
                e3 = lapackf77_slange("i", &N, &N, h_work, &N, work) / N;

                blasf77_sgemm("c", "n", &N, &N, &M, &c_one, h_A, &lda, h_A, &lda, &c_zero, h_work, &N);
                for (magma_int_t ii = 0; ii < N*N; ii += N+1 ) {
                    h_work[ii] = MAGMA_S_SUB(h_work[ii], c_one);
                }
                e2 = lapackf77_slange("f", &N, &N, h_work, &N, work) / N;
                e4 = lapackf77_slange("i", &N, &N, h_work, &N, work) / N;

                if (opts.version != 4)
                    error = e1;
                else
                    error = e1 / (10.*max(M,N));

                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e / %8.2e   %8.2e / %8.2e   %8.2e  %s\n",
                       (long long) M, (long long) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                       e1, e2, e3, e4, e5,
                       (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (long long) M, (long long) N, gpu_perf, 1000.*gpu_time );
            }
            
            magma_free_pinned( tau    );
            magma_free_pinned( h_work );
            magma_free_pinned( h_rwork );
           
            magma_free_cpu( h_A  );
            magma_free_cpu( h_R  );
            magma_free_cpu( work );

            magma_free( d_A   );
            magma_free( dtau  );
            magma_free( dwork );

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
