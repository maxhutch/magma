/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zgegqr_gpu.cpp normal z -> s, Fri Jan 30 19:00:25 2015
       @author Stan Tomov

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgegqr
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float           e, e1, e2, e3, e4, e5, *work;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float c_one     = MAGMA_S_ONE;
    float c_zero    = MAGMA_S_ZERO;
    float *h_A, *h_R, *tau, *dtau, *h_work, *h_rwork, tmp[1];

    magmaFloat_ptr d_A, dwork;
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn;
    magma_int_t ione     = 1, ldwork;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    // versions 1...4 are valid
    if (opts.version < 1 || opts.version > 4) {
        printf("Unknown version %d; exiting\n", (int) opts.version );
        return -1;
    }
    
    float tol = 10. * opts.tolerance * lapackf77_slamch("E");
    
    printf("version %d\n", (int) opts.version );
    printf("  M     N     CPU GFlop/s (ms)    GPU GFlop/s (ms)      ||I-Q'Q||_F / M     ||I-Q'Q||_I / M    ||A-Q R||_I\n");
    printf("                                                        MAGMA  /  LAPACK    MAGMA  /  LAPACK\n");
    printf("==========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];

            if (N > 128) {
                printf("%5d %5d   skipping because sgegqr requires N <= 128\n",
                        (int) M, (int) N);
                continue;
            }
            if (M < N) {
                printf("%5d %5d   skipping because sgegqr requires M >= N\n",
                        (int) M, (int) N);
                continue;
            }

            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_SGEQRF( M, N ) / 1e9 +  FLOPS_SORGQR( M, N, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_sgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_S_REAL( tmp[0] );
            lwork = max(lwork, 3*N*N);
            
            ldwork = N*N;
            if (opts.version == 2) {
                ldwork = 3*N*N + min_mn + 2;
            }

            TESTING_MALLOC_PIN( tau,    float, min_mn );
            TESTING_MALLOC_PIN( h_work, float, lwork  );
            TESTING_MALLOC_PIN(h_rwork, float, lwork  );

            TESTING_MALLOC_CPU( h_A,   float, n2     );
            TESTING_MALLOC_CPU( h_R,   float, n2     );
            TESTING_MALLOC_CPU( work,  float,             M      );
            
            TESTING_MALLOC_DEV( d_A,   float, ldda*N );
            TESTING_MALLOC_DEV( dtau,  float, min_mn );
            TESTING_MALLOC_DEV( dwork, float, ldwork );

            /* Initialize the matrix */
            lapackf77_slarnv( &ione, ISEED, &n2, h_A );

            lapackf77_slacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_ssetmatrix( M, N, h_R, lda, d_A, ldda );
            
            // warmup
            if ( opts.warmup ) {
                magma_sgegqr_gpu( 1, M, N, d_A, ldda, dwork, h_work, &info );
                magma_ssetmatrix( M, N, h_R, lda, d_A, ldda );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( 0 );
            magma_sgegqr_gpu( opts.version, M, N, d_A, ldda, dwork, h_rwork, &info );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_sgegqr returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));

            magma_sgetmatrix( M, N, d_A, ldda, h_R, M );

            // Regenerate R
            // blasf77_sgemm("t", "n", &N, &N, &M, &c_one, h_R, &M, h_A, &M, &c_zero, h_rwork, &N);
            // magma_sprint(N, N, h_work, N);

            blasf77_strmm("r", "u", "n", "n", &M, &N, &c_one, h_rwork, &N, h_R, &M);
            blasf77_saxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
            e5 = lapackf77_slange("i", &M, &N, h_R, &M, work) /
                 lapackf77_slange("i", &M, &N, h_A, &lda, work);
            magma_sgetmatrix( M, N, d_A, ldda, h_R, M );
 
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
                if (info != 0)
                    printf("lapackf77_sorgqr returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                blasf77_sgemm("c", "n", &N, &N, &M, &c_one, h_R, &M, h_R, &M, &c_zero, h_work, &N);
                for(int ii = 0; ii < N*N; ii += N+1 ) {
                    h_work[ii] = MAGMA_S_SUB(h_work[ii], c_one);
                }
                e1 = lapackf77_slange("f", &N, &N, h_work, &N, work) / N;
                e3 = lapackf77_slange("i", &N, &N, h_work, &N, work) / N;

                blasf77_sgemm("c", "n", &N, &N, &M, &c_one, h_A, &M, h_A, &M, &c_zero, h_work, &N);
                for(int ii = 0; ii < N*N; ii += N+1 ) {
                    h_work[ii] = MAGMA_S_SUB(h_work[ii], c_one);
                }
                e2 = lapackf77_slange("f", &N, &N, h_work, &N, work) / N;
                e4 = lapackf77_slange("i", &N, &N, h_work, &N, work) / N;

                if (opts.version != 4)
                    e = e1;
                else 
                    e = e1 / (10.*max(M,N));

                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e / %8.2e   %8.2e / %8.2e   %8.2e  %s\n",
                       (int) M, (int) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                       e1, e2, e3, e4, e5,
                       (e < tol ? "ok" : "failed"));
                status += ! (e < tol); 
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (int) M, (int) N, gpu_perf, 1000.*gpu_time );
            }
            
            TESTING_FREE_PIN( tau    );
            TESTING_FREE_PIN( h_work );
            TESTING_FREE_PIN( h_rwork );
           
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_R  );
            TESTING_FREE_CPU( work );

            TESTING_FREE_DEV( d_A   );
            TESTING_FREE_DEV( dtau  );
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
