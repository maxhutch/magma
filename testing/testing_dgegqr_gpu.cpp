/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zgegqr_gpu.cpp normal z -> d, Fri Jan 30 19:00:25 2015
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
   -- Testing dgegqr
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           e, e1, e2, e3, e4, e5, *work;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double c_one     = MAGMA_D_ONE;
    double c_zero    = MAGMA_D_ZERO;
    double *h_A, *h_R, *tau, *dtau, *h_work, *h_rwork, tmp[1];

    magmaDouble_ptr d_A, dwork;
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
    
    double tol = 10. * opts.tolerance * lapackf77_dlamch("E");
    
    printf("version %d\n", (int) opts.version );
    printf("  M     N     CPU GFlop/s (ms)    GPU GFlop/s (ms)      ||I-Q'Q||_F / M     ||I-Q'Q||_I / M    ||A-Q R||_I\n");
    printf("                                                        MAGMA  /  LAPACK    MAGMA  /  LAPACK\n");
    printf("==========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];

            if (N > 128) {
                printf("%5d %5d   skipping because dgegqr requires N <= 128\n",
                        (int) M, (int) N);
                continue;
            }
            if (M < N) {
                printf("%5d %5d   skipping because dgegqr requires M >= N\n",
                        (int) M, (int) N);
                continue;
            }

            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_DGEQRF( M, N ) / 1e9 +  FLOPS_DORGQR( M, N, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_dgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_D_REAL( tmp[0] );
            lwork = max(lwork, 3*N*N);
            
            ldwork = N*N;
            if (opts.version == 2) {
                ldwork = 3*N*N + min_mn + 2;
            }

            TESTING_MALLOC_PIN( tau,    double, min_mn );
            TESTING_MALLOC_PIN( h_work, double, lwork  );
            TESTING_MALLOC_PIN(h_rwork, double, lwork  );

            TESTING_MALLOC_CPU( h_A,   double, n2     );
            TESTING_MALLOC_CPU( h_R,   double, n2     );
            TESTING_MALLOC_CPU( work,  double,             M      );
            
            TESTING_MALLOC_DEV( d_A,   double, ldda*N );
            TESTING_MALLOC_DEV( dtau,  double, min_mn );
            TESTING_MALLOC_DEV( dwork, double, ldwork );

            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );

            lapackf77_dlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_dsetmatrix( M, N, h_R, lda, d_A, ldda );
            
            // warmup
            if ( opts.warmup ) {
                magma_dgegqr_gpu( 1, M, N, d_A, ldda, dwork, h_work, &info );
                magma_dsetmatrix( M, N, h_R, lda, d_A, ldda );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( 0 );
            magma_dgegqr_gpu( opts.version, M, N, d_A, ldda, dwork, h_rwork, &info );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_dgegqr returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));

            magma_dgetmatrix( M, N, d_A, ldda, h_R, M );

            // Regenerate R
            // blasf77_dgemm("t", "n", &N, &N, &M, &c_one, h_R, &M, h_A, &M, &c_zero, h_rwork, &N);
            // magma_dprint(N, N, h_work, N);

            blasf77_dtrmm("r", "u", "n", "n", &M, &N, &c_one, h_rwork, &N, h_R, &M);
            blasf77_daxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
            e5 = lapackf77_dlange("i", &M, &N, h_R, &M, work) /
                 lapackf77_dlange("i", &M, &N, h_A, &lda, work);
            magma_dgetmatrix( M, N, d_A, ldda, h_R, M );
 
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();

                /* Orthogonalize on the CPU */
                lapackf77_dgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                lapackf77_dorgqr(&M, &N, &N, h_A, &lda, tau, h_work, &lwork, &info );

                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_dorgqr returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                blasf77_dgemm("c", "n", &N, &N, &M, &c_one, h_R, &M, h_R, &M, &c_zero, h_work, &N);
                for(int ii = 0; ii < N*N; ii += N+1 ) {
                    h_work[ii] = MAGMA_D_SUB(h_work[ii], c_one);
                }
                e1 = lapackf77_dlange("f", &N, &N, h_work, &N, work) / N;
                e3 = lapackf77_dlange("i", &N, &N, h_work, &N, work) / N;

                blasf77_dgemm("c", "n", &N, &N, &M, &c_one, h_A, &M, h_A, &M, &c_zero, h_work, &N);
                for(int ii = 0; ii < N*N; ii += N+1 ) {
                    h_work[ii] = MAGMA_D_SUB(h_work[ii], c_one);
                }
                e2 = lapackf77_dlange("f", &N, &N, h_work, &N, work) / N;
                e4 = lapackf77_dlange("i", &N, &N, h_work, &N, work) / N;

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
