/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zgeqp3.cpp normal z -> d, Mon May  2 23:31:17 2016

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

#define REAL

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgeqp3
*/
int main( int argc, char** argv)
{
    TESTING_INIT();
    
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    double *h_A, *h_R, *tau, *h_work;
    magma_int_t *jpvt;
    magma_int_t M, N, n2, lda, lwork, j, info, min_mn, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% M     N     CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||A*P - Q*R||_F\n");
    printf("%%====================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            nb     = magma_get_dgeqp3_nb( M, N );
            gflops = FLOPS_DGEQRF( M, N ) / 1e9;
            
            lwork = ( N+1 )*nb;
            #ifdef REAL
            lwork += 2*N;
            #endif
            if ( opts.check )
                lwork = max( lwork, M*N + N );
            
            #ifdef COMPLEX
            double *rwork;
            TESTING_MALLOC_CPU( rwork,  double, 2*N );
            #endif
            TESTING_MALLOC_CPU( jpvt,   magma_int_t,        N      );
            TESTING_MALLOC_CPU( tau,    double, min_mn );
            TESTING_MALLOC_CPU( h_A,    double, n2     );
            
            TESTING_MALLOC_PIN( h_R,    double, n2     );
            TESTING_MALLOC_PIN( h_work, double, lwork  );
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                for( j = 0; j < N; j++)
                    jpvt[j] = 0;
                
                cpu_time = magma_wtime();
                lapackf77_dgeqp3( &M, &N, h_R, &lda, jpvt, tau, h_work, &lwork,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapack_dgeqp3 returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            lapackf77_dlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            for( j = 0; j < N; j++)
                jpvt[j] = 0;
            
            gpu_time = magma_wtime();
            magma_dgeqp3( M, N, h_R, lda, jpvt, tau, h_work, lwork,
                          #ifdef COMPLEX
                          rwork,
                          #endif
                          &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_dgeqp3 returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) M, (int) N, gpu_perf, gpu_time );
            }
            if ( opts.check ) {
                double error, ulp;
                ulp = lapackf77_dlamch( "P" );
                
                // Compute norm( A*P - Q*R )
                error = lapackf77_dqpt01( &M, &N, &min_mn, h_A, h_R, &lda,
                                          tau, jpvt, h_work, &lwork );
                error *= ulp;
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf("     ---  \n");
            }
            
            #ifdef COMPLEX
            TESTING_FREE_CPU( rwork );
            #endif
            TESTING_FREE_CPU( jpvt );
            TESTING_FREE_CPU( tau  );
            TESTING_FREE_CPU( h_A  );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
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
