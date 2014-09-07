/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    double           error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_R, *tau, *h_work, tmp[1];
    magma_int_t M, N, n2, lda, lwork, info, min_mn, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1}, ISEED2[4];
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );

    magma_int_t status = 0;
    double tol, eps = lapackf77_dlamch("E");
    tol = opts.tolerance * eps;

    opts.lapack |= ( opts.check == 2 );  // check (-c2) implies lapack (-l)
    
    printf("ngpu %d\n", (int) opts.ngpu );
    if ( opts.check == 1 ) {
        printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R-Q'A||_1 / (M*||A||_1) ||I-Q'Q||_1 / M\n");
        printf("===============================================================================================\n");
    } else {
        printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
        printf("=======================================================================\n");
    }
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            nb     = magma_get_zgeqrf_nb(M);
            gflops = FLOPS_ZGEQRF( M, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_zgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            lwork = max( lwork, max( N*nb, 2*nb*nb ));
            
            TESTING_MALLOC_CPU( tau,    magmaDoubleComplex, min_mn );
            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_work, magmaDoubleComplex, lwork  );
            
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, n2     );
            
            /* Initialize the matrix */
            for ( int j=0; j<4; j++ )
                ISEED2[j] = ISEED[j]; // saving seeds
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            
            magma_zgeqrf(M, N, h_R, lda, tau, h_work, lwork, &info);
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zgeqrf(M, N, h_R, lda, tau, h_work, lwork, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgeqrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                magmaDoubleComplex *tau;
                TESTING_MALLOC_CPU( tau, magmaDoubleComplex, min_mn );
                cpu_time = magma_wtime();
                lapackf77_zgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                TESTING_FREE_CPU( tau );
            }

            if ( opts.check == 1 && M >= N ) {
                /* =====================================================================
                   Check the result -- zqrt02 requires M >= N
                   =================================================================== */
                magma_int_t lwork = n2+N;
                magmaDoubleComplex *h_W1, *h_W2, *h_W3;
                double *h_RW, results[2];

                TESTING_MALLOC_CPU( h_W1, magmaDoubleComplex, n2    ); // Q
                TESTING_MALLOC_CPU( h_W2, magmaDoubleComplex, n2    ); // R
                TESTING_MALLOC_CPU( h_W3, magmaDoubleComplex, lwork ); // WORK
                TESTING_MALLOC_CPU( h_RW, double, M );  // RWORK
                lapackf77_zlarnv( &ione, ISEED2, &n2, h_A );
                lapackf77_zqrt02( &M, &N, &min_mn, h_A, h_R, h_W1, h_W2, &lda, tau, h_W3, &lwork,
                                  h_RW, results );
                results[0] *= eps;
                results[1] *= eps;

                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e                  %8.2e",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, results[0], results[1] );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)    %8.2e                  %8.2e",
                           (int) M, (int) N, gpu_perf, gpu_time, results[0], results[1] );
                }
                // todo also check results[1] < tol?
                printf("   %s\n", (results[0] < tol ? "ok" : "failed"));
                status += ! (results[0] < tol);

                TESTING_FREE_CPU( h_W1 );
                TESTING_FREE_CPU( h_W2 );
                TESTING_FREE_CPU( h_W3 );
                TESTING_FREE_CPU( h_RW );
            }
            else if ( opts.check == 2 ) {
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                error = lapackf77_zlange("f", &M, &N, h_A, &lda, work);
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                error = lapackf77_zlange("f", &M, &N, h_R, &lda, work) / error;
                
                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)    %8.2e",
                           (int) M, (int) N, gpu_perf, gpu_time, error );
                }
                printf("   %s\n", (error < 2 * tol ? "ok" : "failed"));
                status += ! (error < 2 * tol);
            }
            else {
                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   ---",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---",
                           (int) M, (int) N, gpu_perf, gpu_time);
                }
                printf("%s\n", (opts.check != 0 ? "  (error check only for M >= N)" : ""));
            }
            
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_PIN( h_R    );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
