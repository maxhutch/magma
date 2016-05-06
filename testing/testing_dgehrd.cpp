/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates

       @generated from testing/testing_zgehrd.cpp normal z -> d, Mon May  2 23:31:20 2016
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
   -- Testing dgehrd
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double *h_A, *h_R, *h_Q, *h_work, *tau, *twork, *T;
    magmaDouble_ptr dT;
    #ifdef COMPLEX
    double      *rwork;
    #endif
    double      eps, result[2];
    magma_int_t N, n2, lda, nb, lwork, ltwork, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    
    eps   = lapackf77_dlamch( "E" );
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% version %d, ngpu = %d\n", int(opts.version), int(abs_ngpu) );
    
    printf("%%   N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |A-QHQ^H|/N|A|   |I-QQ^H|/N\n");
    printf("%%==========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            nb     = magma_get_dgehrd_nb( N );
            // magma needs larger workspace than lapack, esp. multi-gpu verison
            lwork  = N*nb;
            if (opts.ngpu != 1) {
                lwork += N*nb*abs_ngpu;
            }
            gflops = FLOPS_DGEHRD( N ) / 1e9;
            
            TESTING_MALLOC_CPU( h_A,    double, n2    );
            TESTING_MALLOC_CPU( tau,    double, N     );
            TESTING_MALLOC_CPU( T,      double, nb*N  );  // for multi GPU
            
            TESTING_MALLOC_PIN( h_R,    double, n2    );
            TESTING_MALLOC_PIN( h_work, double, lwork );
            
            TESTING_MALLOC_DEV( dT,     double, nb*N  );  // for single GPU
            
            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.version == 1 ) {
                if ( opts.ngpu == 1 ) {
                    magma_dgehrd( N, ione, N, h_R, lda, tau, h_work, lwork, dT, &info );
                }
                else {
                    magma_dgehrd_m( N, ione, N, h_R, lda, tau, h_work, lwork, T, &info );
                }
            }
            else {
                // LAPACK-complaint arguments, no dT array
                printf( "magma_dgehrd2\n" );
                magma_dgehrd2( N, ione, N, h_R, lda, tau, h_work, lwork, &info );
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_dgehrd returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                ltwork = 2*N*N;
                TESTING_MALLOC_PIN( h_Q,   double, lda*N  );
                TESTING_MALLOC_CPU( twork, double, ltwork );
                #ifdef COMPLEX
                TESTING_MALLOC_CPU( rwork, double, N );
                #endif
                
                lapackf77_dlacpy( MagmaFullStr, &N, &N, h_R, &lda, h_Q, &lda );
                for( int j = 0; j < N-1; ++j )
                    for( int i = j+2; i < N; ++i )
                        h_R[i+j*lda] = MAGMA_D_ZERO;
                
                if ( opts.version == 1 ) {
                    if ( opts.ngpu != 1 ) {
                        magma_dsetmatrix( nb, N, T, nb, dT, nb, opts.queue );
                    }
                    magma_dorghr( N, ione, N, h_Q, lda, tau, dT, nb, &info );
                }
                else {
                    // for magma_dgehrd2, no dT array
                    lapackf77_dorghr( &N, &ione, &N, h_Q, &lda, tau, h_work, &lwork, &info );
                }
                if (info != 0) {
                    printf("magma_dorghr returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                    return -1;
                }
                lapackf77_dhst01( &N, &ione, &N,
                                  h_A, &lda, h_R, &lda,
                                  h_Q, &lda, twork, &ltwork,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  result );
                
                TESTING_FREE_PIN( h_Q   );
                TESTING_FREE_CPU( twork );
                #ifdef COMPLEX
                TESTING_FREE_CPU( rwork );
                #endif
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_dgehrd( &N, &ione, &N, h_A, &lda, tau, h_work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_dgehrd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) N, gpu_perf, gpu_time );
            }
            if ( opts.check ) {
                bool okay = (result[0]*eps < tol) && (result[1]*eps < tol);
                status += ! okay;
                printf("   %8.2e        %8.2e   %s\n",
                       result[0]*eps, result[1]*eps,
                       (okay ? "ok" : "failed") );
            }
            else {
                printf("     ---             ---\n");
            }
            
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( T      );
            
            TESTING_FREE_PIN( h_R    );
            TESTING_FREE_PIN( h_work );
            
            TESTING_FREE_DEV( dT     );
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
