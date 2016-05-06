/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zgeqr2_gpu.cpp normal z -> d, Mon May  2 23:31:15 2016
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
   -- Testing dgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();
    
    // Constants
    const double c_zero    = MAGMA_D_ZERO;
    const double c_one     = MAGMA_D_ONE;
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const double d_one     =  1;
    const double d_neg_one = -1;

    // Local variables
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           Anorm, error=0, error2=0;
    double *h_A, *h_R, *tau, *dtau, *h_work, tmp[1];
    magmaDouble_ptr d_A;
    magmaDouble_ptr dwork;
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% M     N     CPU Gflop/s (ms)    GPU Gflop/s (ms)    |R - Q^H*A|   |I - Q^H*Q|\n");
    printf("%%==============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_DGEQRF( M, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_dgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_D_REAL( tmp[0] );
            
            TESTING_MALLOC_CPU( tau,    double, min_mn );
            TESTING_MALLOC_CPU( h_A,    double, n2     );
            TESTING_MALLOC_CPU( h_work, double, lwork  );
            
            TESTING_MALLOC_PIN( h_R,    double, n2     );
            
            TESTING_MALLOC_DEV( d_A,    double, ldda*N );
            TESTING_MALLOC_DEV( dtau,   double, min_mn );
            TESTING_MALLOC_DEV( dwork,  double, min_mn );
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_dlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_dsetmatrix( M, N, h_R, lda, d_A, ldda, opts.queue );
            
            // warmup
            if ( opts.warmup ) {
                magma_dgeqr2_gpu( M, N, d_A, ldda, dtau, dwork, opts.queue, &info );
                magma_dsetmatrix( M, N, h_R, lda, d_A, ldda, opts.queue );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( opts.queue );

            magma_dgeqr2_gpu( M, N, d_A, ldda, dtau, dwork, opts.queue, &info );

            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_dgeqr2_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the result, following zqrt01 except using the reduced Q.
               This works for any M,N (square, tall, wide).
               =================================================================== */
            if ( opts.check ) {
                magma_dgetmatrix( M, N, d_A, ldda, h_R, lda, opts.queue );
                magma_dgetvector( min_mn, dtau, 1, tau, 1, opts.queue );
                
                magma_int_t ldq = M;
                magma_int_t ldr = min_mn;
                double *Q, *R;
                double *cwork;
                TESTING_MALLOC_CPU( Q,     double, ldq*min_mn );  // M by K
                TESTING_MALLOC_CPU( R,     double, ldr*N );       // K by N
                TESTING_MALLOC_CPU( cwork, double,             min_mn );
                
                // generate M by K matrix Q, where K = min(M,N)
                lapackf77_dlacpy( "Lower", &M, &min_mn, h_R, &lda, Q, &ldq );
                lapackf77_dorgqr( &M, &min_mn, &min_mn, Q, &ldq, tau, h_work, &lwork, &info );
                assert( info == 0 );
                
                // copy K by N matrix R
                lapackf77_dlaset( "Lower", &min_mn, &N, &c_zero, &c_zero, R, &ldr );
                lapackf77_dlacpy( "Upper", &min_mn, &N, h_R, &lda,        R, &ldr );
                
                // error = || R - Q^H*A || / (N * ||A||)
                blasf77_dgemm( "Conj", "NoTrans", &min_mn, &N, &M,
                               &c_neg_one, Q, &ldq, h_A, &lda, &c_one, R, &ldr );
                Anorm = lapackf77_dlange( "1", &M,      &N, h_A, &lda, cwork );
                error = lapackf77_dlange( "1", &min_mn, &N, R,   &ldr, cwork );
                if ( N > 0 && Anorm > 0 )
                    error /= (N*Anorm);
                
                // set R = I (K by K identity), then R = I - Q^H*Q
                // error = || I - Q^H*Q || / N
                lapackf77_dlaset( "Upper", &min_mn, &min_mn, &c_zero, &c_one, R, &ldr );
                blasf77_dsyrk( "Upper", "Conj", &min_mn, &M, &d_neg_one, Q, &ldq, &d_one, R, &ldr );
                error2 = safe_lapackf77_dlansy( "1", "Upper", &min_mn, R, &ldr, cwork );
                if ( N > 0 )
                    error2 /= N;
                
                TESTING_FREE_CPU( Q     );  Q     = NULL;
                TESTING_FREE_CPU( R     );  R     = NULL;
                TESTING_FREE_CPU( cwork );  cwork = NULL;
            }
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                lapackf77_dgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_dgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) M, (int) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time );
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) M, (int) N, gpu_perf, 1000.*gpu_time );
            }
            
            if ( opts.check ) {
                bool okay = (error < tol) && (error2 < tol);
                status += ! okay;
                printf("   %8.2e      %8.2e   %s\n",
                       error, error2, (okay ? "ok" : "failed"));
            }
            else {
                printf("     ---  \n");
            }
            
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_PIN( h_R   );
            
            TESTING_FREE_DEV( d_A   );
            TESTING_FREE_DEV( dtau  );
            TESTING_FREE_DEV( dwork );
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
