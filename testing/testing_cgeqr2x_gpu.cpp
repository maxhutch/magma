/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Stan Tomov
       @author Mark Gates

       @generated from testing/testing_zgeqr2x_gpu.cpp normal z -> c, Mon May  2 23:31:15 2016
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
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    /* Constants */
    magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    float d_one     = MAGMA_D_ONE;
    float d_neg_one = MAGMA_D_NEG_ONE;
    
    /* Local variables */
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float           Anorm, error, error2, diff, terr, rwork[1];
    magmaFloatComplex *h_A, *h_T, *h_R, *tau, *h_work, tmp[1];
    magmaFloatComplex_ptr d_A, d_T, ddA, dtau;
    magmaFloat_ptr dwork;

    magma_int_t M, N, lda, ldda, lwork, size, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    #define BLOCK_SIZE 64

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("%% version %d\n", (int) opts.version );
    printf("%% It's okay if |Q - Q_lapack| is large; MAGMA and LAPACK\n"
           "%% just chose different Householder reflectors, both valid.\n\n");
    
    printf("%%   M     N    CPU Gflop/s (ms)    GPU Gflop/s (ms)   |R - Q^H*A|   |I - Q^H*Q|   |T - T_magma|   |Q - Q_magma|\n");
    printf("%%==============================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M     = opts.msize[itest];
            N     = opts.nsize[itest];

            if (N > 128) {
                printf("%5d %5d   skipping because cgeqr2x requires N <= 128\n",
                        (int) M, (int) N);
                continue;
            }
            if (M < N) {
                printf("%5d %5d   skipping because cgeqr2x requires M >= N\n",
                        (int) M, (int) N);
                continue;
            }

            min_mn = min( M, N );
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            // TODO: flops should be GEQRF + LARFT (whatever that is)
            gflops = (FLOPS_CGEQRF( M, N ) + FLOPS_CGEQRT( M, N )) / 1e9;

            lwork = -1;
            lapackf77_cgeqrf( &M, &N, NULL, &M, NULL, tmp, &lwork, &info );
            lwork = (magma_int_t)MAGMA_C_REAL( tmp[0] );
            lwork = max( lwork, N*N );
        
            /* Allocate memory for the matrix */
            TESTING_MALLOC_CPU( tau,    magmaFloatComplex, min_mn );
            TESTING_MALLOC_CPU( h_A,    magmaFloatComplex, lda*N  );
            TESTING_MALLOC_CPU( h_T,    magmaFloatComplex, N*N    );
            TESTING_MALLOC_CPU( h_work, magmaFloatComplex, lwork );
            
            TESTING_MALLOC_PIN( h_R,    magmaFloatComplex, lda*N  );
            
            TESTING_MALLOC_DEV( d_A,    magmaFloatComplex, ldda*N );
            TESTING_MALLOC_DEV( d_T,    magmaFloatComplex, N*N    );
            TESTING_MALLOC_DEV( ddA,    magmaFloatComplex, N*N    );
            TESTING_MALLOC_DEV( dtau,   magmaFloatComplex, min_mn );
            
            TESTING_MALLOC_DEV( dwork,  float, max(5*min_mn, (BLOCK_SIZE*2+2)*min_mn) );
            
            magmablas_claset( MagmaFull, N, N, c_zero, c_zero, ddA, N, opts.queue );
            magmablas_claset( MagmaFull, N, N, c_zero, c_zero, d_T, N, opts.queue );
            
            /* Initialize the matrix */
            size = lda*N;
            lapackf77_clarnv( &ione, ISEED, &size, h_A );
            lapackf77_clacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_csetmatrix( M, N, h_R, lda, d_A, ldda, opts.queue );
    
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( opts.queue );
    
            if (opts.version == 1) {
                magma_cgeqr2x_gpu(  M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info );
            }
            else if (opts.version == 2) {
                magma_cgeqr2x2_gpu( M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info );
            }
            else if (opts.version == 3) {
                magma_cgeqr2x3_gpu( M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info );
            }
            else {
                /*
                  Going through NULL stream is faster
                  Going through any stream is slower
                  Doing two streams in parallel is slower than doing them sequentially
                  Queuing happens on the NULL stream - user defined buffers are smaller?
                */
                magma_cgeqr2x4_gpu( M, N, d_A, ldda, dtau, d_T, ddA, dwork, opts.queue, &info );
            }
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;

            if (info != 0) {
                printf("magma_cgeqr2x_gpu version %d returned error %d: %s.\n",
                       (int) opts.version, (int) info, magma_strerror( info ));
            }
            else if ( opts.check ) {
                /* =====================================================================
                   Check the result, following zqrt01 except using the reduced Q.
                   This works for any M,N (square, tall, wide).
                   =================================================================== */
                magma_cgetmatrix( M, N, d_A, ldda, h_R, lda, opts.queue );
                magma_cgetmatrix( N, N, ddA, N,    h_T, N, opts.queue );
                magma_cgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn, opts.queue );
                // Restore the upper triangular part of A before the check
                lapackf77_clacpy( "Upper", &N, &N, h_T, &N, h_R, &lda );

                magma_int_t ldq = M;
                magma_int_t ldr = min_mn;
                magmaFloatComplex *Q, *R;
                float *work;
                TESTING_MALLOC_CPU( Q,    magmaFloatComplex, ldq*min_mn );  // M by K
                TESTING_MALLOC_CPU( R,    magmaFloatComplex, ldr*N );       // K by N
                TESTING_MALLOC_CPU( work, float,             min_mn );
                
                // generate M by K matrix Q, where K = min(M,N)
                lapackf77_clacpy( "Lower", &M, &min_mn, h_R, &lda, Q, &ldq );
                lapackf77_cungqr( &M, &min_mn, &min_mn, Q, &ldq, tau, h_work, &lwork, &info );
                assert( info == 0 );
                
                // copy K by N matrix R
                lapackf77_claset( "Lower", &min_mn, &N, &c_zero, &c_zero, R, &ldr );
                lapackf77_clacpy( "Upper", &min_mn, &N, h_R, &lda,        R, &ldr );
                
                // error = || R - Q^H*A || / (N * ||A||)
                blasf77_cgemm( "Conj", "NoTrans", &min_mn, &N, &M,
                               &c_neg_one, Q, &ldq, h_A, &lda, &c_one, R, &ldr );
                Anorm = lapackf77_clange( "1", &M,      &N, h_A, &lda, work );
                error = lapackf77_clange( "1", &min_mn, &N, R,   &ldr, work );
                if ( N > 0 && Anorm > 0 )
                    error /= (N*Anorm);
                
                // set R = I (K by K identity), then R = I - Q^H*Q
                // error = || I - Q^H*Q || / N
                lapackf77_claset( "Upper", &min_mn, &min_mn, &c_zero, &c_one, R, &ldr );
                blasf77_cherk( "Upper", "Conj", &min_mn, &M, &d_neg_one, Q, &ldq, &d_one, R, &ldr );
                error2 = safe_lapackf77_clanhe( "1", "Upper", &min_mn, R, &ldr, work );
                if ( N > 0 )
                    error2 /= N;
                
                TESTING_FREE_CPU( Q    );  Q    = NULL;
                TESTING_FREE_CPU( R    );  R    = NULL;
                TESTING_FREE_CPU( work );  work = NULL;

                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                lapackf77_cgeqrf( &M, &N, h_A, &lda, tau, h_work, &lwork, &info );
                lapackf77_clarft( MagmaForwardStr, MagmaColumnwiseStr,
                                  &M, &N, h_A, &lda, tau, h_work, &N );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_cgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }

                /* =====================================================================
                   Check the result compared to LAPACK
                   Okay if these are different -- just chose different Householder reflectors
                   =================================================================== */
                size = lda*N;
                blasf77_caxpy( &size, &c_neg_one, h_A, &ione, h_R, &ione );
                Anorm = lapackf77_clange( "M", &M, &N, h_A, &lda, rwork );
                diff =  lapackf77_clange( "M", &M, &N, h_R, &lda, rwork );
                if ( Anorm > 0) {
                    diff /= (N*Anorm);
                }
                
                /* =====================================================================
                   Check if T is correct
                   =================================================================== */
                // Recompute T in h_work for d_A (magma), in case it is different than h_A (lapack)
                magma_cgetmatrix( M, N, d_A, ldda, h_R, lda, opts.queue );
                magma_cgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn, opts.queue );
                lapackf77_clarft( MagmaForwardStr, MagmaColumnwiseStr,
                                  &M, &N, h_R, &lda, tau, h_work, &N );
                
                magma_cgetmatrix( N, N, d_T, N, h_T, N, opts.queue );
                size = N*N;
                blasf77_caxpy( &size, &c_neg_one, h_work, &ione, h_T, &ione );
                Anorm = lapackf77_clantr( "F", "U", "N", &N, &N, h_work, &N, rwork );
                terr  = lapackf77_clantr( "F", "U", "N", &N, &N, h_T,    &N, rwork );
                if (Anorm > 0) {
                    terr /= Anorm;
                }
                
                bool okay = (error < tol) && (error2 < tol) && (terr < tol);
                status += ! okay;
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e      %8.2e      %8.2e        %8.2e   %s\n",
                       (int) M, (int) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                       error, error2, terr, diff,
                       (okay ? "ok" : "failed"));
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (int) M, (int) N, gpu_perf, 1000.*gpu_time);
            }
            
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_T    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_PIN( h_R    );
        
            TESTING_FREE_DEV( d_A   );
            TESTING_FREE_DEV( d_T   );
            TESTING_FREE_DEV( ddA   );
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
