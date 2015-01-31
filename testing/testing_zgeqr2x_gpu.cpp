/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

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
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, error2;

    magmaDoubleComplex  c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex *h_A, *h_T, *h_R, *tau, *h_work, tmp[1];
    magmaDoubleComplex_ptr d_A,  d_T, ddA, dtau;
    magmaDoubleComplex_ptr d_A2, d_T2, ddA2, dtau2;
    magmaDouble_ptr dwork, dwork2;

    magma_int_t M, N, lda, ldda, lwork, n2, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;

    #define BLOCK_SIZE 64

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol = 10. * opts.tolerance * lapackf77_dlamch("E");
    
    magma_queue_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    printf("version %d\n", (int) opts.version );
    printf("  M     N     CPU GFlop/s (ms)    GPU GFlop/s (ms)   ||R - Q^H*A||   ||R_T||\n");
    printf("=============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M     = opts.msize[itest];
            N     = opts.nsize[itest];

            if (N > 128) {
                printf("%5d %5d   skipping because zgeqr2x requires N <= 128\n",
                        (int) M, (int) N);
                continue;
            }
            if (M < N) {
                printf("%5d %5d   skipping because zgeqr2x requires M >= N\n",
                        (int) M, (int) N);
                continue;
            }

            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = (FLOPS_ZGEQRF( M, N ) + FLOPS_ZGEQRT( M, N )) / 1e9;

            /* Allocate memory for the matrix */
            TESTING_MALLOC_CPU( tau,   magmaDoubleComplex, min_mn );
            TESTING_MALLOC_CPU( h_A,   magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_T,   magmaDoubleComplex, N*N    );
        
            TESTING_MALLOC_PIN( h_R,   magmaDoubleComplex, n2     );
        
            TESTING_MALLOC_DEV( d_A,   magmaDoubleComplex, ldda*N );
            TESTING_MALLOC_DEV( d_T,   magmaDoubleComplex, N*N    );
            TESTING_MALLOC_DEV( ddA,   magmaDoubleComplex, N*N    );
            TESTING_MALLOC_DEV( dtau,  magmaDoubleComplex, min_mn );
        
            TESTING_MALLOC_DEV( d_A2,  magmaDoubleComplex, ldda*N );
            TESTING_MALLOC_DEV( d_T2,  magmaDoubleComplex, N*N    );
            TESTING_MALLOC_DEV( ddA2,  magmaDoubleComplex, N*N    );
            TESTING_MALLOC_DEV( dtau2, magmaDoubleComplex, min_mn );
        
            TESTING_MALLOC_DEV( dwork,  double, max(5*min_mn, (BLOCK_SIZE*2+2)*min_mn) );
            TESTING_MALLOC_DEV( dwork2, double, max(5*min_mn, (BLOCK_SIZE*2+2)*min_mn) );
            
            // todo replace with magma_zlaset
            magmablas_zlaset( MagmaFull, N, N, c_zero, c_zero, ddA,  N );
            magmablas_zlaset( MagmaFull, N, N, c_zero, c_zero, d_T,  N );
            magmablas_zlaset( MagmaFull, N, N, c_zero, c_zero, ddA2, N );
            magmablas_zlaset( MagmaFull, N, N, c_zero, c_zero, d_T2, N );
        
            lwork = -1;
            lapackf77_zgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            lwork = max(lwork, N*N);
        
            TESTING_MALLOC_CPU( h_work, magmaDoubleComplex, lwork );

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( M, N, h_R, lda,  d_A, ldda );
            magma_zsetmatrix( M, N, h_R, lda, d_A2, ldda );
    
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime(0);
    
            if (opts.version == 1)
                magma_zgeqr2x_gpu(M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info);
            else if (opts.version == 2)
                magma_zgeqr2x2_gpu(M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info);
            else if (opts.version == 3)
                magma_zgeqr2x3_gpu(M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info);
            else {
                printf( "call magma_zgeqr2x4_gpu\n" );
                /*
                  Going through NULL stream is faster
                  Going through any stream is slower
                  Doing two streams in parallel is slower than doing them sequentially
                  Queuing happens on the NULL stream - user defined buffers are smaller?
                */
                magma_zgeqr2x4_gpu(M, N, d_A, ldda, dtau, d_T, ddA, dwork, NULL, &info);
                //magma_zgeqr2x4_gpu(M, N, d_A, ldda, dtau, d_T, ddA, dwork, &info, stream[1]);
                //magma_zgeqr2x4_gpu(M, N, d_A2, ldda, dtau2, d_T2, ddA2, dwork2, &info, stream[0]);
                //magma_zgeqr2x4_gpu(M, N, d_A2, ldda, dtau2, d_T2, ddA2, dwork2, &info, NULL);
                //gflops *= 2;
            }
            gpu_time = magma_sync_wtime(0) - gpu_time;
            gpu_perf = gflops / gpu_time;

            if (info != 0) {
                printf("magma_zgeqr2x_gpu version %d returned error %d: %s.\n",
                       (int) opts.version, (int) info, magma_strerror( info ));
            } 
            else {
                if ( opts.check ) {
                    /* =====================================================================
                       Check the result, following zqrt01 except using the reduced Q.
                       This works for any M,N (square, tall, wide).
                       =================================================================== */
                    magma_zgetmatrix( M, N, d_A, ldda, h_R, M );
                    magma_zgetmatrix( N, N, ddA, N,    h_T, N );
                    magma_zgetmatrix( min_mn, 1, dtau, min_mn,   tau, min_mn );

                    // Restore the upper triangular part of A before the check
                    for(int col=0; col < N; col++){
                        for(int row=0; row <= col; row++)
                            h_R[row + col*M] = h_T[row + col*N];
                    }

                    magma_int_t ldq = M;
                    magma_int_t ldr = min_mn;
                    magmaDoubleComplex *Q, *R;
                    double *work;
                    TESTING_MALLOC_CPU( Q,    magmaDoubleComplex, ldq*min_mn );  // M by K
                    TESTING_MALLOC_CPU( R,    magmaDoubleComplex, ldr*N );       // K by N
                    TESTING_MALLOC_CPU( work, double,             min_mn );
                    
                    // generate M by K matrix Q, where K = min(M,N)
                    lapackf77_zlacpy( "Lower", &M, &min_mn, h_R, &M, Q, &ldq );
                    lapackf77_zungqr( &M, &min_mn, &min_mn, Q, &ldq, tau, h_work, &lwork, &info );
                    assert( info == 0 );

                    // copy K by N matrix R
                    lapackf77_zlaset( "Lower", &min_mn, &N, &c_zero, &c_zero, R, &ldr );
                    lapackf77_zlacpy( "Upper", &min_mn, &N, h_R, &M,        R, &ldr );

                    // error = || R - Q^H*A || / (N * ||A||)
                    blasf77_zgemm( "Conj", "NoTrans", &min_mn, &N, &M,
                                   &c_neg_one, Q, &ldq, h_A, &lda, &c_one, R, &ldr );
                    double Anorm = lapackf77_zlange( "1", &M,      &N, h_A, &lda, work );
                    error2 = lapackf77_zlange( "1", &min_mn, &N, R,   &ldr, work );
                    if ( N > 0 && Anorm > 0 )
                        error2 /= (N*Anorm);

                    TESTING_FREE_CPU( Q    );  Q    = NULL;
                    TESTING_FREE_CPU( R    );  R    = NULL;
                    TESTING_FREE_CPU( work );  work = NULL;

                    /* =====================================================================
                       Performs operation using LAPACK
                       =================================================================== */
                    cpu_time = magma_wtime();
                    //lapackf77_zgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                    lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_R, &M, h_A, &lda );
                    lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                                      &M, &N, h_A, &lda, tau, h_work, &N);
                    //magma_zgeqr2(&M, &N, h_A, &lda, tau, h_work, &info);
                                              
                    cpu_time = magma_wtime() - cpu_time;
                    cpu_perf = gflops / cpu_time;
                    if (info != 0)
                        printf("lapackf77_zgeqrf returned error %d: %s.\n",
                               (int) info, magma_strerror( info ));


                    /* =====================================================================
                       Check the result compared to LAPACK
                       =================================================================== */

                    // Restore the upper triangular part of A before the check
                    for(int col=0; col < N; col++){
                        for(int row=0; row <= col; row++)
                            h_R[row + col*M] = h_T[row + col*N];
                    }
                
                    error = lapackf77_zlange("M", &M, &N, h_A, &lda, work);
                    blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                    error = lapackf77_zlange("M", &M, &N, h_R, &lda, work) / (N * error);
     
                    // Check if T is the same
                    magma_zgetmatrix( N, N, d_T, N, h_T, N );
    
                    double terr = 0.;
                    for(int col=0; col < N; col++)
                        for(int row=0; row <= col; row++)
                            terr += (  MAGMA_Z_ABS(h_work[row + col*N] - h_T[row + col*N])*
                                       MAGMA_Z_ABS(h_work[row + col*N] - h_T[row + col*N])  );
                    terr = sqrt( terr );
    
                    // If comparison to LAPACK fail, check || R - Q^H*A || / (N * ||A||)
                    // and print fail if both fails, otherwise print ok (*) 
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)     %8.2e     %8.2e   %s\n",
                           (int) M, (int) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                           error2, terr, (error2 < tol ? "ok" : "failed" )); 

                    status += ! (error2 < tol);
                }
                else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                           (int) M, (int) N, gpu_perf, 1000.*gpu_time);
                }
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
        
            TESTING_FREE_DEV( d_A2   );
            TESTING_FREE_DEV( d_T2   );
            TESTING_FREE_DEV( ddA2   );
            TESTING_FREE_DEV( dtau2  );
            TESTING_FREE_DEV( dwork2 );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );

    TESTING_FINALIZE();
    return status;
}
