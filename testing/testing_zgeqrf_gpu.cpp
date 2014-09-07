/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s
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
    magmaDoubleComplex *h_A, *d_A, *h_R, *tau, *dT, *h_work, tmp[1];
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn, nb, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1}, ISEED2[4];
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
 
    magma_int_t status = 0;
    double tol;
    opts.lapack |= (opts.version == 2 && opts.check == 2);  // check (-c2) implies lapack (-l)

    if ( opts.version != 2 && opts.check == 1 ) {
        printf( "NOTE: version %d requires -c2 check due to the special structure of the\n"
                "MAGMA zgeqrf results; using -c2.\n\n", (int) opts.version );
        opts.check = 2;
    }
    printf( "version %d\n", (int) opts.version );
    if ( opts.version == 2 ) {
        if ( opts.check == 1 ) {
            printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R-Q'A||_1 / (M*||A||_1*eps) ||I-Q'Q||_1 / (M*eps)\n");
            printf("=========================================================================================================\n");
        } else {
            printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
            printf("=======================================================================\n");
        }
        tol = 1.0;
    } else {
        printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||Ax-b||_F/(N*||A||_F*||x||_F)\n");
        printf("====================================================================================\n");
        tol = opts.tolerance * lapackf77_dlamch("E");
    }
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_ZGEQRF( M, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_zgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            
            TESTING_MALLOC_CPU( tau,    magmaDoubleComplex, min_mn );
            TESTING_MALLOC_CPU( h_A,    magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_work, magmaDoubleComplex, lwork  );
            
            TESTING_MALLOC_PIN( h_R,    magmaDoubleComplex, n2     );
            
            TESTING_MALLOC_DEV( d_A,    magmaDoubleComplex, ldda*N );
            
            /* Initialize the matrix */
            for ( int j=0; j<4; j++ )
                ISEED2[j] = ISEED[j]; // save seeds
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( M, N, h_R, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.version == 2 ) {
                magma_zgeqrf2_gpu( M, N, d_A, ldda, tau, &info);
            }
            else {
                nb = magma_get_zgeqrf_nb( M );
                size = (2*min(M, N) + (N+31)/32*32 )*nb;
                TESTING_MALLOC_DEV( dT, magmaDoubleComplex, size );
                if ( opts.version == 3 ) {
                    magma_zgeqrf3_gpu( M, N, d_A, ldda, tau, dT, &info);
                }
                else {
                    magma_zgeqrf_gpu( M, N, d_A, ldda, tau, dT, &info);
                }
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgeqrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                magmaDoubleComplex *tau2;
                TESTING_MALLOC_CPU( tau2, magmaDoubleComplex, min_mn );
                cpu_time = magma_wtime();
                lapackf77_zgeqrf(&M, &N, h_A, &lda, tau2, h_work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                TESTING_FREE_CPU( tau2 );
            }

            if ( opts.check == 1 && M >= N ) {
                /* =====================================================================
                   Check the result -- only version 1, zqrt02 requires M >= N
                   =================================================================== */
                magma_int_t lwork = n2+N;
                magmaDoubleComplex *h_W1, *h_W2, *h_W3;
                double *h_RW, results[2];
                
                magma_zgetmatrix( M, N, d_A, ldda, h_R, M );

                TESTING_MALLOC_CPU( h_W1, magmaDoubleComplex, n2    ); // Q
                TESTING_MALLOC_CPU( h_W2, magmaDoubleComplex, n2    ); // R
                TESTING_MALLOC_CPU( h_W3, magmaDoubleComplex, lwork ); // WORK
                TESTING_MALLOC_CPU( h_RW, double, M );  // RWORK
                lapackf77_zlarnv( &ione, ISEED2, &n2, h_A );
                lapackf77_zqrt02( &M, &N, &min_mn, h_A, h_R, h_W1, h_W2, &lda, tau, h_W3, &lwork,
                                  h_RW, results );

                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e                      %8.2e",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, results[0], results[1] );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)    %8.2e                      %8.2e",
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
            else if ( opts.check == 2 && opts.version == 2 ) {
                /* =====================================================================
                   Check the result compared to LAPACK -- only version 2
                   =================================================================== */
                magma_zgetmatrix( M, N, d_A, ldda, h_R, M );
                error = lapackf77_zlange("f", &M, &N, h_A, &lda, work);
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                error = lapackf77_zlange("f", &M, &N, h_R, &lda, work) / error;

                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)   %8.2e",
                           (int) M, (int) N, gpu_perf, gpu_time, error );
                }
                printf("   %s\n", (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else if ( opts.check == 2 && M >= N ) {
                /* =====================================================================
                   Check the result by solving linear system -- only versions 1 & 3, M >= N
                   =================================================================== */
                magma_int_t lwork;
                magmaDoubleComplex *x, *b, *d_B, *hwork;
                const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
                const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
                const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
                const magma_int_t ione = 1;

                // initialize RHS, b = A*random
                TESTING_MALLOC_CPU( x, magmaDoubleComplex, N );
                TESTING_MALLOC_CPU( b, magmaDoubleComplex, M );
                lapackf77_zlarnv( &ione, ISEED, &N, x );
                blasf77_zgemv( "Notrans", &M, &N, &c_one, h_A, &lda, x, &ione, &c_zero, b, &ione );
                // copy to GPU
                TESTING_MALLOC_DEV( d_B, magmaDoubleComplex, M );
                magma_zsetvector( M, b, 1, d_B, 1 );

                if ( opts.version == 1 ) {
                    // allocate hwork
                    magma_zgeqrs_gpu( M, N, 1,
                                      d_A, ldda, tau, dT,
                                      d_B, M, tmp, -1, &info );
                    lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
                    TESTING_MALLOC_CPU( hwork, magmaDoubleComplex, lwork );

                    // solve linear system
                    magma_zgeqrs_gpu( M, N, 1,
                                      d_A, ldda, tau, dT,
                                      d_B, M, hwork, lwork, &info );
                   if (info != 0)
                       printf("magma_zgeqrs returned error %d: %s.\n",
                              (int) info, magma_strerror( info ));
                    TESTING_FREE_CPU( hwork );
                }
                else {
                    // allocate hwork
                    magma_zgeqrs3_gpu( M, N, 1,
                                       d_A, ldda, tau, dT,
                                       d_B, M, tmp, -1, &info );
                    lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
                    TESTING_MALLOC_CPU( hwork, magmaDoubleComplex, lwork );

                    // solve linear system
                    magma_zgeqrs3_gpu( M, N, 1,
                                       d_A, ldda, tau, dT,
                                       d_B, M, hwork, lwork, &info );
                   if (info != 0)
                       printf("magma_zgeqrs3 returned error %d: %s.\n",
                              (int) info, magma_strerror( info ));
                    TESTING_FREE_CPU( hwork );
                }
                magma_zgetvector( N, d_B, 1, x, 1 );

                // compute r = Ax - b, saved in b
                lapackf77_zlarnv( &ione, ISEED2, &n2, h_A );
                blasf77_zgemv( "Notrans", &M, &N, &c_one, h_A, &lda, x, &ione, &c_neg_one, b, &ione );

                // compute residual |Ax - b| / (n*|A|*|x|)
                double norm_x, norm_A, norm_r, work[1];
                norm_A = lapackf77_zlange( "F", &M, &N, h_A, &lda, work );
                norm_r = lapackf77_zlange( "F", &M, &ione, b, &M, work );
                norm_x = lapackf77_zlange( "F", &N, &ione, x, &N, work );

                TESTING_FREE_CPU( x );
                TESTING_FREE_CPU( b );
                TESTING_FREE_DEV( d_B );

                error = norm_r / (N * norm_A * norm_x);
                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)   %8.2e",
                           (int) M, (int) N, gpu_perf, gpu_time, error );
                }
                printf("   %s\n", (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
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
            
            TESTING_FREE_PIN( h_R );
            
            TESTING_FREE_DEV( d_A );
            
            if ( opts.version != 2 )
                TESTING_FREE_DEV( dT );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return status;
}
