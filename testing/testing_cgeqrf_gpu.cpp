/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:57 2013
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
   -- Testing cgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    float           error, work[1];
    magmaFloatComplex  c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex *h_A, *d_A, *h_R, *tau, *dT, *h_work, tmp[1];
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn, nb, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1}, ISEED2[4];
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
 
    magma_int_t status = 0;
    float tol;
    opts.lapack |= (opts.version == 2 && opts.check == 2);  // check (-c2) implies lapack (-l)

    if ( opts.version != 2 && opts.check == 1 ) {
        printf( "  ===================================================================\n"
                "  NOTE: -c check for this version will be wrong\n"
                "  because tester ignores the special structure of MAGMA cgeqrf resuls.\n"
                "  We reset it to -c2.\n" 
                "  ===================================================================\n\n");
        opts.check = 2;
    }
    if ( opts.version == 2 ) {
        if ( opts.check == 1 ) {
            printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R-Q'A||_1 / (M*||A||_1*eps) ||I-Q'Q||_1 / (M*eps)\n");
            printf("=========================================================================================================\n");
        } else {
            printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
            printf("=======================================================================\n");
        }
        tol = 1.0;
    } else {
        printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||Ax-b||_F/(N*||A||_F*||x||_F)\n");
        printf("====================================================================================\n");
        tol = opts.tolerance * lapackf77_slamch("E");
    }
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_CGEQRF( M, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_cgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_C_REAL( tmp[0] );
            
            TESTING_MALLOC_CPU( tau,    magmaFloatComplex, min_mn );
            TESTING_MALLOC_CPU( h_A,    magmaFloatComplex, n2     );
            TESTING_MALLOC_CPU( h_work, magmaFloatComplex, lwork  );
            
            TESTING_MALLOC_PIN( h_R,    magmaFloatComplex, n2     );
            
            TESTING_MALLOC_DEV( d_A,    magmaFloatComplex, ldda*N );
            
            /* Initialize the matrix */
            for ( int j=0; j<4; j++ ) ISEED2[j] = ISEED[j]; // saving seeds
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            lapackf77_clacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_csetmatrix( M, N, h_R, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.version == 2 ) {
                magma_cgeqrf2_gpu( M, N, d_A, ldda, tau, &info);
            }
            else {
                nb = magma_get_cgeqrf_nb( M );
                size = (2*min(M, N) + (N+31)/32*32 )*nb;
                TESTING_MALLOC_DEV( dT, magmaFloatComplex, size );
                if ( opts.version == 3 ) {
                    magma_cgeqrf3_gpu( M, N, d_A, ldda, tau, dT, &info);
                }
                else {
                    magma_cgeqrf_gpu( M, N, d_A, ldda, tau, dT, &info);
                }
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_cgeqrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                magmaFloatComplex *tau2;
                TESTING_MALLOC_CPU( tau2, magmaFloatComplex, min_mn );
                cpu_time = magma_wtime();
                lapackf77_cgeqrf(&M, &N, h_A, &lda, tau2, h_work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_cgeqrf returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                TESTING_FREE_CPU( tau2 );
            }

            if ( opts.check == 1 ) {
                /* =====================================================================
                   Check the result 
                   =================================================================== */
                magma_int_t lwork = n2+N;
                magmaFloatComplex *h_W1, *h_W2, *h_W3;
                float *h_RW, results[2];
                magma_cgetmatrix( M, N, d_A, ldda, h_R, M );

                TESTING_MALLOC_CPU( h_W1, magmaFloatComplex, n2    ); // Q
                TESTING_MALLOC_CPU( h_W2, magmaFloatComplex, n2    ); // R
                TESTING_MALLOC_CPU( h_W3, magmaFloatComplex, lwork ); // WORK
                TESTING_MALLOC_CPU( h_RW, float, M );  // RWORK
                lapackf77_clarnv( &ione, ISEED2, &n2, h_A );
                lapackf77_cqrt02( &M, &N, &min_mn, h_A, h_R, h_W1, h_W2, &lda, tau, h_W3, &lwork,
                                  h_RW, results );

                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e                      %8.2e",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, results[0],results[1] );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)    %8.2e                      %8.2e",
                           (int) M, (int) N, gpu_perf, gpu_time, results[0],results[1] );
                } 
                printf("%s\n", (results[0] < tol ? "" : "  failed"));
                status |= ! (results[0] < tol);
            
                TESTING_FREE_CPU( h_W1 );
                TESTING_FREE_CPU( h_W2 );
                TESTING_FREE_CPU( h_W3 );
                TESTING_FREE_CPU( h_RW );
            }
            else if ( opts.check == 2 ) {
                if ( opts.version == 2 ) {
                    /* =====================================================================
                       Check the result compared to LAPACK
                       =================================================================== */
                    magma_cgetmatrix( M, N, d_A, ldda, h_R, M );
                    error = lapackf77_clange("f", &M, &N, h_A, &lda, work);
                    blasf77_caxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                    error = lapackf77_clange("f", &M, &N, h_R, &lda, work) / error;

                    if ( opts.lapack ) {
                        printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e",
                               (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error );
                    } else {
                        printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)   %8.2e",
                               (int) M, (int) N, gpu_perf, gpu_time, error );
                    }
                    printf("%s\n", (error < tol ? "" : "  failed"));
                    status |= ! (error < tol);
                }
                else if ( M >= N ) {
                    magma_int_t lwork;
                    magmaFloatComplex *x, *b, *d_B, *hwork;
                    const magmaFloatComplex c_zero    = MAGMA_C_ZERO;
                    const magmaFloatComplex c_one     = MAGMA_C_ONE;
                    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
                    const magma_int_t ione = 1;

                    // initialize RHS, b = A*random
                    TESTING_MALLOC_CPU( x, magmaFloatComplex, N );
                    TESTING_MALLOC_CPU( b, magmaFloatComplex, M );
                    lapackf77_clarnv( &ione, ISEED, &N, x );
                    blasf77_cgemv( "Notrans", &M, &N, &c_one, h_A, &lda, x, &ione, &c_zero, b, &ione );
                    // copy to GPU
                    TESTING_MALLOC_DEV( d_B, magmaFloatComplex, M );
                    magma_csetvector( M, b, 1, d_B, 1 );

                    if ( opts.version == 1 ) {
                        // allocate hwork
                        magma_cgeqrs_gpu( M, N, 1,
                                          d_A, ldda, tau, dT,
                                          d_B, M, tmp, -1, &info );
                        lwork = (magma_int_t)MAGMA_C_REAL( tmp[0] );
                        TESTING_MALLOC_CPU( hwork, magmaFloatComplex, lwork );

                        // solve linear system
                        magma_cgeqrs_gpu( M, N, 1,
                                          d_A, ldda, tau, dT,
                                          d_B, M, hwork, lwork, &info );
                       if (info != 0)
                           printf("magma_cgeqrs returned error %d: %s.\n",
                                  (int) info, magma_strerror( info ));
                        TESTING_FREE_CPU( hwork );
                    }
                    else {
                        // allocate hwork
                        magma_cgeqrs3_gpu( M, N, 1,
                                           d_A, ldda, tau, dT,
                                           d_B, M, tmp, -1, &info );
                        lwork = (magma_int_t)MAGMA_C_REAL( tmp[0] );
                        TESTING_MALLOC_CPU( hwork, magmaFloatComplex, lwork );

                        // solve linear system
                        magma_cgeqrs3_gpu( M, N, 1,
                                           d_A, ldda, tau, dT,
                                           d_B, M, hwork, lwork, &info );
                       if (info != 0)
                           printf("magma_cgeqrs3 returned error %d: %s.\n",
                                  (int) info, magma_strerror( info ));
                        TESTING_FREE_CPU( hwork );
                    }
                    magma_cgetvector( N, d_B, 1, x, 1 );

                    // compute r = Ax - b, saved in b
                    lapackf77_clarnv( &ione, ISEED2, &n2, h_A );
                    blasf77_cgemv( "Notrans", &M, &N, &c_one, h_A, &lda, x, &ione, &c_neg_one, b, &ione );

                    // compute residual |Ax - b| / (n*|A|*|x|)
                    float norm_x, norm_A, norm_r, work[1];
                    norm_A = lapackf77_clange( "F", &M, &N, h_A, &lda, work );
                    norm_r = lapackf77_clange( "F", &M, &ione, b, &M, work );
                    norm_x = lapackf77_clange( "F", &N, &ione, x, &N, work );

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
                    printf("%s\n", (error < tol ? "" : "  failed"));
                    status |= ! (error < tol);
                }
                else {
                    if ( opts.lapack ) {
                        printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   --- ",
                               (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
                    } else {
                        printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     --- ",
                               (int) M, (int) N, gpu_perf, gpu_time);
                    }
                    printf("%s ", (opts.check != 0 ? "  (error check only for M>=N)" : ""));
                }
            }
            else {
                if ( opts.lapack ) {
                    printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   ---\n",
                           (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
                } else {
                    printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                           (int) M, (int) N, gpu_perf, gpu_time);
                }

            }
            
            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_PIN( h_R );
            
            TESTING_FREE_DEV( d_A );
            
            if ( opts.version != 2 )
                TESTING_FREE_DEV( dT );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return status;
}
