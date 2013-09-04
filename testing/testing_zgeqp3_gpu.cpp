/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

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

#define PRECISION_z

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeqp3
*/
int main( int argc, char** argv) 
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    magma_int_t      checkres;
    magmaDoubleComplex *h_A, *h_R, *tau, *h_work;
    magma_int_t *jpvt;

    /* Matrix size */
    magma_int_t M = 0, N = 0, n2, lda, lwork;
    const int MAXTESTS = 10;
    magma_int_t msize[MAXTESTS] = { 1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112 };
    magma_int_t nsize[MAXTESTS] = { 1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112 };

    magma_int_t i, j, info, min_mn, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    checkres = getenv("MAGMA_TESTINGS_CHECK") != NULL;
    
    printf( "\nUsage: %s -N <m,n> -c\n", argv[0] );
    printf( "  -N can be repeated up to %d times. If only m is given, then m=n.\n", MAXTESTS );
    printf( "  -c or setting $MAGMA_TESTINGS_CHECK runs LAPACK and checks result.\n\n" );
    int ntest = 0;
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
            magma_assert( ntest < MAXTESTS, "error: -N repeated more than maximum %d tests\n", MAXTESTS );
            int m, n;
            info = sscanf( argv[++i], "%d,%d", &m, &n );
            if ( info == 2 && m > 0 && n > 0 ) {
                msize[ ntest ] = m;
                nsize[ ntest ] = n;
            }
            else if ( info == 1 && m > 0 ) {
                msize[ ntest ] = m;
                nsize[ ntest ] = m;  // implicitly
            }
            else {
                printf( "error: -N %s is invalid; ensure m > 0, n > 0.\n", argv[i] );
                exit(1);
            }
            M = max( M, msize[ ntest ] );
            N = max( N, nsize[ ntest ] );
            ntest++;
        }
        else if ( strcmp("-M", argv[i]) == 0 ) {
            printf( "-M has been replaced in favor of -N m,n to allow -N to be repeated.\n\n" );
            exit(1);
        }
        else if ( strcmp("-c", argv[i]) == 0 ) {
            checkres = true;
        }
        else {
            printf( "invalid argument: %s\n", argv[i] );
            exit(1);
        }
    }
    if ( ntest == 0 ) {
        ntest = MAXTESTS;
        M = msize[ntest-1];
        N = nsize[ntest-1];
    }

    n2  = M * N;
    min_mn = min(M, N);
    nb = magma_get_zgeqp3_nb(min_mn);

#if defined(PRECISION_z) || defined(PRECISION_c)
    double *drwork, *rwork;
    TESTING_DEVALLOC(   drwork, double, 2*N        +( N+1 )*nb);
    TESTING_MALLOC( rwork, double, 2*N );
#endif
    TESTING_MALLOC(    jpvt, magma_int_t,     N );
    TESTING_MALLOC(    tau,  magmaDoubleComplex, min_mn);
    TESTING_MALLOC(    h_A,  magmaDoubleComplex, n2 );
    TESTING_HOSTALLOC( h_R,  magmaDoubleComplex, n2 );

    lwork = ( N+1 )*nb;
#if defined(PRECISION_d) || defined(PRECISION_s)
    lwork += 2*N;
#endif
    if ( checkres )
        lwork = max(lwork, M * N + N);
    TESTING_HOSTALLOC(h_work, magmaDoubleComplex, lwork ); 

    printf("  M     N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||A*P - Q*R||_F\n");
    printf("=====================================================================\n");
    for( i = 0; i < ntest; ++i ) {
        M = msize[i];
        N = nsize[i];
        min_mn= min(M, N);
        lda   = M;
        n2    = lda*N;
        gflops = FLOPS_ZGEQRF( M, N ) / 1e9;

        /* Initialize the matrix */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );

        /* =====================================================================
           Performs operation using LAPACK
           =================================================================== */
        if ( checkres ) {
            for (j = 0; j < N; j++)
                jpvt[j] = 0;
    
            cpu_time = magma_wtime();
#if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zgeqp3(&M, &N, h_R, &lda, jpvt, tau, h_work, &lwork, rwork, &info);
#else
            lapackf77_zgeqp3(&M, &N, h_R, &lda, jpvt, tau, h_work, &lwork, &info);
#endif
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0)
                printf("lapack_zgeqp3 returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
        }
        
        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
        for (j = 0; j < N; j++) 
            jpvt[j] = 0 ;

        {
            magmaDoubleComplex *d_R, *dtau, *d_work;

            /* allocate gpu workspaces */
            TESTING_DEVALLOC( d_R,    magmaDoubleComplex, lda*N );
            TESTING_DEVALLOC( dtau,   magmaDoubleComplex, min_mn);
            TESTING_DEVALLOC( d_work, magmaDoubleComplex, lwork );

            /* copy A to gpu */
            magma_zsetmatrix( M, N, h_R, lda, d_R, lda );

            /* call gpu-interface */
            gpu_time = magma_wtime();
            #if defined(PRECISION_z) || defined(PRECISION_c)
            magma_zgeqp3_gpu(M, N, d_R, lda, jpvt, dtau, d_work, lwork, drwork, &info);
            #else
            magma_zgeqp3_gpu(M, N, d_R, lda, jpvt, dtau, d_work, lwork, &info);
            #endif
            cudaDeviceSynchronize();
            gpu_time = magma_wtime() - gpu_time;

            /* copy outputs to cpu */
            magma_zgetmatrix( M, N, d_R, lda, h_R, lda );
            magma_zgetvector( min_mn, dtau, 1, tau, 1 );

            /* cleanup */
            TESTING_DEVFREE( d_work );
            TESTING_DEVFREE( dtau );
            TESTING_DEVFREE( d_R );
        }

        gpu_perf = gflops / gpu_time;
        if (info != 0)
            printf("magma_zgeqp3 returned error %d: %s.\n",
                   (int) info, magma_strerror( info ));

        /* =====================================================================
           Check the result 
           =================================================================== */
        if ( checkres ) {
            double error, ulp;
    
            magma_int_t minmn = min(M, N);
            ulp = lapackf77_dlamch( "P" );
            
            // Compute norm( A*P - Q*R )
            error = lapackf77_zqpt01( &M, &N, &minmn, h_A, h_R, &lda, 
                                      tau, jpvt, h_work, &lwork );
            error *= ulp;
    
            printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e\n",
                   (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time, error);
        }
        else {
            printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                   (int) M, (int) N, gpu_perf, gpu_time);
        }
    }

    /* Memory clean up */
#if defined(PRECISION_z) || defined(PRECISION_c)
    TESTING_FREE( rwork );
    TESTING_DEVFREE( drwork );
#endif
    TESTING_FREE( jpvt );
    TESTING_FREE( tau );
    TESTING_FREE( h_A );
    TESTING_HOSTFREE( h_R );
    TESTING_HOSTFREE( h_work );

    /* Shutdown */
    TESTING_FINALIZE();
    return 0;
}
