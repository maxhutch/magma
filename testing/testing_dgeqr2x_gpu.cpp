/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:56 2013

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
#include "common_magma.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgeqrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];

    double  c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_T, *h_R, *tau, *h_work, tmp[1];
    double *d_A,  *d_T, *ddA, *dtau;
    double *d_A2, *d_T2, *ddA2, *dtau2;
    double *dwork, *dwork2;

    /* Matrix size */
    magma_int_t M = 0, N = 0, n2, lda, ldda, lwork;
    const int MAXTESTS = 10;
    magma_int_t msize[MAXTESTS] = { 1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112 };
    magma_int_t nsize[MAXTESTS] = { 1024, 2048, 3072, 4032, 5184, 6016, 7040, 8064, 9088, 10112 };

    magma_int_t i, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t checkres, version = 3;

    checkres = getenv("MAGMA_TESTINGS_CHECK") != NULL;

    // process command line arguments
    printf( "\nUsage: %s -N <m,n> -c -v <version 1..4>\n", argv[0] );
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
        else if ( strcmp("-v", argv[i]) == 0 ) {
            version = atoi( argv[++i] );
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

    ldda   = ((M+31)/32)*32;
    n2     = M * N;
    min_mn = min(M, N);

    /* Allocate memory for the matrix */
    TESTING_MALLOC_CPU( tau,   double, min_mn );
    TESTING_MALLOC_CPU( h_A,   double, n2     );
    TESTING_MALLOC_CPU( h_T,   double, N*N    );

    TESTING_MALLOC_PIN( h_R,   double, n2     );

    TESTING_MALLOC_DEV( d_A,   double, ldda*N );
    TESTING_MALLOC_DEV( d_T,   double, N*N    );
    TESTING_MALLOC_DEV( ddA,   double, N*N    );
    TESTING_MALLOC_DEV( dtau,  double, min_mn );

    TESTING_MALLOC_DEV( d_A2,  double, ldda*N );
    TESTING_MALLOC_DEV( d_T2,  double, N*N    );
    TESTING_MALLOC_DEV( ddA2,  double, N*N    );
    TESTING_MALLOC_DEV( dtau2, double, min_mn );

#define BLOCK_SIZE 64
    TESTING_MALLOC_DEV( dwork,  double, max(5*min_mn, (BLOCK_SIZE*2+2)*min_mn) );
    TESTING_MALLOC_DEV( dwork2, double, max(5*min_mn, (BLOCK_SIZE*2+2)*min_mn) );

    cudaMemset(ddA, 0, N*N*sizeof(double));
    cudaMemset(d_T, 0, N*N*sizeof(double));

    cudaMemset(ddA2, 0, N*N*sizeof(double));
    cudaMemset(d_T2, 0, N*N*sizeof(double));

    lwork = -1;
    lapackf77_dgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
    lwork = (magma_int_t)MAGMA_D_REAL( tmp[0] );
    lwork = max(lwork, N*N);

    TESTING_MALLOC_CPU( h_work, double, lwork );

    cudaStream_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );


    printf("  M     N     CPU GFlop/s (ms)    GPU GFlop/s (ms)   ||R||_F/||A||_F  ||R_T||\n");
    printf("=============================================================================\n");
    for( i = 0; i < ntest; ++i ) {
        M = msize[i];
        N = nsize[i];
        min_mn= min(M, N);
        lda   = M;
        n2    = lda*N;
        ldda  = ((M+31)/32)*32;
        gflops = (FLOPS_DGEQRF( M, N ) + FLOPS_DGEQRT( M, N)) / 1e9;

        /* Initialize the matrix */
        lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
        lapackf77_dlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
        magma_dsetmatrix( M, N, h_R, lda,  d_A, ldda );
        magma_dsetmatrix( M, N, h_R, lda, d_A2, ldda );

        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */

        cudaDeviceSynchronize();
        gpu_time = magma_wtime();

        if (version == 1)
            magma_dgeqr2x_gpu(&M, &N, d_A, &ldda, dtau, d_T, ddA, dwork, &info);
        else if (version == 2)
            magma_dgeqr2x2_gpu(&M, &N, d_A, &ldda, dtau, d_T, ddA, dwork, &info);
        else if (version == 3)
            magma_dgeqr2x3_gpu(&M, &N, d_A, &ldda, dtau, d_T, ddA, dwork, &info);
        else {
          /*
            Going through NULL stream is faster
            Going through any stream is slower
            Doing two streams in parallel is slower than doing them sequentially
            Queuing happens on the NULL stream - user defined buffers are smaller?
          */
          //magma_dgeqr2x4_gpu(&M, &N,  d_A, &ldda, dtau, d_T, ddA, dwork, &info, NULL);
          magma_dgeqr2x4_gpu(&M, &N,  d_A, &ldda, dtau, d_T, ddA, dwork, &info, stream[1]);
          magma_dgeqr2x4_gpu(&M, &N, d_A2, &ldda,dtau2,d_T2,ddA2,dwork2, &info, stream[0]);
          //magma_dgeqr2x4_gpu(&M, &N, d_A2, &ldda,dtau2,d_T2,ddA2,dwork2, &info, NULL);
          //gflops *= 2;
        }

        cudaDeviceSynchronize();
        gpu_time = magma_wtime() - gpu_time;
        gpu_perf = gflops / gpu_time;
        if (info != 0)
            printf("magma_dgeqrf returned error %d: %s.\n",
                   (int) info, magma_strerror( info ));
        
        if ( checkres ) {
          /*
            int tm=1000,tn=1000,tsiz=tm*tn;
            double *myA, *mytau, *mywork;

            TESTING_MALLOC_CPU( myA,    double, tsiz );
            TESTING_MALLOC_CPU( mywork, double, tsiz );
            TESTING_MALLOC_CPU( mytau,  double, tn   );
            lapackf77_dlarnv( &ione, ISEED, &tsiz, myA );
            lapackf77_dgeqrf(&tm, &tn, myA, &tm, mytau, mywork, &tsiz, &info);
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &tm, &tn, myA, &tm, mytau, mywork, &tn);
          */

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_dgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &M, &N, h_A, &lda, tau, h_work, &N);
            //magma_dgeqr2(&M, &N, h_A, &lda, tau, h_work, &info);
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0)
                printf("lapackf77_dgeqrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Check the result compared to LAPACK
               =================================================================== */
            magma_dgetmatrix( M, N, d_A, ldda, h_R, M );
            magma_dgetmatrix( N, N, ddA, N,    h_T, N );

            // Restore the upper triangular part of A before the check
            for(int col=0; col<N; col++){
                for(int row=0; row<=col; row++)
                    h_R[row + col*M] = h_T[row + col*N];
            }
            
            error = lapackf77_dlange("M", &M, &N, h_A, &lda, work);
            blasf77_daxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
            error = lapackf77_dlange("M", &M, &N, h_R, &lda, work) / error;

            // Check if T is the same
            double terr = 0.;
            magma_dgetmatrix( N, N, d_T, N, h_T, N );

            for(int col=0; col<N; col++)
                for(int row=0; row<=col; row++)
                    terr += (  MAGMA_D_ABS(h_work[row + col*N] - h_T[row + col*N])*
                               MAGMA_D_ABS(h_work[row + col*N] - h_T[row + col*N])  );
            terr = magma_dsqrt(terr);

            printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)     %8.2e     %8.2e\n",
                   (int) M, (int) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                   error, terr);
        }
        else {
            printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                   (int) M, (int) N, gpu_perf, 1000.*gpu_time);
        }
    }
    
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );

    /* Memory clean up */
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

    TESTING_FINALIZE();

    return 0;
}
