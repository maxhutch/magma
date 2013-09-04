/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @generated d Wed Aug 14 12:17:59 2013
       @author Mark Gates

*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgeadd_batched
   Code is very similar to testing_dlacpy_batched.cpp
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    double  c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_B;
    double *d_A, *d_B;
    double **hAarray, **hBarray, **dAarray, **dBarray;
    double alpha = MAGMA_D_MAKE( 3.1415, 2.718 );
    magma_int_t M, N, mb, nb, size, lda, ldda, mstride, nstride, ntile;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    mb = (opts.nb == 0 ? 32 : opts.nb);
    nb = (opts.nb == 0 ? 64 : opts.nb);
    mstride = 2*mb;
    nstride = 3*nb;
    
    printf("mb=%d, nb=%d, mstride=%d, nstride=%d\n", (int) mb, (int) nb, (int) mstride, (int) nstride );
    printf("    M     N ntile   CPU GFlop/s (sec)   GPU GFlop/s (sec)   error   \n");
    printf("====================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[i];
            N = opts.nsize[i];
            lda    = M;
            ldda   = ((M+31)/32)*32;
            size   = lda*N;
            
            if ( N < nb || M < nb ) {
                ntile = 0;
            } else {
                ntile = min( (M - nb)/mstride + 1,
                             (N - nb)/nstride + 1 );
            }
            gflops = 2.*mb*nb*ntile / 1e9;
            
            TESTING_MALLOC(   h_A, double, lda *N );
            TESTING_MALLOC(   h_B, double, lda *N );
            TESTING_DEVALLOC( d_A, double, ldda*N );
            TESTING_DEVALLOC( d_B, double, ldda*N );
            
            TESTING_MALLOC(   hAarray, double*, ntile );
            TESTING_MALLOC(   hBarray, double*, ntile );
            TESTING_DEVALLOC( dAarray, double*, ntile );
            TESTING_DEVALLOC( dBarray, double*, ntile );
            
            lapackf77_dlarnv( &ione, ISEED, &size, h_A );
            lapackf77_dlarnv( &ione, ISEED, &size, h_B );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_dsetmatrix( M, N, h_A, lda, d_A, ldda );
            magma_dsetmatrix( M, N, h_B, lda, d_B, ldda );
            
            // setup pointers
            for( int tile = 0; tile < ntile; ++tile ) {
                int offset = tile*mstride + tile*nstride*ldda;
                hAarray[tile] = &d_A[offset];
                hBarray[tile] = &d_B[offset];
            }
            magma_setvector( ntile, sizeof(double*), hAarray, 1, dAarray, 1 );
            magma_setvector( ntile, sizeof(double*), hBarray, 1, dBarray, 1 );
            
            gpu_time = magma_sync_wtime( 0 );
            magmablas_dgeadd_batched( mb, nb, alpha, dAarray, ldda, dBarray, ldda, ntile );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            for( int tile = 0; tile < ntile; ++tile ) {
                int offset = tile*mstride + tile*nstride*lda;
                for( int j = 0; j < nb; ++j ) {
                    blasf77_daxpy( &mb, &alpha,
                                   &h_A[offset + j*lda], &ione,
                                   &h_B[offset + j*lda], &ione );
                }
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_dgetmatrix( M, N, d_B, ldda, h_A, lda );
            
            error = lapackf77_dlange( "F", &M, &N, h_B, &lda, work );
            blasf77_daxpy(&size, &c_neg_one, h_A, &ione, h_B, &ione);
            error = lapackf77_dlange("f", &M, &N, h_B, &lda, work) / error;

            printf("%5d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e\n",
                   (int) M, (int) N, (int) ntile,
                   cpu_perf, cpu_time, gpu_perf, gpu_time, error );
            
            TESTING_FREE( h_A );
            TESTING_FREE( h_B );
            TESTING_DEVFREE( d_A );
            TESTING_DEVFREE( d_B );
            
            TESTING_FREE( hAarray );
            TESTING_FREE( hBarray );
            TESTING_DEVFREE( dAarray );
            TESTING_DEVFREE( dBarray );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
