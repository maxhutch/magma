/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zlacpy_batched.cpp normal z -> s, Mon May  2 23:31:22 2016
       @author Mark Gates

*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing slacpy_batched
   Code is very similar to testing_sgeadd_batched.cpp
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float           error, work[1];
    float  c_neg_one = MAGMA_S_NEG_ONE;
    float *h_A, *h_B;
    magmaFloat_ptr d_A, d_B;
    float **hAarray, **hBarray, **dAarray, **dBarray;
    magma_int_t M, N, mb, nb, size, lda, ldda, mstride, nstride, ntile;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );

    mb = (opts.nb == 0 ? 32 : opts.nb);
    nb = (opts.nb == 0 ? 64 : opts.nb);
    mstride = 2*mb;
    nstride = 3*nb;
    
    printf("%% mb=%d, nb=%d, mstride=%d, nstride=%d\n", (int) mb, (int) nb, (int) mstride, (int) nstride );
    printf("%%   M     N ntile    CPU Gflop/s (ms)    GPU Gflop/s (ms)   check\n");
    printf("%%================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            size   = lda*N;
            
            if ( N < nb || M < nb ) {
                ntile = 0;
            } else {
                ntile = min( (M - nb)/mstride + 1,
                             (N - nb)/nstride + 1 );
            }
            gbytes = 2.*mb*nb*ntile / 1e9;
            
            TESTING_MALLOC_CPU( h_A, float, lda *N );
            TESTING_MALLOC_CPU( h_B, float, lda *N );
            TESTING_MALLOC_DEV( d_A, float, ldda*N );
            TESTING_MALLOC_DEV( d_B, float, ldda*N );
            
            TESTING_MALLOC_CPU( hAarray, float*, ntile );
            TESTING_MALLOC_CPU( hBarray, float*, ntile );
            TESTING_MALLOC_DEV( dAarray, float*, ntile );
            TESTING_MALLOC_DEV( dBarray, float*, ntile );
            
            lapackf77_slarnv( &ione, ISEED, &size, h_A );
            lapackf77_slarnv( &ione, ISEED, &size, h_B );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_ssetmatrix( M, N, h_A, lda, d_A, ldda, opts.queue );
            magma_ssetmatrix( M, N, h_B, lda, d_B, ldda, opts.queue );
            
            // setup pointers
            for( magma_int_t tile = 0; tile < ntile; ++tile ) {
                magma_int_t offset = tile*mstride + tile*nstride*ldda;
                hAarray[tile] = &d_A[offset];
                hBarray[tile] = &d_B[offset];
            }
            magma_setvector( ntile, sizeof(float*), hAarray, 1, dAarray, 1, opts.queue );
            magma_setvector( ntile, sizeof(float*), hBarray, 1, dBarray, 1, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_slacpy_batched( MagmaFull, mb, nb, dAarray, ldda, dBarray, ldda, ntile, opts.queue );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            for( magma_int_t tile = 0; tile < ntile; ++tile ) {
                magma_int_t offset = tile*mstride + tile*nstride*lda;
                lapackf77_slacpy( MagmaFullStr, &mb, &nb,
                                  &h_A[offset], &lda,
                                  &h_B[offset], &lda );
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_sgetmatrix( M, N, d_B, ldda, h_A, lda, opts.queue );
            
            blasf77_saxpy(&size, &c_neg_one, h_A, &ione, h_B, &ione);
            error = lapackf77_slange("f", &M, &N, h_B, &lda, work);
            bool okay = (error == 0);
            status += ! okay;

            printf("%5d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   (int) M, (int) N, (int) ntile,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   (okay ? "ok" : "failed") );
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            
            TESTING_FREE_CPU( hAarray );
            TESTING_FREE_CPU( hBarray );
            TESTING_FREE_DEV( dAarray );
            TESTING_FREE_DEV( dBarray );
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
