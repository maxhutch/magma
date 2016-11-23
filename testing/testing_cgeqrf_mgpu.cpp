/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zgeqrf_mgpu.cpp, normal z -> c, Sun Nov 20 20:20:36 2016
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
   -- Testing cgeqrf_mgpu
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    const float             d_neg_one = MAGMA_D_NEG_ONE;
    const float             d_one     = MAGMA_D_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    const magma_int_t        ione      = 1;
    
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    float           Anorm, error=0, error2=0;
    magmaFloatComplex *h_A, *h_R, *tau, *h_work, tmp[1], unused[1];
    magmaFloatComplex_ptr d_lA[ MagmaMaxGPUs ];
    magma_int_t M, N, n2, lda, lwork, info, min_mn, nb;
    magma_int_t ldda, n_local, ngpu;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code
 
    int status = 0;
    float tol = opts.tolerance * lapackf77_slamch("E");

    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev = 0; dev < opts.ngpu; ++dev ) {
        magma_queue_create( dev, &queues[dev] );
    }
    
    printf("%% ngpu %lld\n", (long long) opts.ngpu );
    printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |R - Q^H*A|   |I - Q^H*Q|\n");
    printf("%%==============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            nb     = magma_get_cgeqrf_nb( M, N );
            gflops = FLOPS_CGEQRF( M, N ) / 1e9;
            
            // ngpu must be at least the number of blocks
            ngpu = min( opts.ngpu, magma_ceildiv(N,nb) );
            if ( ngpu < opts.ngpu ) {
                printf( " * too many GPUs for the matrix size, using %lld GPUs\n", (long long) ngpu );
            }
            
            // query for workspace size
            lwork = -1;
            lapackf77_cgeqrf( &M, &N, unused, &M, unused, tmp, &lwork, &info );
            lwork = (magma_int_t) MAGMA_C_REAL( tmp[0] );
            lwork = max( lwork, N*nb );
            
            // Allocate host memory for the matrix
            TESTING_CHECK( magma_cmalloc_cpu( &tau,    min_mn ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_A,    n2     ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_work, lwork  ));
            
            TESTING_CHECK( magma_cmalloc_pinned( &h_R,    n2     ));
            
            // Allocate device memory
            for( int dev = 0; dev < ngpu; dev++ ) {
                n_local = ((N/nb)/ngpu)*nb;
                if (dev < (N/nb) % ngpu)
                    n_local += nb;
                else if (dev == (N/nb) % ngpu)
                    n_local += N % nb;
                magma_setdevice( dev );
                TESTING_CHECK( magma_cmalloc( &d_lA[dev], ldda*n_local ));
            }
            
            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            lapackf77_clacpy( MagmaFullStr, &M, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_csetmatrix_1D_col_bcyclic( ngpu, M, N, nb, h_R, lda, d_lA, ldda, queues );

            gpu_time = magma_wtime();
            magma_cgeqrf2_mgpu( ngpu, M, N, d_lA, ldda, tau, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_cgeqrf2_mgpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            magma_cgetmatrix_1D_col_bcyclic( ngpu, M, N, nb, d_lA, ldda, h_R, lda, queues );
            
            /* =====================================================================
               Check the result, following zqrt01 except using the reduced Q.
               This works for any M,N (square, tall, wide).
               =================================================================== */
            if ( opts.check ) {
                magma_int_t ldq = M;
                magma_int_t ldr = min_mn;
                magmaFloatComplex *Q, *R;
                float *work;
                TESTING_CHECK( magma_cmalloc_cpu( &Q,    ldq*min_mn ));  // M by K
                TESTING_CHECK( magma_cmalloc_cpu( &R,    ldr*N ));       // K by N
                TESTING_CHECK( magma_smalloc_cpu( &work, min_mn ));
            
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
                
                magma_free_cpu( Q    );  Q    = NULL;
                magma_free_cpu( R    );  R    = NULL;
                magma_free_cpu( work );  work = NULL;
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_cgeqrf( &M, &N, h_A, &lda, tau, h_work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_cgeqrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            printf("%5lld %5lld   ", (long long) M, (long long) N );
            if ( opts.lapack ) {
                printf( "%7.2f (%7.2f)", cpu_perf, cpu_time );
            }
            else {
                printf("  ---   (  ---  )" );
            }
            printf( "   %7.2f (%7.2f)   ", gpu_perf, gpu_time );
            if ( opts.check ) {
                bool okay = (error < tol && error2 < tol);
                status += ! okay;
                printf( "%11.2e   %11.2e   %s\n", error, error2, (okay ? "ok" : "failed") );
            }
            else {
                printf( "    ---\n" );
            }
            
            magma_free_cpu( tau    );
            magma_free_cpu( h_A    );
            magma_free_cpu( h_work );
            
            magma_free_pinned( h_R    );
            
            for( int dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                magma_free( d_lA[dev] );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    for( int dev = 0; dev < opts.ngpu; ++dev ) {
        magma_queue_destroy( queues[dev] );
    }
 
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
