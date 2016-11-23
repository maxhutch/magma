/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zher2k_mgpu.cpp, normal z -> d, Sun Nov 20 20:20:34 2016
       
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

// define ICHI to test with Ichi's version, too
#undef ICHI


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing magma_dsyr2k_mgpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    double c_neg_one = MAGMA_D_NEG_ONE;
    double alpha = MAGMA_D_MAKE( 1.2345, 4.321 );
    double beta = 3.14159;
    
    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    double           error, work[1];
    double *hC, *hR, *hR2, *hA, *hB;
    magmaDouble_ptr dA[MagmaMaxGPUs], dB[MagmaMaxGPUs], dC[MagmaMaxGPUs];
    magma_int_t n, k, size, lda, ldda, nb, ngpu, nqueue;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_queue_t queues[MagmaMaxGPUs][20], queues0[MagmaMaxGPUs];
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    ngpu    = opts.ngpu;
    nb      = (opts.nb      > 0 ? opts.nb      : 64);
    nqueue  = (opts.nqueue  > 0 ? opts.nqueue  :  2);
    
    printf( "%% version 1: magmablas_dsyr2k_mgpu2     %s\n", (opts.version == 1 ? "(enabled)" : ""));
    //printf( "%% version 2: magmablas_dsyr2k_mgpu_spec %s\n", (opts.version == 2 ? "(enabled)" : ""));
#ifdef ICHI
    printf( "%% version 3: magma_dsyr2k_mgpu (Ichi)   %s\n", (opts.version == 3 ? "(enabled)" : ""));
#endif
    printf( "\n" );
    
    printf("%% nb %lld, ngpu %lld, nqueue %lld\n", (long long) nb, (long long) ngpu, (long long) nqueue );
    printf("%%   n     k    nb offset  CPU Gflop/s (sec)   GPU Gflop/s (sec)   |R|/(|V|*|W|+|A|)\n");
    printf("%%==================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        n = opts.nsize[itest];
        k = opts.ksize[itest];
        
        for( int offset = 0; offset < n; offset += min(k,nb) ) {
            for( int iter = 0; iter < opts.niter; ++iter ) {
                lda    = n;
                ldda   = magma_roundup( n, opts.align );  // multiple of 32 by default
                gflops = FLOPS_DSYR2K( k, n-offset ) / 1e9;
                
                TESTING_CHECK( magma_dmalloc_cpu( &hC,  lda*n ));
                TESTING_CHECK( magma_dmalloc_cpu( &hR,  lda*n ));
                TESTING_CHECK( magma_dmalloc_cpu( &hR2, lda*n ));
                TESTING_CHECK( magma_dmalloc_cpu( &hA,  lda*k ));
                TESTING_CHECK( magma_dmalloc_cpu( &hB,  lda*k ));
                for( int dev = 0; dev < ngpu; ++dev ) {
                    magma_int_t nlocal = ((n / nb) / ngpu + 1) * nb;
                    magma_setdevice( dev );
                    TESTING_CHECK( magma_dmalloc( &dC[dev], ldda*nlocal ));
                    TESTING_CHECK( magma_dmalloc( &dA[dev], ldda*k*2    ));
                    //TESTING_CHECK( magma_dmalloc( &dB[dev], ldda*k      ));
                    for( int i = 0; i < nqueue; ++i ) {
                        magma_queue_create( dev, &queues[dev][i] );
                    }
                    queues0[dev] = queues[dev][0];
                }
                
                size = lda*n;
                lapackf77_dlarnv( &ione, ISEED, &size, hC );
                size = lda*k;
                lapackf77_dlarnv( &ione, ISEED, &size, hA );
                lapackf77_dlarnv( &ione, ISEED, &size, hB );
            
                // for error checks
                double Anorm = lapackf77_dlange( "F", &n, &k, hA, &lda, work );
                double Bnorm = lapackf77_dlange( "F", &n, &k, hB, &lda, work );
                double Cnorm = safe_lapackf77_dlansy( "F", lapack_uplo_const(opts.uplo), &n, hC, &lda, work );
                
                /* ====================================================================
                   Performs operation using MAGMA
                   =================================================================== */
                magma_dsetmatrix_1D_col_bcyclic( ngpu, n, n, nb, hC, lda, dC, ldda, queues0 );
                for( int dev = 0; dev < ngpu; ++dev ) {
                    magma_setdevice( dev );
                    dB[dev] = dA[dev] + ldda*k;
                    magma_dsetmatrix( n, k, hA, lda, dA[dev], ldda, opts.queue );
                    magma_dsetmatrix( n, k, hB, lda, dB[dev], ldda, opts.queue );
                }
                
                gpu_time = magma_sync_wtime(0);
                
                // only Lower, NoTrans implemented
                if ( opts.version == 1 ) {
                    magmablas_dsyr2k_mgpu2(
                        MagmaLower, MagmaNoTrans, n-offset, k,
                        alpha, dA, ldda, 0,
                               dB, ldda, 0,
                        beta,  dC, ldda, offset,
                        ngpu, nb, queues, nqueue );
                }
                else if ( opts.version == 2 ) {
                    // see src/obsolete and magmablas/obsolete
                    printf( "magmablas_dsyr2k_mgpu_spec not compiled\n" );
                    //magmablas_dsyr2k_mgpu_spec(
                    //    MagmaLower, MagmaNoTrans, n-offset, k,
                    //    alpha, dA, ldda, 0,
                    //           dB, ldda, 0,
                    //    beta,  dC, ldda, offset,
                    //    ngpu, nb, queues, nqueue );
                }
                else {
#ifdef ICHI
                    // assumes that dA and dB are stored consecutively?
                    magma_dsyr2k_mgpu(
                        ngpu, MagmaLower, MagmaNoTrans, nb, n-offset, k,
                        alpha, dA, ldda,
                               //dB, ldda,
                        beta,  dC, ldda, offset,
                        nqueue, queues );
#endif
                }
                
                gpu_time = magma_sync_wtime(0) - gpu_time;
                gpu_perf = gflops / gpu_time;
                
                // Get dC back to the CPU to compare with the CPU result.
                magma_dgetmatrix_1D_col_bcyclic( ngpu, n, n, nb, dC, ldda, hR, lda, queues0 );
                
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                if ( opts.lapack || opts.check ) {
                    // store ||V||*||W|| + ||A||
                    magma_int_t n_offset = n - offset;
                    Anorm  = lapackf77_dlange("f", &n_offset, &k, hA, &lda, work );
                    Anorm *= lapackf77_dlange("f", &n_offset, &k, hB, &lda, work );
                    Anorm += lapackf77_dlange("f", &n_offset, &n_offset, &hC[offset + offset*lda], &lda, work );
                    
                    cpu_time = magma_wtime();
                    blasf77_dsyr2k( "Lower", "NoTrans", &n_offset, &k,
                                    &alpha, hA, &lda,
                                            hB, &lda,
                                    &beta,  &hC[offset + offset*lda], &lda );
                    cpu_time = magma_wtime() - cpu_time;
                    cpu_perf = gflops / cpu_time;
                    
                    // compute relative error ||R||/||A||, where R := A_magma - A_lapack = R - A
                    size = lda*n;
                    blasf77_daxpy( &size, &c_neg_one, hC, &ione, hR, &ione );
                    error = safe_lapackf77_dlansy("fro", "Lower", &n_offset, &hR[offset + offset*lda], &lda, work)
                            / (2*sqrt(double(k+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);
                    
                    printf( "%5lld %5lld %5lld %5lld   %7.1f (%7.4f)   %7.1f (%7.4f)   %8.2e   %s\n",
                            (long long) n, (long long) k, (long long) nb, (long long) offset,
                            cpu_perf, cpu_time, gpu_perf, gpu_time,
                            error, (error < tol ? "ok" : "failed"));
                            //, gpu_perf2, gpu_time2, error, error2 );
                    status += ! (error < tol);
                }
                else {
                    printf( "%5lld %5lld %5lld %5lld     ---   (  ---  )   %7.1f (%7.4f)     ---\n",
                            (long long) n, (long long) k, (long long) nb, (long long) offset,
                            gpu_perf, gpu_time );
                }
                
                magma_free_cpu( hC  );
                magma_free_cpu( hR  );
                magma_free_cpu( hR2 );
                magma_free_cpu( hA  );
                magma_free_cpu( hB  );
                for( int dev = 0; dev < ngpu; ++dev ) {
                    magma_setdevice( dev );
                    magma_free( dC[dev] );
                    magma_free( dA[dev] );
                    //magma_free( dB[dev] );
                    for( int i = 0; i < nqueue; ++i ) {
                        magma_queue_destroy( queues[dev][i] );
                    }
                }
                fflush( stdout );
            }
            if ( opts.niter > 1 ) {
                printf( "\n" );
            }
        } // offset
        printf( "\n" );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
