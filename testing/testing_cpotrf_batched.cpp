/*
   -- MAGMA (version 2.0.2) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date May 2016

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from testing/testing_zpotrf_batched.cpp normal z -> c, Mon May  2 23:31:23 2016
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>  // cudaMemset

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#include "magma_threadsetting.h"  // to work around MKL bug

#if defined(_OPENMP)
#include <omp.h>
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cpotrf_batched
*/

int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaFloatComplex *h_A, *h_R;
    magmaFloatComplex *d_A;
    magma_int_t N, n2, lda, ldda, info;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    float      work[1], error;
    magma_int_t status = 0;
    magmaFloatComplex **d_A_array = NULL;
    magma_int_t *dinfo_magma;
    magma_int_t *cpu_info;

    magma_int_t batchCount;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    batchCount = opts.batchcount;
    float tol = opts.tolerance * lapackf77_slamch("E");

    magma_queue_t queue = opts.queue;

    printf("%% BatchCount   N    CPU Gflop/s (ms)    GPU Gflop/s (ms)   ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("%%===================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N   = opts.nsize[itest];
            lda = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            n2  = lda* N  * batchCount;

            gflops = batchCount * FLOPS_CPOTRF( N ) / 1e9;

            TESTING_MALLOC_CPU( cpu_info, magma_int_t, batchCount);
            TESTING_MALLOC_CPU( h_A, magmaFloatComplex, n2);
            TESTING_MALLOC_PIN( h_R, magmaFloatComplex, n2);
            TESTING_MALLOC_DEV(  d_A, magmaFloatComplex, ldda * N * batchCount);
            TESTING_MALLOC_DEV(  dinfo_magma,  magma_int_t, batchCount);
            
            TESTING_MALLOC_DEV( d_A_array, magmaFloatComplex*, batchCount );

            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            for (int i=0; i < batchCount; i++)
            {
               magma_cmake_hpd( N, h_A + i * lda * N, lda ); // need modification
            }
            
            magma_int_t columns = N * batchCount;
            lapackf77_clacpy( MagmaFullStr, &N, &(columns), h_A, &lda, h_R, &lda );

            magma_csetmatrix( N, columns, h_A, lda, d_A, ldda, opts.queue );

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            cudaMemset( dinfo_magma, 0, batchCount * sizeof(magma_int_t) );

            magma_cset_pointer( d_A_array, d_A, ldda, 0, 0, ldda * N, batchCount, queue );
            gpu_time = magma_sync_wtime( opts.queue );
            info = magma_cpotrf_batched( opts.uplo, N, d_A_array, ldda, dinfo_magma, batchCount, queue);
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1, opts.queue );
            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_cpotrf_batched matrix %d returned diag error %d\n", i, (int)cpu_info[i] );
                    status = -1;
                }
            }
            if (info != 0) {
                //printf("magma_cpotrf_batched returned argument error %d: %s.\n", (int) info, magma_strerror( info ));
                status = -1;
            }                
            if (status == -1)
                goto cleanup;


            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                // #define BATCHED_DISABLE_PARCPU
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++)
                {
                    magma_int_t locinfo;
                    lapackf77_cpotrf( lapack_uplo_const(opts.uplo), &N, h_A + s * lda * N, &lda, &locinfo );
                    if (locinfo != 0) {
                        printf("lapackf77_cpotrf matrix %d returned error %d: %s.\n",
                               (int) s, (int) locinfo, magma_strerror( locinfo ));
                    }
                }

                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
            
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                #ifdef MAGMA_WITH_MKL
                // work around MKL bug in multi-threaded clanhe
                magma_int_t la_threads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads( 1 );
                #endif
                
                magma_cgetmatrix( N, columns, d_A, ldda, h_R, lda, opts.queue );
                magma_int_t NN = lda*N;
                const char* uplo = lapack_uplo_const(opts.uplo);
                error = 0;
                for (int i=0; i < batchCount; i++)
                {
                    float Anorm, err;
                    blasf77_caxpy(&NN, &c_neg_one, h_A + i * lda*N, &ione, h_R + i * lda*N, &ione);
                    Anorm = lapackf77_clanhe("f", uplo, &N, h_A + i * lda*N, &lda, work);
                    err   = lapackf77_clanhe("f", uplo, &N, h_R + i * lda*N, &lda, work)
                          / Anorm;
                    if ( isnan(err) || isinf(err) ) {
                        error = err;
                        break;
                    }
                    error = max( err, error );
                }
                bool okay = (error < tol);
                status += ! okay;
                
                #ifdef MAGMA_WITH_MKL
                // end single thread to work around MKL bug
                magma_set_lapack_numthreads( la_threads );
                #endif
                
                printf("%10d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (int)batchCount, (int) N, cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                       error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%10d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---\n",
                       (int)batchCount, (int) N, gpu_perf, gpu_time*1000. );
            }
cleanup:
            TESTING_FREE_CPU( cpu_info );
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_A_array );
            TESTING_FREE_DEV( dinfo_magma );
            if (status == -1)
                break;
            fflush( stdout );
        }
        if (status == -1)
            break;

        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
