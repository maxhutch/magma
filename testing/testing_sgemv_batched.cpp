/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zgemv_batched.cpp normal z -> s, Mon May  2 23:31:21 2016
       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
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

#if defined(_OPENMP)
#include <omp.h>
#include "magma_threadsetting.h"
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgemm_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    float          magma_error, work[1];
    magma_int_t M, N, Xm, Ym, lda, ldda;
    magma_int_t sizeA, sizeX, sizeY;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magma_int_t batchCount;

    float *h_A, *h_X, *h_Y, *h_Ymagma;
    float *d_A, *d_X, *d_Y;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_S_MAKE( -0.48,  0.38 );
    float **A_array = NULL;
    float **X_array = NULL;
    float **Y_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    batchCount = opts.batchcount;
    opts.lapack |= opts.check;

    float tol = opts.tolerance * lapackf77_slamch("E");

    printf("%% trans = %s\n", lapack_trans_const(opts.transA) );
    printf("%% BatchCount   M     N   MAGMA Gflop/s (ms)    CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_SGEMV( M, N ) / 1e9 * batchCount;

            if ( opts.transA == MagmaNoTrans ) {
                Xm = N;
                Ym = M;
            } else {
                Xm = M;
                Ym = N;
            }

            sizeA = lda*N*batchCount;
            sizeX = incx*Xm*batchCount;
            sizeY = incy*Ym*batchCount;

            TESTING_MALLOC_CPU( h_A,  float, sizeA );
            TESTING_MALLOC_CPU( h_X,  float, sizeX );
            TESTING_MALLOC_CPU( h_Y,  float, sizeY  );
            TESTING_MALLOC_CPU( h_Ymagma,  float, sizeY  );

            TESTING_MALLOC_DEV( d_A, float, ldda*N*batchCount );
            TESTING_MALLOC_DEV( d_X, float, sizeX );
            TESTING_MALLOC_DEV( d_Y, float, sizeY );

            TESTING_MALLOC_DEV( A_array, float*, batchCount );
            TESTING_MALLOC_DEV( X_array, float*, batchCount );
            TESTING_MALLOC_DEV( Y_array, float*, batchCount );

            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_slarnv( &ione, ISEED, &sizeX, h_X );
            lapackf77_slarnv( &ione, ISEED, &sizeY, h_Y );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_ssetmatrix( M, N*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_ssetvector( Xm*batchCount, h_X, incx, d_X, incx, opts.queue );
            magma_ssetvector( Ym*batchCount, h_Y, incy, d_Y, incy, opts.queue );
            
            magma_sset_pointer( A_array, d_A, ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_sset_pointer( X_array, d_X, 1, 0, 0, incx*Xm, batchCount, opts.queue );
            magma_sset_pointer( Y_array, d_Y, 1, 0, 0, incy*Ym, batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_sgemv_batched(opts.transA, M, N,
                             alpha, A_array, ldda,
                                    X_array, incx,
                             beta,  Y_array, incy, batchCount, opts.queue);
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_sgetvector( Ym*batchCount, d_Y, incy, h_Ymagma, incy, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int i=0; i < batchCount; i++)
                {
                   blasf77_sgemv(
                               lapack_trans_const(opts.transA),
                               &M, &N,
                               &alpha, h_A + i*lda*N, &lda,
                                       h_X + i*Xm, &incx,
                               &beta,  h_Y + i*Ym, &incy );
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for magma, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                magma_error = 0;
                for (int s=0; s < batchCount; s++) {
                    float Anorm = lapackf77_slange( "F", &M, &N, h_A + s * lda * N, &lda, work );
                    float Xnorm = lapackf77_slange( "F", &Xm, &ione, h_X + s * Xm, &Xm, work );
                    
                    blasf77_saxpy( &Ym, &c_neg_one, h_Y + s * Ym, &incy, h_Ymagma + s * Ym, &incy );
                    float err = lapackf77_slange( "F", &Ym, &ione, h_Ymagma + s * Ym, &Ym, work ) / (Anorm * Xnorm);

                    if ( isnan(err) || isinf(err) ) {
                      magma_error = err;
                      break;
                    }
                    magma_error = max( err, magma_error );
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%10d %5d %5d    %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                   (int) batchCount, (int) M, (int) N,
                   magma_perf,  1000.*magma_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%10d %5d %5d    %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                   (int) batchCount, (int) M, (int) N,
                   magma_perf,  1000.*magma_time);
            }
            
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_X  );
            TESTING_FREE_CPU( h_Y  );
            TESTING_FREE_CPU( h_Ymagma  );

            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_X );
            TESTING_FREE_DEV( d_Y );
            TESTING_FREE_DEV( A_array );
            TESTING_FREE_DEV( X_array );
            TESTING_FREE_DEV( Y_array );

            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
