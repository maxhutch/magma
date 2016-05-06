/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zgemm_batched.cpp normal z -> s, Mon May  2 23:31:21 2016
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

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    float          magma_error, cublas_error, Cnorm, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magma_int_t NN;
    magma_int_t batchCount;

    float *h_A, *h_B, *h_C, *h_Cmagma, *h_Ccublas;
    float *d_A, *d_B, *d_C;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_S_MAKE( -0.48,  0.38 );
    float **A_array = NULL;
    float **B_array = NULL;
    float **C_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    
    batchCount = opts.batchcount;

    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("%% If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "%% transA = %s, transB = %s\n",
           lapack_trans_const(opts.transA),
           lapack_trans_const(opts.transB));
    printf("%% BatchCount   M     N     K   MAGMA Gflop/s (ms)   CUBLAS Gflop/s (ms)    CPU Gflop/s (ms)   MAGMA error  CUBLAS error\n");
    printf("%%======================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_SGEMM( M, N, K ) / 1e9 * batchCount;

            if ( opts.transA == MagmaNoTrans ) {
                lda = Am = M;
                An = K;
            } else {
                lda = Am = K;
                An = M;
            }
            
            if ( opts.transB == MagmaNoTrans ) {
                ldb = Bm = K;
                Bn = N;
            } else {
                ldb = Bm = N;
                Bn = K;
            }
            ldc = M;
            
            NN = N * batchCount;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*An*batchCount;
            sizeB = ldb*Bn*batchCount;
            sizeC = ldc*N*batchCount;
            
            TESTING_MALLOC_CPU( h_A,  float, sizeA );
            TESTING_MALLOC_CPU( h_B,  float, sizeB );
            TESTING_MALLOC_CPU( h_C,  float, sizeC  );
            TESTING_MALLOC_CPU( h_Cmagma,  float, sizeC  );
            TESTING_MALLOC_CPU( h_Ccublas, float, sizeC  );

            TESTING_MALLOC_DEV( d_A, float, ldda*An*batchCount );
            TESTING_MALLOC_DEV( d_B, float, lddb*Bn*batchCount );
            TESTING_MALLOC_DEV( d_C, float, lddc*N*batchCount  );

            TESTING_MALLOC_DEV( A_array, float*, batchCount );
            TESTING_MALLOC_DEV( B_array, float*, batchCount );
            TESTING_MALLOC_DEV( C_array, float*, batchCount );

            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_slarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_slarnv( &ione, ISEED, &sizeC, h_C );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_ssetmatrix( Am, An*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_ssetmatrix( Bm, Bn*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            magma_ssetmatrix( M, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );
            
            magma_sset_pointer( A_array, d_A, ldda, 0, 0, ldda*An, batchCount, opts.queue );
            magma_sset_pointer( B_array, d_B, lddb, 0, 0, lddb*Bn, batchCount, opts.queue );
            magma_sset_pointer( C_array, d_C, lddc, 0, 0, lddc*N,  batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_sgemm_batched(opts.transA, opts.transB, M, N, K,
                             alpha, A_array, ldda,
                                    B_array, lddb,
                             beta,  C_array, lddc, batchCount, opts.queue);
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_sgetmatrix( M, N*batchCount, d_C, lddc, h_Cmagma, ldc, opts.queue );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */

            magma_ssetmatrix( M, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );

            cublas_time = magma_sync_wtime( opts.queue );

            cublasSgemmBatched(opts.handle, cublas_trans_const(opts.transA), cublas_trans_const(opts.transB), M, N, K,
                               &alpha, (const float**) A_array, ldda,
                               (const float**) B_array, lddb,
                               &beta,  C_array, lddc, batchCount );

            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_sgetmatrix( M, N*batchCount, d_C, lddc, h_Ccublas, ldc, opts.queue );
          
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
                   blasf77_sgemm(
                               lapack_trans_const(opts.transA), lapack_trans_const(opts.transB),
                               &M, &N, &K,
                               &alpha, h_A + i*lda*An, &lda,
                                       h_B + i*ldb*Bn, &ldb,
                               &beta,  h_C + i*ldc*N, &ldc );
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
                // compute relative error for both magma & cublas, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                magma_error  = 0;
                cublas_error = 0;
                for (int s=0; s < batchCount; s++)
                {
                    magma_int_t C_batchSize = ldc * N;
 
                    Cnorm = lapackf77_slange( "M", &M, &N, h_C + s*C_batchSize, &ldc, work );

                    // ----- magma error
                    blasf77_saxpy( &C_batchSize, &c_neg_one, h_C + s*C_batchSize, &ione, h_Cmagma + s*C_batchSize, &ione );
                    float err = lapackf77_slange( "M", &M, &N, h_Cmagma + s*C_batchSize, &ldc, work ) / Cnorm;

                    if ( isnan(err) || isinf(err) ) {
                        magma_error = err;
                        break;
                    }
                    magma_error = max( err, magma_error );

                    // ----- cublas error
                    blasf77_saxpy( &C_batchSize, &c_neg_one, h_C + s*C_batchSize, &ione, h_Ccublas + s*C_batchSize, &ione );
                    err = lapackf77_slange( "M", &M, &N, h_Ccublas + s*C_batchSize, &ldc, work ) / Cnorm;
                    
                    if ( isnan(err) || isinf(err) ) {
                        cublas_error = err;
                        break;
                    }
                    cublas_error = max( err, cublas_error );
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%10d %5d %5d %5d    %7.2f (%7.2f)     %7.2f (%7.2f)   %7.2f (%7.2f)      %8.2e     %8.2e  %s\n",
                   (int) batchCount, (int) M, (int) N, (int) K,
                   magma_perf,  1000.*magma_time,
                   cublas_perf, 1000.*cublas_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, cublas_error, (okay ? "ok" : "failed"));
            }
            else {
                // compute relative error for magma, relative to cublas
                Cnorm = lapackf77_slange( "M", &M, &NN, h_Ccublas, &ldc, work );
                blasf77_saxpy( &sizeC, &c_neg_one, h_Ccublas, &ione, h_Cmagma, &ione );
                magma_error = lapackf77_slange( "M", &M, &NN, h_Cmagma, &ldc, work ) / Cnorm;

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%10d %5d %5d %5d    %7.2f (%7.2f)     %7.2f (%7.2f)     ---   (  ---  )    %8.2e     ---  %s\n",
                   (int) batchCount, (int) M, (int) N, (int) K,
                   magma_perf,  1000.*magma_time,
                   cublas_perf, 1000.*cublas_time,
                   magma_error, (okay ? "ok" : "failed") );
            }
            
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_B  );
            TESTING_FREE_CPU( h_C  );
            TESTING_FREE_CPU( h_Cmagma  );
            TESTING_FREE_CPU( h_Ccublas );

            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            TESTING_FREE_DEV( d_C );
            TESTING_FREE_DEV( A_array );
            TESTING_FREE_DEV( B_array );
            TESTING_FREE_DEV( C_array );

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
