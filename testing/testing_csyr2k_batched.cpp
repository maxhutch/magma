/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016

       @generated from testing/testing_zsyr2k_batched.cpp, normal z -> c, Tue Aug 30 09:39:17 2016
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
#endif
#include "../control/magma_threadsetting.h"  // internal header

#define PRECISION_c

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing csyr2k_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    float          magma_error, Cnorm, work[1];
    magma_int_t N, K;
    magma_int_t An, Ak, Bn, Bk;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    magmaFloatComplex *h_A, *h_B, *h_C, *h_Cmagma;
    magmaFloatComplex *d_A, *d_B, *d_C;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex alpha = MAGMA_C_MAKE(  0.29, -0.86 );
    magmaFloatComplex beta  = MAGMA_C_MAKE( -0.48,  0.38 );
    
    magmaFloatComplex **A_array = NULL;
    magmaFloatComplex **B_array = NULL;
    magmaFloatComplex **C_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    
    batchCount = opts.batchcount;

    float tol = opts.tolerance * lapackf77_slamch("E");
    opts.lapack |= opts.check;
    
    TESTING_CHECK( magma_malloc((void**)&A_array, batchCount*sizeof(magmaFloatComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&B_array, batchCount*sizeof(magmaFloatComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&C_array, batchCount*sizeof(magmaFloatComplex*)) );
            
    printf("%% If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "%% relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "%% uplo = %s, trans = %s\n",
           lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA));

    if(opts.transA == MagmaConjTrans){
        opts.transA = MagmaTrans;
        printf("%% WARNING: transA = MagmaConjTrans changed to MagmaTrans\n");
    }
    
    printf("%% BatchCount   N     K   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)    MAGMA error\n");
    printf("%%======================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = (batchCount * FLOPS_CHER2K( K, N )) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                ldb = Bn = N;
                Ak = K;
                Bk = K;
            } else {
                lda = An = K;
                ldb = Bn = K;
                Ak = N;
                Bk = N;
            }
            
            ldc = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*Ak*batchCount;
            sizeB = ldb*Bk*batchCount;
            sizeC = ldc*N*batchCount;
            
            TESTING_CHECK( magma_cmalloc_cpu(&h_A, sizeA) );
            TESTING_CHECK( magma_cmalloc_cpu(&h_B, sizeB) );
            TESTING_CHECK( magma_cmalloc_cpu(&h_C, sizeC) );
            TESTING_CHECK( magma_cmalloc_cpu(&h_Cmagma, sizeC) );

            TESTING_CHECK( magma_cmalloc(&d_A, ldda*Ak*batchCount) );
            TESTING_CHECK( magma_cmalloc(&d_B, lddb*Bk*batchCount) );
            TESTING_CHECK( magma_cmalloc(&d_C, lddc*N*batchCount)  );
            
            /* Initialize the matrices */
            lapackf77_clarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_clarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_clarnv( &ione, ISEED, &sizeC, h_C );
            for (int i=0; i < batchCount; i++){
                magma_cmake_symmetric( N, h_C + i * ldc * N, ldc );
            }
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_csetmatrix( An, Ak*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_csetmatrix( Bn, Bk*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            magma_csetmatrix( N, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );
            
            magma_cset_pointer( A_array, d_A, ldda, 0, 0, ldda*Ak, batchCount, opts.queue );
            magma_cset_pointer( B_array, d_B, lddb, 0, 0, lddb*Bk, batchCount, opts.queue );
            magma_cset_pointer( C_array, d_C, lddc, 0, 0, lddc*N,  batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_csyr2k_batched(opts.uplo, opts.transA, N, K, 
                             alpha, A_array, ldda, 
                                    B_array, lddb, 
                             beta, C_array, lddc, batchCount, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_cgetmatrix( N, N*batchCount, d_C, lddc, h_Cmagma, ldc, opts.queue );
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
                   blasf77_csyr2k(
                               lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA),
                               &N, &K,
                               &alpha, h_A + i*lda*Ak, &lda,
                                       h_B + i*ldb*Bk, &ldb,
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
                #ifdef MAGMA_WITH_MKL
                // work around MKL bug in multi-threaded clansy
                magma_int_t la_threads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads( 1 );
                #endif
                
                // compute relative error of magma relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                magma_error  = 0;
                for (int s=0; s < batchCount; s++)
                {
                    magma_int_t C_batchSize = ldc * N;
                    
                    Cnorm = lapackf77_clansy( "fro", lapack_uplo_const(opts.uplo), &N, h_C + s*C_batchSize, &ldc, work );

                    // ----- magma error
                    blasf77_caxpy( &C_batchSize, &c_neg_one, h_C + s*C_batchSize, &ione, h_Cmagma + s*C_batchSize, &ione );
                    float err = lapackf77_clansy( "fro", lapack_uplo_const(opts.uplo), &N, h_Cmagma + s*C_batchSize, &ldc, work ) / Cnorm;

                    if ( isnan(err) || isinf(err) ) {
                        magma_error = err;
                        break;
                    }
                    magma_error = max( err, magma_error );
                }

                #ifdef MAGMA_WITH_MKL
                // end single thread to work around MKL bug
                magma_set_lapack_numthreads( la_threads );
                #endif
                
                bool okay = (magma_error < tol);
                status += ! okay;
                printf("%10lld %5lld %5lld    %7.2f (%7.2f)     %7.2f (%7.2f)     %8.2e  %s\n",
                   (long long) batchCount, (long long) N, (long long) K,
                   magma_perf,  1000.*magma_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, (okay ? "ok" : "failed"));
            }
            
            magma_free_cpu( h_A  );
            magma_free_cpu( h_B  );
            magma_free_cpu( h_C  );
            magma_free_cpu( h_Cmagma  );

            magma_free( d_A );
            magma_free( d_B );
            magma_free( d_C );
            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    magma_free( A_array );
    magma_free( B_array );
    magma_free( C_array );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
