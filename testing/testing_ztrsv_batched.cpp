/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> c d s
       @author Tingxing Dong

*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>  // for CUDA_VERSION

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "magma_threadsetting.h"
#endif


#define h_A(i,j,s) (h_A + (i) + (j)*lda + (s)*lda*Ak)


//#define PRINTMAT

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrsm_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time=0, cublas_perf=0, cublas_time=0, cpu_perf=0, cpu_time=0;
    double          magma_error, cublas_error, normx, normr, normA, work[1];
    magma_int_t i, j, s, N, info;
    magma_int_t Ak;
    magma_int_t sizeA, sizeB;
    magma_int_t lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;

    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    
    magmaDoubleComplex *h_A, *h_b, *h_bcublas, *h_bmagma, *h_blapack, *h_x;
    magmaDoubleComplex *d_A, *d_b;
    magmaDoubleComplex **d_A_array = NULL;
    magmaDoubleComplex **d_b_array = NULL;
    
    magmaDoubleComplex **dwork_array = NULL;

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_ONE;
    magma_int_t status = 0;
    magma_int_t batchCount;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    batchCount = opts.batchcount;
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% uplo = %s, transA = %s, diag = %s \n",
           lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%% BatchCount   N   MAGMA Gflop/s (ms)   CUBLAS Gflop/s (ms)    CPU Gflop/s (ms)      MAGMA   CUBLAS error\n");
    printf("%%========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.msize[itest];

            gflops = FLOPS_ZTRSM(opts.side, N, 1) / 1e9 * batchCount;

            lda = Ak = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default

            sizeA = lda*Ak*batchCount;
            sizeB = N*batchCount;

            TESTING_MALLOC_CPU( h_A,       magmaDoubleComplex, sizeA  );
            TESTING_MALLOC_CPU( h_b,       magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( h_x,       magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( h_blapack, magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( h_bcublas, magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( h_bmagma,  magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( ipiv,      magma_int_t,        Ak      );
            
            TESTING_MALLOC_DEV( d_A,       magmaDoubleComplex, ldda*Ak*batchCount );
            TESTING_MALLOC_DEV( d_b,       magmaDoubleComplex, N*batchCount  );
            
            TESTING_MALLOC_DEV( d_A_array,   magmaDoubleComplex*, batchCount );
            TESTING_MALLOC_DEV( d_b_array,   magmaDoubleComplex*, batchCount );
            TESTING_MALLOC_DEV( dwork_array, magmaDoubleComplex*, batchCount );


            magmaDoubleComplex_ptr dwork=NULL; // invA and work are workspace in ztrsm
            magma_int_t dwork_batchSize = N;
            TESTING_MALLOC_DEV( dwork, magmaDoubleComplex, dwork_batchSize * batchCount );
    
            magma_zset_pointer( dwork_array, dwork, N, 0, 0, dwork_batchSize, batchCount, opts.queue );

            memset( h_bmagma, 0, batchCount*N*sizeof(magmaDoubleComplex) );
            magmablas_zlaset( MagmaFull, N, batchCount, c_zero, c_zero, dwork, N, opts.queue );

            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );

            for (s=0; s < batchCount; s++) {
                lapackf77_zgetrf( &Ak, &Ak, h_A + s * lda * Ak, &lda, ipiv, &info );
                for( j = 0; j < Ak; ++j ) {
                    for( i = 0; i < j; ++i ) {
                        *h_A(i,j,s) = *h_A(j,i,s);
                    }
                }
            }

            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_b );
            memcpy( h_blapack, h_b, sizeB*sizeof(magmaDoubleComplex) );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( Ak, Ak*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N,  batchCount, h_b, N, d_b, N, opts.queue );

            magma_zset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*Ak, batchCount, opts.queue );
            magma_zset_pointer( d_b_array, d_b, N, 0, 0, N, batchCount, opts.queue );
            magma_zset_pointer( dwork_array, dwork, N, 0, 0, N, batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );

            magmablas_ztrsv_work_batched(opts.uplo, opts.transA, opts.diag, 
                                    N, d_A_array, ldda,
                                    d_b_array, 1, dwork_array, batchCount, opts.queue); 

            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_zgetmatrix( N, batchCount, dwork, N, h_bmagma, N, opts.queue );


            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_zsetmatrix( N, batchCount, h_b, N, d_b, N, opts.queue );
            magma_zset_pointer( d_b_array, d_b, N, 0, 0, N, batchCount, opts.queue );

            // CUBLAS version <= 6.0 has magmaDoubleComplex **            dA_array, no cast needed.
            // CUBLAS version    6.5 has magmaDoubleComplex const**       dA_array, requiring cast.
            // Correctly, it should be   magmaDoubleComplex const* const* dA_array, to avoid requiring cast.
            #if CUDA_VERSION >= 6050
                cublas_time = magma_sync_wtime( opts.queue );
                cublasZtrsmBatched(
                    opts.handle, cublas_side_const(MagmaLeft), cublas_uplo_const(opts.uplo),
                    cublas_trans_const(opts.transA), cublas_diag_const(opts.diag),
                    N, 1, &alpha,
                    (const magmaDoubleComplex**) d_A_array, ldda,
                    d_b_array, N, batchCount);
                cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
                cublas_perf = gflops / cublas_time;
            #else
                MAGMA_UNUSED( alpha );
            #endif

            magma_zgetmatrix( N, batchCount, d_b, N, h_bcublas, N, opts.queue );
            
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
                for (s=0; s < batchCount; s++) {
                    blasf77_ztrsv(
                        lapack_uplo_const(opts.uplo),
                        lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                        &N, 
                        h_A + s * lda * Ak, &lda,
                        h_blapack + s * N,  &ione);
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
            // ||b - Ax|| / (||A||*||x||)
            magma_error  = 0;
            cublas_error = 0;
            for (s=0; s < batchCount; s++) {
                // error for CUBLAS
                normA = lapackf77_zlange( "F", &N, &N, h_A + s * lda * Ak, &lda, work );
                double err;

                #if CUDA_VERSION >= 6050
                normx = lapackf77_zlange( "F", &N, &ione, h_bcublas + s * N, &ione, work );
                blasf77_ztrmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &N,
                               h_A + s * lda * Ak, &lda,
                               h_bcublas + s * N, &ione );

                blasf77_zaxpy( &N, &c_neg_one, h_b + s * N, &ione, h_bcublas + s * N, &ione );
                normr = lapackf77_zlange( "F", &N, &ione, h_bcublas + s * N, &N, work );
                err = normr / (normA*normx);
                
                if ( isnan(err) || isinf(err) ) {
                    printf("error for matrix %d cublas_error = %7.2f where normr=%7.2f normx=%7.2f and normA=%7.2f\n", 
                            s, err, normr, normx, normA);
                    cublas_error = err;
                    break;
                }
                cublas_error = max( err, cublas_error );
                #endif

                // error for MAGMA
                normx = lapackf77_zlange( "F", &N, &ione, h_bmagma + s * N, &ione, work );
                blasf77_ztrmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &N,
                               h_A + s * lda * Ak, &lda,
                               h_bmagma  + s * N, &ione );

                blasf77_zaxpy( &N, &c_neg_one, h_b + s * N, &ione, h_bmagma + s * N, &ione );
                normr = lapackf77_zlange( "F", &N, &ione, h_bmagma + s * N, &N, work );
                err = normr / (normA*normx);

                if ( isnan(err) || isinf(err) ) {
                    printf("error for matrix %d magma_error = %7.2f where normr=%7.2f normx=%7.2f and normA=%7.2f\n", 
                            s, err, normr, normx, normA);
                    magma_error = err;
                    break;
                }
                magma_error = max( err, magma_error );
            }
            bool okay = (magma_error < tol && cublas_error < tol);
            status += ! okay;

            if ( opts.lapack ) {
                printf("%10d %5d    %7.2f (%7.2f)     %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %s\n",
                        (int)batchCount, (int) N,
                        magma_perf,  1000.*magma_time,
                        cublas_perf, 1000.*cublas_time,
                        cpu_perf,    1000.*cpu_time,
                        magma_error, cublas_error,
                        (okay ? "ok" : "failed"));
            }
            else {
                printf("%10d %5d    %7.2f (%7.2f)     %7.2f (%7.2f)     ---   (  ---  )   %8.2e   %8.2e   %s\n",
                        (int)batchCount, (int) N,
                        magma_perf,  1000.*magma_time,
                        cublas_perf, 1000.*cublas_time,
                        magma_error, cublas_error,
                        (okay ? "ok" : "failed"));
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_b );
            TESTING_FREE_CPU( h_x );
            TESTING_FREE_CPU( h_blapack );
            TESTING_FREE_CPU( h_bcublas );
            TESTING_FREE_CPU( h_bmagma  );
            TESTING_FREE_CPU( ipiv );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_b );
            TESTING_FREE_DEV( d_A_array );
            TESTING_FREE_DEV( d_b_array );

            TESTING_FREE_DEV( dwork );
            TESTING_FREE_DEV( dwork_array );
            
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
