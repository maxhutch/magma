/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> c d s
       @author Chongxiao Cao
       @author Tingxing Dong
       @author Azzam Haidar
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
#include "batched_kernel_param.h"  // for TRI_NB; TODO: in control
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
    double          magma_error, cublas_error, lapack_error, work[1];
    magma_int_t M, N, info;
    magma_int_t Ak;
    magma_int_t sizeA, sizeB;
    magma_int_t lda, ldb, ldda, lddb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;

    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    
    magmaDoubleComplex *h_A, *h_B, *h_Bcublas, *h_Bmagma, *h_Blapack, *h_X;
    magmaDoubleComplex *d_A, *d_B;
    magmaDoubleComplex **d_A_array = NULL;
    magmaDoubleComplex **d_B_array = NULL;
    
    magmaDoubleComplex **dW1_displ  = NULL;
    magmaDoubleComplex **dW2_displ  = NULL;
    magmaDoubleComplex **dW3_displ  = NULL;
    magmaDoubleComplex **dW4_displ  = NULL;
    magmaDoubleComplex **dinvA_array = NULL;
    magmaDoubleComplex **dwork_array = NULL;

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magma_int_t status = 0;
    magma_int_t batchCount;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    batchCount = opts.batchcount;
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% side = %s, uplo = %s, transA = %s, diag = %s \n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%% BatchCount   M     N   MAGMA Gflop/s (ms)   CUBLAS Gflop/s (ms)    CPU Gflop/s (ms)      MAGMA     CUBLAS   LAPACK error\n");
    printf("%%=========================================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_ZTRSM(opts.side, M, N) / 1e9 * batchCount;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                Ak  = M;
            } else {
                lda = N;
                Ak  = N;
            }
            
            ldb = M;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default

            sizeA = lda*Ak*batchCount;
            sizeB = ldb*N*batchCount;
            magma_int_t NN = ldb*N;

            TESTING_MALLOC_CPU( h_A,       magmaDoubleComplex, sizeA  );
            TESTING_MALLOC_CPU( h_B,       magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( h_X,       magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( h_Blapack, magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( h_Bcublas, magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( h_Bmagma,  magmaDoubleComplex, sizeB   );
            TESTING_MALLOC_CPU( ipiv,      magma_int_t,        Ak      );
            
            TESTING_MALLOC_DEV( d_A,       magmaDoubleComplex, ldda*Ak*batchCount );
            TESTING_MALLOC_DEV( d_B,       magmaDoubleComplex, lddb*N*batchCount  );
            
            TESTING_MALLOC_DEV( d_A_array, magmaDoubleComplex*, batchCount );
            TESTING_MALLOC_DEV( d_B_array, magmaDoubleComplex*, batchCount );

            TESTING_MALLOC_DEV( dW1_displ,   magmaDoubleComplex*, batchCount );
            TESTING_MALLOC_DEV( dW2_displ,   magmaDoubleComplex*, batchCount );
            TESTING_MALLOC_DEV( dW3_displ,   magmaDoubleComplex*, batchCount );
            TESTING_MALLOC_DEV( dW4_displ,   magmaDoubleComplex*, batchCount );
            TESTING_MALLOC_DEV( dinvA_array, magmaDoubleComplex*, batchCount );
            TESTING_MALLOC_DEV( dwork_array, magmaDoubleComplex*, batchCount );

            magmaDoubleComplex* dinvA=NULL;
            magmaDoubleComplex* dwork=NULL; // invA and work are workspace in ztrsm
 
            magma_int_t dinvA_batchSize = magma_roundup( Ak, TRI_NB )*TRI_NB;
            magma_int_t dwork_batchSize = lddb*N;
            TESTING_MALLOC_DEV( dinvA, magmaDoubleComplex, dinvA_batchSize * batchCount );
            TESTING_MALLOC_DEV( dwork, magmaDoubleComplex, dwork_batchSize * batchCount );
    
            magma_zset_pointer( dwork_array, dwork, lddb, 0, 0, dwork_batchSize, batchCount, opts.queue );
            magma_zset_pointer( dinvA_array, dinvA, magma_roundup( Ak, TRI_NB ), 0, 0, dinvA_batchSize, batchCount, opts.queue );

            memset( h_Bmagma, 0, batchCount*ldb*N*sizeof(magmaDoubleComplex) );
            magmablas_zlaset( MagmaFull, lddb, N*batchCount, c_zero, c_zero, dwork, lddb, opts.queue );

            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );

            for (int s=0; s < batchCount; s++) {
                lapackf77_zgetrf( &Ak, &Ak, h_A + s * lda * Ak, &lda, ipiv, &info );
                for( int j = 0; j < Ak; ++j ) {
                    for( int i = 0; i < j; ++i ) {
                        *h_A(i,j,s) = *h_A(j,i,s);
                    }
                }
            }

            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            memcpy( h_Blapack, h_B, sizeB*sizeof(magmaDoubleComplex) );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( Ak, Ak*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( M,  N*batchCount, h_B, ldb, d_B, lddb, opts.queue );

            magma_zset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*Ak, batchCount, opts.queue );
            magma_zset_pointer( d_B_array, d_B, lddb, 0, 0, lddb*N, batchCount, opts.queue );
            magma_zset_pointer( dwork_array, dwork, lddb, 0, 0, lddb*N, batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            #if 1
                magmablas_ztrsm_outofplace_batched(
                    opts.side, opts.uplo, opts.transA, opts.diag, 1,
                    M, N, alpha,
                    d_A_array,    ldda, // dA
                    d_B_array,    lddb, // dB
                    dwork_array,  lddb, // dX output
                    dinvA_array,  dinvA_batchSize,
                    dW1_displ,   dW2_displ,
                    dW3_displ,   dW4_displ,
                    1, batchCount, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                magma_zgetmatrix( M, N*batchCount, dwork, lddb, h_Bmagma, ldb, opts.queue );
            #else
                magmablas_ztrsm_batched(
                    opts.side, opts.uplo, opts.transA, opts.diag,
                    M, N, alpha,
                    d_A_array, ldda,
                    d_B_array, lddb,
                    batchCount, opts.queue );
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                magma_zgetmatrix( M, N*batchCount, d_B, lddb, h_Bmagma, ldb, opts.queue );
            #endif
       
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_zsetmatrix( M, N*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            magma_zset_pointer( d_B_array, d_B, lddb, 0, 0, lddb*N, batchCount, opts.queue );

            // CUBLAS version <= 6.0 has magmaDoubleComplex **            dA_array, no cast needed.
            // CUBLAS version    6.5 has magmaDoubleComplex const**       dA_array, requiring cast.
            // Correctly, it should be   magmaDoubleComplex const* const* dA_array, to avoid requiring cast.
            #if CUDA_VERSION >= 6050
                cublas_time = magma_sync_wtime( opts.queue );
                cublasZtrsmBatched(
                    opts.handle, cublas_side_const(opts.side), cublas_uplo_const(opts.uplo),
                    cublas_trans_const(opts.transA), cublas_diag_const(opts.diag),
                    M, N, &alpha,
                    (const magmaDoubleComplex**) d_A_array, ldda,
                    d_B_array, lddb, batchCount);
                cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
                cublas_perf = gflops / cublas_time;
            #endif

            magma_zgetmatrix( M, N*batchCount, d_B, lddb, h_Bcublas, ldb, opts.queue );
            
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
                for (int s=0; s < batchCount; s++) {
                    blasf77_ztrsm(
                        lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                        lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                        &M, &N, &alpha,
                        h_A       + s * lda * Ak, &lda,
                        h_Blapack + s * ldb * N,  &ldb );
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
            // ||b - 1/alpha*A*x|| / (||A||*||x||)
            magmaDoubleComplex alpha2 = MAGMA_Z_DIV( c_one, alpha );
            double normR, normX, normA=0.;
            magma_error  = 0.0;
            cublas_error = 0.0;

            memcpy( h_X, h_Bmagma, sizeB*sizeof(magmaDoubleComplex) );

            // check magma
            for (int s=0; s < batchCount; s++) {
                normA = lapackf77_zlange( "M", &Ak, &Ak, h_A + s * lda * Ak, &lda, work );
                blasf77_ztrmm(
                    lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                    lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                    &M, &N, &alpha2,
                    h_A + s * lda * Ak, &lda,
                    h_X + s * ldb * N,  &ldb );

                blasf77_zaxpy( &NN, &c_neg_one, h_B + s * ldb * N, &ione, h_X + s * ldb * N, &ione );

                normR = lapackf77_zlange( "M", &M, &N, h_X + s * ldb * N,      &ldb, work );
                normX = lapackf77_zlange( "M", &M, &N, h_Bmagma + s * ldb * N, &ldb, work );
                double err = normR/(normX*normA);

                if ( isnan(err) || isinf(err) ) {
                    printf("error for matrix %d magma_error = %7.2f where normR=%7.2f normX=%7.2f and normA=%7.2f\n",
                           s, err, normR, normX, normA);
                    magma_error = err;
                    break;
                }
                magma_error = max( err, magma_error );
            }

            memcpy( h_X, h_Bcublas, sizeB*sizeof(magmaDoubleComplex) );
            // check cublas
            #if CUDA_VERSION >= 6050
            for (int s=0; s < batchCount; s++) {
                normA = lapackf77_zlange( "M", &Ak, &Ak, h_A + s * lda * Ak, &lda, work );
                blasf77_ztrmm(
                    lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                    lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                    &M, &N, &alpha2,
                    h_A + s * lda * Ak, &lda,
                    h_X + s * ldb * N, &ldb );

                blasf77_zaxpy( &NN, &c_neg_one, h_B + s * ldb * N, &ione, h_X  + s * ldb * N, &ione );
                normR = lapackf77_zlange( "M", &M, &N, h_X  + s * ldb * N,       &ldb, work );
                normX = lapackf77_zlange( "M", &M, &N, h_Bcublas  + s * ldb * N, &ldb, work );
                double err = normR/(normX*normA);

                if ( isnan(err) || isinf(err) ) {
                    printf("error for matrix %d cublas_error = %7.2f where normR=%7.2f normX=%7.2f and normA=%7.2f\n",
                           s, err, normR, normX, normA);
                    cublas_error = err;
                    break;
                }
                cublas_error = max( err, cublas_error );
            }
            #endif
            bool okay = (magma_error < tol && cublas_error < tol);
            status += ! okay;

            if ( opts.lapack ) {
                // check lapack
                // this verifies that the matrix wasn't so bad that it couldn't be solved accurately.
                lapack_error = 0.0;
                memcpy( h_X, h_Blapack, sizeB*sizeof(magmaDoubleComplex) );
                for (int s=0; s < batchCount; s++) {
                    blasf77_ztrmm(
                        lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                        lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                        &M, &N, &alpha2,
                        h_A + s * lda * Ak, &lda,
                        h_X + s * ldb * N,  &ldb );
    
                    blasf77_zaxpy( &NN, &c_neg_one, h_B + s * ldb * N, &ione, h_X + s * ldb * N, &ione );
                    normR = lapackf77_zlange( "M", &M, &N, h_X + s * ldb * N,       &ldb, work );
                    normX = lapackf77_zlange( "M", &M, &N, h_Blapack + s * ldb * N, &ldb, work );
                    double err = normR/(normX*normA);
                    lapack_error = max( err, lapack_error );
                }

                printf("%10d %5d %5d    %7.2f (%7.2f)     %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e   %s\n",
                        (int)batchCount, (int) M, (int) N,
                        magma_perf,  1000.*magma_time,
                        cublas_perf, 1000.*cublas_time,
                        cpu_perf,    1000.*cpu_time,
                        magma_error, cublas_error, lapack_error,
                        (okay ? "ok" : "failed"));
            }
            else {
                printf("%10d %5d %5d    %7.2f (%7.2f)     %7.2f (%7.2f)     ---   (  ---  )   %8.2e   %8.2e     ---      %s\n",
                        (int)batchCount, (int) M, (int) N,
                        magma_perf,  1000.*magma_time,
                        cublas_perf, 1000.*cublas_time,
                        magma_error, cublas_error,
                        (okay ? "ok" : "failed"));
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( h_X );
            TESTING_FREE_CPU( h_Blapack );
            TESTING_FREE_CPU( h_Bcublas );
            TESTING_FREE_CPU( h_Bmagma  );
            TESTING_FREE_CPU( ipiv );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            TESTING_FREE_DEV( d_A_array );
            TESTING_FREE_DEV( d_B_array );

            TESTING_FREE_DEV( dW1_displ );
            TESTING_FREE_DEV( dW2_displ );
            TESTING_FREE_DEV( dW3_displ );
            TESTING_FREE_DEV( dW4_displ );

            TESTING_FREE_DEV( dinvA );
            TESTING_FREE_DEV( dwork );
            TESTING_FREE_DEV( dwork_array );
            TESTING_FREE_DEV( dinvA_array );
            
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
