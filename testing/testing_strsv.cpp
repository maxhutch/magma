/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_ztrsv.cpp normal z -> s, Mon May  2 23:31:06 2016
       @author Chongxiao Cao
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

#define h_A(i,j) (h_A + (i) + (j)*lda)

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing strsm
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, cublas_perf, cublas_time, cpu_perf=0, cpu_time=0;
    float          cublas_error, normA, normx, normr, work[1];
    magma_int_t i, j, N, info;
    magma_int_t sizeA;
    magma_int_t lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;

    float *h_A, *h_b, *h_x, *h_xcublas;
    magmaFloat_ptr d_A, d_x;
    float c_neg_one = MAGMA_S_NEG_ONE;
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("%% uplo = %s, transA = %s, diag = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%%   N  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)   CUBLAS error\n");
    printf("%%===========================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            gflops = FLOPS_STRSM(opts.side, N, 1) / 1e9;
            lda    = N;
            ldda   = magma_roundup( lda, opts.align );  // multiple of 32 by default
            sizeA  = lda*N;
            
            TESTING_MALLOC_CPU( ipiv,      magma_int_t,        N     );
            TESTING_MALLOC_CPU( h_A,       float, lda*N );
            TESTING_MALLOC_CPU( h_b,       float, N     );
            TESTING_MALLOC_CPU( h_x,       float, N     );
            TESTING_MALLOC_CPU( h_xcublas, float, N     );
            
            TESTING_MALLOC_DEV( d_A, float, ldda*N );
            TESTING_MALLOC_DEV( d_x, float, N      );
            
            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_sgetrf( &N, &N, h_A, &lda, ipiv, &info );
            for( j = 0; j < N; ++j ) {
                for( i = 0; i < j; ++i ) {
                    *h_A(i,j) = *h_A(j,i);
                }
            }
            
            lapackf77_slarnv( &ione, ISEED, &N, h_b );
            blasf77_scopy( &N, h_b, &ione, h_x, &ione );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_ssetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            magma_ssetvector( N, h_x, 1, d_x, 1, opts.queue );
            
            cublas_time = magma_sync_wtime( opts.queue );
            #ifdef HAVE_CUBLAS
                cublasStrsv( opts.handle, cublas_uplo_const(opts.uplo),
                             cublas_trans_const(opts.transA), cublas_diag_const(opts.diag),
                             N,
                             d_A, ldda,
                             d_x, 1 );
            #else
                magma_strsv( opts.uplo, opts.transA, opts.diag,
                             N,
                             d_A, 0, ldda,
                             d_x, 0, 1, opts.queue );
            #endif
            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_sgetvector( N, d_x, 1, h_xcublas, 1, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_strsv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &N,
                               h_A, &lda,
                               h_x, &ione );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            // ||b - Ax|| / (||A||*||x||)
            // error for CUBLAS
            normA = lapackf77_slange( "F", &N, &N, h_A, &lda, work );
            
            normx = lapackf77_slange( "F", &N, &ione, h_xcublas, &ione, work );
            blasf77_strmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                           &N,
                           h_A, &lda,
                           h_xcublas, &ione );
            blasf77_saxpy( &N, &c_neg_one, h_b, &ione, h_xcublas, &ione );
            normr = lapackf77_slange( "F", &N, &ione, h_xcublas, &N, work );
            cublas_error = normr / (normA*normx);

            if ( opts.lapack ) {
                printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (int) N,
                        cublas_perf, 1000.*cublas_time,
                        cpu_perf,    1000.*cpu_time,
                        cublas_error, (cublas_error < tol ? "ok" : "failed"));
                status += ! (cublas_error < tol);
            }
            else {
                printf("%5d   %7.2f (%7.2f)     ---  (  ---  )   %8.2e   %s\n",
                        (int) N,
                        cublas_perf, 1000.*cublas_time,
                        cublas_error, (cublas_error < tol ? "ok" : "failed"));
                status += ! (cublas_error < tol);
            }
            
            TESTING_FREE_CPU( ipiv );
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_b  );
            TESTING_FREE_CPU( h_x  );
            TESTING_FREE_CPU( h_xcublas );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_x );
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
