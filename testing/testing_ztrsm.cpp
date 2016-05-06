/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> c d s
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
#include "magma_operators.h"  // for MAGMA_Z_DIV
#include "testings.h"

#define h_A(i,j) (h_A + (i) + (j)*lda)

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrsm
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf=0, magma_time=0, cublas_perf, cublas_time, cpu_perf=0, cpu_time=0;
    double          magma_error=0, cublas_error, lapack_error, work[1];
    magma_int_t M, N, info;
    magma_int_t Ak;
    magma_int_t sizeA, sizeB;
    magma_int_t lda, ldb, ldda, lddb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;
    
    magmaDoubleComplex *h_A, *h_B, *h_Bcublas, *h_Bmagma, *h_Blapack, *h_X;
    magmaDoubleComplex_ptr d_A, d_B;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% side = %s, uplo = %s, transA = %s, diag = %s, ngpu = %d\n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag), int(abs_ngpu) );
    
    printf("%%   M     N  MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)      MAGMA     CUBLAS   LAPACK error\n");
    printf("%%============================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_ZTRSM(opts.side, M, N) / 1e9;

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
            
            sizeA = lda*Ak;
            sizeB = ldb*N;
            
            TESTING_MALLOC_CPU( h_A,       magmaDoubleComplex, lda*Ak  );
            TESTING_MALLOC_CPU( h_B,       magmaDoubleComplex, ldb*N   );
            TESTING_MALLOC_CPU( h_X,       magmaDoubleComplex, ldb*N   );
            TESTING_MALLOC_CPU( h_Blapack, magmaDoubleComplex, ldb*N   );
            TESTING_MALLOC_CPU( h_Bcublas, magmaDoubleComplex, ldb*N   );
            TESTING_MALLOC_CPU( h_Bmagma,  magmaDoubleComplex, ldb*N   );
            TESTING_MALLOC_CPU( ipiv,      magma_int_t,        Ak      );
            
            TESTING_MALLOC_DEV( d_A,       magmaDoubleComplex, ldda*Ak );
            TESTING_MALLOC_DEV( d_B,       magmaDoubleComplex, lddb*N  );
            
            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zgetrf( &Ak, &Ak, h_A, &lda, ipiv, &info );
            for( int j = 0; j < Ak; ++j ) {
                for( int i = 0; i < j; ++i ) {
                    *h_A(i,j) = *h_A(j,i);
                }
            }
            
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            memcpy( h_Blapack, h_B, sizeB*sizeof(magmaDoubleComplex) );
            magma_zsetmatrix( Ak, Ak, h_A, lda, d_A, ldda, opts.queue );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            #if defined(HAVE_CUBLAS)
                magma_zsetmatrix( M, N, h_B, ldb, d_B, lddb, opts.queue );
                
                magma_time = magma_sync_wtime( opts.queue );
                if (opts.ngpu == 1) {
                    magmablas_ztrsm( opts.side, opts.uplo, opts.transA, opts.diag,
                                     M, N,
                                     alpha, d_A, ldda,
                                            d_B, lddb, opts.queue );
                }
                else {
                    magma_ztrsm_m( abs_ngpu, opts.side, opts.uplo, opts.transA, opts.diag,
                                   M, N,
                                   alpha, d_A, ldda,
                                          d_B, lddb );
                }
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_zgetmatrix( M, N, d_B, lddb, h_Bmagma, ldb, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_zsetmatrix( M, N, h_B, ldb, d_B, lddb, opts.queue );
            
            cublas_time = magma_sync_wtime( opts.queue );
            #if defined(HAVE_CUBLAS)
                // opts.handle also uses opts.queue 
                cublasZtrsm( opts.handle,
                             cublas_side_const(opts.side), cublas_uplo_const(opts.uplo),
                             cublas_trans_const(opts.transA), cublas_diag_const(opts.diag),
                             M, N,
                             &alpha, d_A, ldda,
                                     d_B, lddb );
            #elif defined(HAVE_clBLAS)
                clblasZtrsm( clblasColumnMajor,
                             clblas_side_const(opts.side), clblas_uplo_const(opts.uplo),
                             clblas_trans_const(opts.transA), clblas_diag_const(opts.diag),
                             M, N,
                             alpha, d_A, 0, ldda,
                                    d_B, 0, lddb,
                             1, &opts.queue, 0, NULL, NULL );
            #endif
            cublas_time = magma_sync_wtime( opts.queue ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetmatrix( M, N, d_B, lddb, h_Bcublas, ldb, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ztrsm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &alpha, h_A, &lda,
                                       h_Blapack, &ldb );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            // ||b - 1/alpha*A*x|| / (||A||*||x||)
            magmaDoubleComplex inv_alpha = MAGMA_Z_DIV( c_one, alpha );
            double normR, normX, normA;
            normA = lapackf77_zlange( "M", &Ak, &Ak, h_A, &lda, work );
            
            #if defined(HAVE_CUBLAS)
                // check magma
                memcpy( h_X, h_Bmagma, sizeB*sizeof(magmaDoubleComplex) );
                blasf77_ztrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &inv_alpha, h_A, &lda,
                                           h_X, &ldb );
                
                blasf77_zaxpy( &sizeB, &c_neg_one, h_B, &ione, h_X, &ione );
                normR = lapackf77_zlange( "M", &M, &N, h_X,      &ldb, work );
                normX = lapackf77_zlange( "M", &M, &N, h_Bmagma, &ldb, work );
                magma_error = normR/(normX*normA);
            #endif

            // check cublas
            memcpy( h_X, h_Bcublas, sizeB*sizeof(magmaDoubleComplex) );
            blasf77_ztrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                           &M, &N,
                           &inv_alpha, h_A, &lda,
                                       h_X, &ldb );

            blasf77_zaxpy( &sizeB, &c_neg_one, h_B, &ione, h_X, &ione );
            normR = lapackf77_zlange( "M", &M, &N, h_X,       &ldb, work );
            normX = lapackf77_zlange( "M", &M, &N, h_Bcublas, &ldb, work );
            cublas_error = normR/(normX*normA);

            if ( opts.lapack ) {
                // check lapack
                // this verifies that the matrix wasn't so bad that it couldn't be solved accurately.
                memcpy( h_X, h_Blapack, sizeB*sizeof(magmaDoubleComplex) );
                blasf77_ztrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &inv_alpha, h_A, &lda,
                                           h_X, &ldb );
    
                blasf77_zaxpy( &sizeB, &c_neg_one, h_B, &ione, h_X, &ione );
                normR = lapackf77_zlange( "M", &M, &N, h_X,       &ldb, work );
                normX = lapackf77_zlange( "M", &M, &N, h_Blapack, &ldb, work );
                lapack_error = normR/(normX*normA);
                
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e   %s\n",
                        (int) M, (int) N,
                        magma_perf,  1000.*magma_time,
                        cublas_perf, 1000.*cublas_time,
                        cpu_perf,    1000.*cpu_time,
                        magma_error, cublas_error, lapack_error,
                        (magma_error < tol && cublas_error < tol? "ok" : "failed"));
                status += ! (magma_error < tol && cublas_error < tol);
            }
            else {
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)     ---   (  ---  )   %8.2e   %8.2e     ---      %s\n",
                        (int) M, (int) N,
                        magma_perf,  1000.*magma_time,
                        cublas_perf, 1000.*cublas_time,
                        magma_error, cublas_error,
                        (magma_error < tol && cublas_error < tol ? "ok" : "failed"));
                status += ! (magma_error < tol && cublas_error < tol);
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
