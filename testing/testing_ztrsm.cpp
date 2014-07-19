/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @precisions normal z -> c d s
       @author Chongxiao Cao
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define h_A(i,j) (h_A + (i) + (j)*lda)

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrsm
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time=0, cublas_perf, cublas_time, cpu_perf=0, cpu_time=0;
    double          magma_error, cublas_error, work[1];
    magma_int_t M, N, info;
    magma_int_t Ak;
    magma_int_t sizeA, sizeB;
    magma_int_t lda, ldb, ldda, lddb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;

    magmaDoubleComplex *h_A, *h_B, *h_Bcublas, *h_Bmagma, *h_B1, *h_X1, *h_X2;
    magmaDoubleComplex *d_A, *d_B;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("side = %s, uplo = %s, transA = %s, diag = %s \n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("    M     N  MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  CUBLAS error\n");
    printf("==================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_ZTRSM(opts.side, M, N) / 1e9;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                Ak = M;
            } else {
                lda = N;
                Ak = N;
            }
            
            ldb = M;
            
            ldda = ((lda+31)/32)*32;
            lddb = ((ldb+31)/32)*32;
            
            sizeA = lda*Ak;
            sizeB = ldb*N;
            
            TESTING_MALLOC_CPU( h_A,       magmaDoubleComplex, lda*Ak  );
            TESTING_MALLOC_CPU( h_B,       magmaDoubleComplex, ldb*N   );
            TESTING_MALLOC_CPU( h_B1,      magmaDoubleComplex, ldb*N   );
            TESTING_MALLOC_CPU( h_X1,      magmaDoubleComplex, ldb*N   );
            TESTING_MALLOC_CPU( h_X2,      magmaDoubleComplex, ldb*N   );
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
            memcpy(h_B1, h_B, sizeB*sizeof(magmaDoubleComplex));
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( Ak, Ak, h_A, lda, d_A, ldda );
            magma_zsetmatrix( M, N, h_B, ldb, d_B, lddb );
            
            magma_time = magma_sync_wtime( NULL );
            magmablas_ztrsm( opts.side, opts.uplo, opts.transA, opts.diag, 
                             M, N,
                             alpha, d_A, ldda,
                                    d_B, lddb );
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_zgetmatrix( M, N, d_B, lddb, h_Bmagma, ldb );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_zsetmatrix( M, N, h_B, ldb, d_B, lddb );
            
            cublas_time = magma_sync_wtime( NULL );
            cublasZtrsm( handle, cublas_side_const(opts.side), cublas_uplo_const(opts.uplo),
                         cublas_trans_const(opts.transA), cublas_diag_const(opts.diag),
                         M, N, 
                         &alpha, d_A, ldda,
                                 d_B, lddb );
            cublas_time = magma_sync_wtime( NULL ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_zgetmatrix( M, N, d_B, lddb, h_Bcublas, ldb );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ztrsm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag), 
                               &M, &N,
                               &alpha, h_A, &lda,
                                       h_B, &ldb );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            // ||b - Ax|| / (||A||*||x||)
            memcpy(h_X1, h_Bmagma, sizeB*sizeof(magmaDoubleComplex));
            
            magmaDoubleComplex alpha2 = MAGMA_Z_DIV(  c_one, alpha );
            blasf77_ztrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag), 
                            &M, &N,
                            &alpha2, h_A, &lda,
                            h_X1, &ldb );

            blasf77_zaxpy( &sizeB, &c_neg_one, h_B1, &ione, h_X1, &ione );
            double norm1 =  lapackf77_zlange( "M", &M, &N, h_X1, &ldb, work );
            double normx =  lapackf77_zlange( "M", &M, &N, h_Bmagma, &ldb, work );
            double normA =  lapackf77_zlange( "M", &Ak, &Ak, h_A, &lda, work );

            magma_error = norm1/(normx*normA);

            memcpy(h_X2, h_Bcublas, sizeB*sizeof(magmaDoubleComplex));
            blasf77_ztrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag), 
                            &M, &N,
                            &alpha2, h_A, &lda,
                            h_X2, &ldb );

            blasf77_zaxpy( &sizeB, &c_neg_one, h_B1, &ione, h_X2, &ione );
            norm1 =  lapackf77_zlange( "M", &M, &N, h_X2, &ldb, work );
            normx =  lapackf77_zlange( "M", &M, &N, h_Bcublas, &ldb, work );
            normA =  lapackf77_zlange( "M", &Ak, &Ak, h_A, &lda, work );
            
            cublas_error = norm1/(normx*normA);
            
            if ( opts.lapack ) {
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e     %8.2e   %s\n",
                        (int) M, (int) N,
                        magma_perf,  1000.*magma_time,
                        cublas_perf, 1000.*cublas_time,
                        cpu_perf,    1000.*cpu_time,
                        magma_error, cublas_error,
                        (magma_error < tol && cublas_error < tol? "ok" : "failed"));
                status += ! (magma_error < tol && cublas_error < tol);
            }
            else {
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)     ---   (  ---  )   %8.2e     %8.2e   %s\n",
                        (int) M, (int) N,
                        magma_perf,  1000.*magma_time,
                        cublas_perf, 1000.*cublas_time,
                        magma_error, cublas_error,
                        (magma_error < tol && cublas_error < tol? "ok" : "failed"));
                status += ! (magma_error < tol && cublas_error < tol);
            }
            
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_B  );
            TESTING_FREE_CPU( h_B1 );
            TESTING_FREE_CPU( h_X1 );
            TESTING_FREE_CPU( h_X2 );
            TESTING_FREE_CPU( h_Bcublas );
            TESTING_FREE_CPU( h_Bmagma  );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
