/*
    -- MAGMA (version 1.5.0-beta1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date April 2014

       @generated from testing_zgemm.cpp normal z -> d, Fri Apr 25 15:06:03 2014
       @author Mark Gates
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


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgemm
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf, cublas_time, cpu_perf, cpu_time;
    double          magma_error, cublas_error, Cnorm, work[1];
    magma_int_t M, N, K;
    magma_int_t Am, An, Bm, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    double *h_A, *h_B, *h_C, *h_Cmagma, *h_Ccublas;
    double *d_A, *d_B, *d_C;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double alpha = MAGMA_D_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "transA = %s, transB = %s\n",
           lapack_trans_const(opts.transA),
           lapack_trans_const(opts.transB) );
    printf("    M     N     K   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  CUBLAS error\n");
    printf("=========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_DGEMM( M, N, K ) / 1e9;

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
            
            ldda = ((lda+31)/32)*32;
            lddb = ((ldb+31)/32)*32;
            lddc = ((ldc+31)/32)*32;
            
            sizeA = lda*An;
            sizeB = ldb*Bn;
            sizeC = ldc*N;
            
            TESTING_MALLOC_CPU( h_A,       double, lda*An );
            TESTING_MALLOC_CPU( h_B,       double, ldb*Bn );
            TESTING_MALLOC_CPU( h_C,       double, ldc*N  );
            TESTING_MALLOC_CPU( h_Cmagma,  double, ldc*N  );
            TESTING_MALLOC_CPU( h_Ccublas, double, ldc*N  );
            
            TESTING_MALLOC_DEV( d_A, double, ldda*An );
            TESTING_MALLOC_DEV( d_B, double, lddb*Bn );
            TESTING_MALLOC_DEV( d_C, double, lddc*N  );
            
            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_dlarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_dlarnv( &ione, ISEED, &sizeC, h_C );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_dsetmatrix( Am, An, h_A, lda, d_A, ldda );
            magma_dsetmatrix( Bm, Bn, h_B, ldb, d_B, lddb );
            magma_dsetmatrix( M, N, h_C, ldc, d_C, lddc );
            
            magma_time = magma_sync_wtime( NULL );
            magmablas_dgemm( opts.transA, opts.transB, M, N, K,
                             alpha, d_A, ldda,
                                    d_B, lddb,
                             beta,  d_C, lddc );
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_dgetmatrix( M, N, d_C, lddc, h_Cmagma, ldc );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_dsetmatrix( M, N, h_C, ldc, d_C, lddc );
            
            cublas_time = magma_sync_wtime( NULL );
            cublasDgemm( handle, cublas_trans_const(opts.transA), cublas_trans_const(opts.transB), M, N, K,
                         &alpha, d_A, ldda,
                                 d_B, lddb,
                         &beta,  d_C, lddc );
            cublas_time = magma_sync_wtime( NULL ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_dgetmatrix( M, N, d_C, lddc, h_Ccublas, ldc );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_dgemm( lapack_trans_const(opts.transA), lapack_trans_const(opts.transB), &M, &N, &K,
                               &alpha, h_A, &lda,
                                       h_B, &ldb,
                               &beta,  h_C, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for both magma & cublas, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                Cnorm = lapackf77_dlange( "M", &M, &N, h_C, &ldc, work );
                
                blasf77_daxpy( &sizeC, &c_neg_one, h_C, &ione, h_Cmagma, &ione );
                magma_error = lapackf77_dlange( "M", &M, &N, h_Cmagma, &ldc, work ) / Cnorm;
                
                blasf77_daxpy( &sizeC, &c_neg_one, h_C, &ione, h_Ccublas, &ione );
                cublas_error = lapackf77_dlange( "M", &M, &N, h_Ccublas, &ldc, work ) / Cnorm;
                
                printf("%5d %5d %5d   %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e     %8.2e\n",
                       (int) M, (int) N, (int) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, cublas_error );
            }
            else {
                // compute relative error for magma, relative to cublas
                Cnorm = lapackf77_dlange( "M", &M, &N, h_Ccublas, &ldc, work );
                
                blasf77_daxpy( &sizeC, &c_neg_one, h_Ccublas, &ione, h_Cmagma, &ione );
                magma_error = lapackf77_dlange( "M", &M, &N, h_Cmagma, &ldc, work ) / Cnorm;
                
                printf("%5d %5d %5d   %7.2f (%7.2f)    %7.2f (%7.2f)     ---   (  ---  )    %8.2e     ---\n",
                       (int) M, (int) N, (int) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       magma_error );
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( h_C );
            TESTING_FREE_CPU( h_Cmagma  );
            TESTING_FREE_CPU( h_Ccublas );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            TESTING_FREE_DEV( d_C );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
