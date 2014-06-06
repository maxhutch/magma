/*
    -- MAGMA (version 1.5.0-beta1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date April 2014

       @generated from testing_zgemm.cpp normal z -> s, Fri Apr 25 15:06:03 2014
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
   -- Testing sgemm
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
    
    float *h_A, *h_B, *h_C, *h_Cmagma, *h_Ccublas;
    float *d_A, *d_B, *d_C;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_S_MAKE( -0.48,  0.38 );
    
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
            gflops = FLOPS_SGEMM( M, N, K ) / 1e9;

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
            
            TESTING_MALLOC_CPU( h_A,       float, lda*An );
            TESTING_MALLOC_CPU( h_B,       float, ldb*Bn );
            TESTING_MALLOC_CPU( h_C,       float, ldc*N  );
            TESTING_MALLOC_CPU( h_Cmagma,  float, ldc*N  );
            TESTING_MALLOC_CPU( h_Ccublas, float, ldc*N  );
            
            TESTING_MALLOC_DEV( d_A, float, ldda*An );
            TESTING_MALLOC_DEV( d_B, float, lddb*Bn );
            TESTING_MALLOC_DEV( d_C, float, lddc*N  );
            
            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_slarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_slarnv( &ione, ISEED, &sizeC, h_C );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_ssetmatrix( Am, An, h_A, lda, d_A, ldda );
            magma_ssetmatrix( Bm, Bn, h_B, ldb, d_B, lddb );
            magma_ssetmatrix( M, N, h_C, ldc, d_C, lddc );
            
            magma_time = magma_sync_wtime( NULL );
            magmablas_sgemm( opts.transA, opts.transB, M, N, K,
                             alpha, d_A, ldda,
                                    d_B, lddb,
                             beta,  d_C, lddc );
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_sgetmatrix( M, N, d_C, lddc, h_Cmagma, ldc );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_ssetmatrix( M, N, h_C, ldc, d_C, lddc );
            
            cublas_time = magma_sync_wtime( NULL );
            cublasSgemm( handle, cublas_trans_const(opts.transA), cublas_trans_const(opts.transB), M, N, K,
                         &alpha, d_A, ldda,
                                 d_B, lddb,
                         &beta,  d_C, lddc );
            cublas_time = magma_sync_wtime( NULL ) - cublas_time;
            cublas_perf = gflops / cublas_time;
            
            magma_sgetmatrix( M, N, d_C, lddc, h_Ccublas, ldc );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_sgemm( lapack_trans_const(opts.transA), lapack_trans_const(opts.transB), &M, &N, &K,
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
                Cnorm = lapackf77_slange( "M", &M, &N, h_C, &ldc, work );
                
                blasf77_saxpy( &sizeC, &c_neg_one, h_C, &ione, h_Cmagma, &ione );
                magma_error = lapackf77_slange( "M", &M, &N, h_Cmagma, &ldc, work ) / Cnorm;
                
                blasf77_saxpy( &sizeC, &c_neg_one, h_C, &ione, h_Ccublas, &ione );
                cublas_error = lapackf77_slange( "M", &M, &N, h_Ccublas, &ldc, work ) / Cnorm;
                
                printf("%5d %5d %5d   %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e     %8.2e\n",
                       (int) M, (int) N, (int) K,
                       magma_perf,  1000.*magma_time,
                       cublas_perf, 1000.*cublas_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, cublas_error );
            }
            else {
                // compute relative error for magma, relative to cublas
                Cnorm = lapackf77_slange( "M", &M, &N, h_Ccublas, &ldc, work );
                
                blasf77_saxpy( &sizeC, &c_neg_one, h_Ccublas, &ione, h_Cmagma, &ione );
                magma_error = lapackf77_slange( "M", &M, &N, h_Cmagma, &ldc, work ) / Cnorm;
                
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
