/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @precisions mixed zc -> ds
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_z

int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t   gflopsF, gflopsS, gpu_perf, gpu_time;
    real_Double_t   gpu_perfdf, gpu_perfds;
    real_Double_t   gpu_perfsf, gpu_perfss;
    double          Rnorm, Anorm;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex *h_A, *h_B, *h_X;
    cuDoubleComplex *d_A, *d_B, *d_X, *d_WORKD;
    cuFloatComplex  *d_As, *d_Bs, *d_WORKS;
    double          *h_workd;
    magma_int_t *h_ipiv, *d_ipiv;
    magma_int_t lda, ldb, ldx;
    magma_int_t ldda, lddb, lddx;
    magma_int_t N, nrhs, gesv_iter, info, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    printf("Epsilon(double): %8.6e\n"
           "Epsilon(single): %8.6e\n\n",
           lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon") );

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    nrhs = opts.nrhs;
    
    printf("    N  NRHS   DP-Factor  DP-Solve  SP-Factor  SP-Solve  MP-Solve  ||b-Ax||/(N*||A||)  Iter\n");
    printf("==========================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            ldb  = ldx = lda = N;
            ldda = ((N+31)/32)*32;
            lddb = lddx = ldda;
            
            gflopsF = FLOPS_ZGETRF( N, N ) / 1e9;
            gflopsS = gflopsF + FLOPS_ZGETRS( N, nrhs ) / 1e9;

            TESTING_MALLOC( h_A, cuDoubleComplex, lda*N    );
            TESTING_MALLOC( h_B, cuDoubleComplex, ldb*nrhs );
            TESTING_MALLOC( h_X, cuDoubleComplex, ldx*nrhs );
            TESTING_MALLOC( h_ipiv, magma_int_t,    N        );
            TESTING_MALLOC( h_workd, double, N );
            
            TESTING_DEVALLOC( d_A,     cuDoubleComplex, ldda*N        );
            TESTING_DEVALLOC( d_B,     cuDoubleComplex, lddb*nrhs     );
            TESTING_DEVALLOC( d_X,     cuDoubleComplex, lddx*nrhs     );
            TESTING_DEVALLOC( d_ipiv,  magma_int_t,     N             );
            TESTING_DEVALLOC( d_WORKS, cuFloatComplex,  ldda*(N+nrhs) );
            TESTING_DEVALLOC( d_WORKD, cuDoubleComplex, N*nrhs        );
            
            /* Initialize matrices */
            size = lda * N;
            lapackf77_zlarnv( &ione, ISEED, &size, h_A );
            size = ldb * nrhs;
            lapackf77_zlarnv( &ione, ISEED, &size, h_B );
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &nrhs, h_B, &ldb, h_X, &ldx);
            
            magma_zsetmatrix( N, N,    h_A, lda, d_A, ldda );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, lddb );
            
            //=====================================================================
            //              MIXED - GPU
            //=====================================================================
            gpu_time = magma_wtime();
            magma_zcgesv_gpu( opts.transA, N, nrhs,
                              d_A, ldda, h_ipiv, d_ipiv,
                              d_B, lddb, d_X, lddx,
                              d_WORKD, d_WORKS, &gesv_iter, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflopsS / gpu_time;
            if (info != 0)
                printf("magma_zcgesv returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //              ERROR DP vs MIXED  - GPU
            //=====================================================================
            magma_zgetmatrix( N, nrhs, d_X, lddx, h_X, ldx );
            
            Anorm = lapackf77_zlange("I", &N, &N, h_A, &lda, h_workd);
            blasf77_zgemm( &opts.transA, MagmaNoTransStr,
                           &N, &nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldx,
                           &c_neg_one, h_B, &ldb);
            Rnorm = lapackf77_zlange("I", &N, &nrhs, h_B, &ldb, h_workd);
            
            //=====================================================================
            //                 Double Precision Factor
            //=====================================================================
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda );
            
            gpu_time = magma_wtime();
            magma_zgetrf_gpu(N, N, d_A, ldda, h_ipiv, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfdf = gflopsF / gpu_time;
            if (info != 0)
                printf("magma_zgetrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Double Precision Solve
            //=====================================================================
            magma_zsetmatrix( N, N,    h_A, lda, d_A, ldda );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, lddb );
            
            gpu_time = magma_wtime();
            magma_zgetrf_gpu(N, N, d_A, ldda, h_ipiv, &info);
            magma_zgetrs_gpu( opts.transA, N, nrhs, d_A, ldda, h_ipiv, d_B, lddb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfds = gflopsS / gpu_time;
            if (info != 0)
                printf("magma_zgetrs returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Single Precision Factor
            //=====================================================================
            d_As = d_WORKS;
            d_Bs = d_WORKS + ldda*N;
            magma_zsetmatrix( N, N,    h_A, lda,  d_A,  ldda );
            magma_zsetmatrix( N, nrhs, h_B, ldb,  d_B,  lddb );
            magmablas_zlag2c( N, N,    d_A, ldda, d_As, ldda, &info );
            magmablas_zlag2c( N, nrhs, d_B, lddb, d_Bs, lddb, &info );
            
            gpu_time = magma_wtime();
            magma_cgetrf_gpu(N, N, d_As, ldda, h_ipiv, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfsf = gflopsF / gpu_time;
            if (info != 0)
                printf("magma_cgetrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Single Precision Solve
            //=====================================================================
            magmablas_zlag2c(N, N,    d_A, ldda, d_As, ldda, &info );
            magmablas_zlag2c(N, nrhs, d_B, lddb, d_Bs, lddb, &info );
            
            gpu_time = magma_wtime();
            magma_cgetrf_gpu( N, N,    d_As, ldda, h_ipiv, &info);
            magma_cgetrs_gpu( opts.transA, N, nrhs, d_As, ldda, h_ipiv,
                              d_Bs, lddb, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfss = gflopsS / gpu_time;
            if (info != 0)
                printf("magma_cgetrs returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            printf("%5d %5d   %7.2f   %7.2f   %7.2f   %7.2f   %7.2f   %8.2e   %3d\n",
                   (int) N, (int) nrhs,
                   gpu_perfdf, gpu_perfds, gpu_perfsf, gpu_perfss, gpu_perf,
                   Rnorm/(N*Anorm), (int) gesv_iter );
            
            TESTING_FREE( h_A );
            TESTING_FREE( h_B );
            TESTING_FREE( h_X );
            TESTING_FREE( h_ipiv );
            TESTING_FREE( h_workd );
            
            TESTING_DEVFREE( d_A );
            TESTING_DEVFREE( d_B );
            TESTING_DEVFREE( d_X );
            TESTING_DEVFREE( d_ipiv );
            TESTING_DEVFREE( d_WORKS );
            TESTING_DEVFREE( d_WORKD );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
