/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated ds Tue Dec 17 13:18:56 2013
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

#define PRECISION_d

int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t   gflopsF, gflopsS, gpu_perf, gpu_time;
    real_Double_t   gpu_perfdf, gpu_perfds;
    real_Double_t   gpu_perfsf, gpu_perfss;
    double          Rnorm, Anorm;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_B, *h_X;
    double *d_A, *d_B, *d_X, *d_WORKD;
    float  *d_As, *d_Bs, *d_WORKS;
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
            
            gflopsF = FLOPS_DGETRF( N, N ) / 1e9;
            gflopsS = gflopsF + FLOPS_DGETRS( N, nrhs ) / 1e9;

            TESTING_MALLOC_CPU( h_A,     double, lda*N    );
            TESTING_MALLOC_CPU( h_B,     double, ldb*nrhs );
            TESTING_MALLOC_CPU( h_X,     double, ldx*nrhs );
            TESTING_MALLOC_CPU( h_ipiv,  magma_int_t,        N        );
            TESTING_MALLOC_CPU( h_workd, double,             N        );
            
            TESTING_MALLOC_DEV( d_A,     double, ldda*N        );
            TESTING_MALLOC_DEV( d_B,     double, lddb*nrhs     );
            TESTING_MALLOC_DEV( d_X,     double, lddx*nrhs     );
            TESTING_MALLOC_DEV( d_ipiv,  magma_int_t,        N             );
            TESTING_MALLOC_DEV( d_WORKS, float,  ldda*(N+nrhs) );
            TESTING_MALLOC_DEV( d_WORKD, double, N*nrhs        );
            
            /* Initialize matrices */
            size = lda * N;
            lapackf77_dlarnv( &ione, ISEED, &size, h_A );
            size = ldb * nrhs;
            lapackf77_dlarnv( &ione, ISEED, &size, h_B );
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &nrhs, h_B, &ldb, h_X, &ldx);
            
            magma_dsetmatrix( N, N,    h_A, lda, d_A, ldda );
            magma_dsetmatrix( N, nrhs, h_B, ldb, d_B, lddb );
            
            //=====================================================================
            //              MIXED - GPU
            //=====================================================================
            gpu_time = magma_wtime();
            magma_dsgesv_gpu( opts.transA, N, nrhs,
                              d_A, ldda, h_ipiv, d_ipiv,
                              d_B, lddb, d_X, lddx,
                              d_WORKD, d_WORKS, &gesv_iter, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflopsS / gpu_time;
            if (info != 0)
                printf("magma_dsgesv returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //              ERROR DP vs MIXED  - GPU
            //=====================================================================
            magma_dgetmatrix( N, nrhs, d_X, lddx, h_X, ldx );
            
            Anorm = lapackf77_dlange("I", &N, &N, h_A, &lda, h_workd);
            blasf77_dgemm( &opts.transA, MagmaNoTransStr,
                           &N, &nrhs, &N,
                           &c_one,     h_A, &lda,
                                       h_X, &ldx,
                           &c_neg_one, h_B, &ldb);
            Rnorm = lapackf77_dlange("I", &N, &nrhs, h_B, &ldb, h_workd);
            
            //=====================================================================
            //                 Double Precision Factor
            //=====================================================================
            magma_dsetmatrix( N, N, h_A, lda, d_A, ldda );
            
            gpu_time = magma_wtime();
            magma_dgetrf_gpu(N, N, d_A, ldda, h_ipiv, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfdf = gflopsF / gpu_time;
            if (info != 0)
                printf("magma_dgetrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Double Precision Solve
            //=====================================================================
            magma_dsetmatrix( N, N,    h_A, lda, d_A, ldda );
            magma_dsetmatrix( N, nrhs, h_B, ldb, d_B, lddb );
            
            gpu_time = magma_wtime();
            magma_dgetrf_gpu(N, N, d_A, ldda, h_ipiv, &info);
            magma_dgetrs_gpu( opts.transA, N, nrhs, d_A, ldda, h_ipiv, d_B, lddb, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfds = gflopsS / gpu_time;
            if (info != 0)
                printf("magma_dgetrs returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Single Precision Factor
            //=====================================================================
            d_As = d_WORKS;
            d_Bs = d_WORKS + ldda*N;
            magma_dsetmatrix( N, N,    h_A, lda,  d_A,  ldda );
            magma_dsetmatrix( N, nrhs, h_B, ldb,  d_B,  lddb );
            magmablas_dlag2s( N, N,    d_A, ldda, d_As, ldda, &info );
            magmablas_dlag2s( N, nrhs, d_B, lddb, d_Bs, lddb, &info );
            
            gpu_time = magma_wtime();
            magma_sgetrf_gpu(N, N, d_As, ldda, h_ipiv, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfsf = gflopsF / gpu_time;
            if (info != 0)
                printf("magma_sgetrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Single Precision Solve
            //=====================================================================
            magmablas_dlag2s(N, N,    d_A, ldda, d_As, ldda, &info );
            magmablas_dlag2s(N, nrhs, d_B, lddb, d_Bs, lddb, &info );
            
            gpu_time = magma_wtime();
            magma_sgetrf_gpu( N, N,    d_As, ldda, h_ipiv, &info);
            magma_sgetrs_gpu( opts.transA, N, nrhs, d_As, ldda, h_ipiv,
                              d_Bs, lddb, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfss = gflopsS / gpu_time;
            if (info != 0)
                printf("magma_sgetrs returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            printf("%5d %5d   %7.2f   %7.2f   %7.2f   %7.2f   %7.2f   %8.2e   %3d\n",
                   (int) N, (int) nrhs,
                   gpu_perfdf, gpu_perfds, gpu_perfsf, gpu_perfss, gpu_perf,
                   Rnorm/(N*Anorm), (int) gesv_iter );
            
            TESTING_FREE_CPU( h_A     );
            TESTING_FREE_CPU( h_B     );
            TESTING_FREE_CPU( h_X     );
            TESTING_FREE_CPU( h_ipiv  );
            TESTING_FREE_CPU( h_workd );
            
            TESTING_FREE_DEV( d_A     );
            TESTING_FREE_DEV( d_B     );
            TESTING_FREE_DEV( d_X     );
            TESTING_FREE_DEV( d_ipiv  );
            TESTING_FREE_DEV( d_WORKS );
            TESTING_FREE_DEV( d_WORKD );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
