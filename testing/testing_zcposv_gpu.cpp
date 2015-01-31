/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions mixed zc -> ds
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_z

int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t   gflopsF, gflopsS, gpu_perf, gpu_time /*cpu_perf, cpu_time*/;
    real_Double_t   gpu_perfdf, gpu_perfds;
    real_Double_t   gpu_perfsf, gpu_perfss;
    double          error, Rnorm, Anorm;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_X;
    magmaDoubleComplex_ptr d_A, d_B, d_X, d_workd;
    magmaFloatComplex  *d_As, *d_Bs, *d_works;
    double          *h_workd;
    magma_int_t lda, ldb, ldx;
    magma_int_t N, nrhs, posv_iter, info, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    printf("Epsilon(double): %8.6e\n"
           "Epsilon(single): %8.6e\n\n",
           lapackf77_dlamch("Epsilon"), lapackf77_slamch("Epsilon") );
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    nrhs = opts.nrhs;
    
    printf("using: uplo = %s\n",
           lapack_uplo_const(opts.uplo));

    printf("    N NRHS   DP-Factor  DP-Solve  SP-Factor  SP-Solve  MP-Solve  Iter   |b-Ax|/|A|\n");
    printf("=====================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            ldb = ldx = lda = N;
            gflopsF = FLOPS_ZPOTRF( N ) / 1e9;
            gflopsS = gflopsF + FLOPS_ZPOTRS( N, nrhs ) / 1e9;
            
            TESTING_MALLOC_CPU( h_A,     magmaDoubleComplex, lda*N    );
            TESTING_MALLOC_CPU( h_B,     magmaDoubleComplex, ldb*nrhs );
            TESTING_MALLOC_CPU( h_X,     magmaDoubleComplex, ldx*nrhs );
            TESTING_MALLOC_CPU( h_workd, double,             N        );
            
            TESTING_MALLOC_DEV( d_A,     magmaDoubleComplex, lda*N        );
            TESTING_MALLOC_DEV( d_B,     magmaDoubleComplex, ldb*nrhs     );
            TESTING_MALLOC_DEV( d_X,     magmaDoubleComplex, ldx*nrhs     );
            TESTING_MALLOC_DEV( d_works, magmaFloatComplex,  lda*(N+nrhs) );
            TESTING_MALLOC_DEV( d_workd, magmaDoubleComplex, N*nrhs       );
            
            /* Initialize the matrix */
            size = lda * N ;
            lapackf77_zlarnv( &ione, ISEED, &size, h_A );
            magma_zmake_hpd( N, h_A, lda );
            
            size = ldb * nrhs ;
            lapackf77_zlarnv( &ione, ISEED, &size, h_B );
            
            magma_zsetmatrix( N, N,    h_A, lda, d_A, lda );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, ldb );
            
            //=====================================================================
            //              Mixed Precision Iterative Refinement - GPU
            //=====================================================================
            gpu_time = magma_wtime();
            magma_zcposv_gpu(opts.uplo, N, nrhs, d_A, lda, d_B, ldb, d_X, ldx,
                             d_workd, d_works, &posv_iter, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflopsS / gpu_time;
            if (info != 0)
                printf("magma_zcposv returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Error Computation
            //=====================================================================
            magma_zgetmatrix( N, nrhs, d_X, ldx, h_X, ldx ) ;
            
            Anorm = lapackf77_zlanhe( "I", lapack_uplo_const(opts.uplo), &N, h_A, &N, h_workd);
            blasf77_zhemm( "L", lapack_uplo_const(opts.uplo), &N, &nrhs,
                           &c_one,     h_A, &lda,
                                       h_X, &ldx,
                           &c_neg_one, h_B, &ldb);
            Rnorm = lapackf77_zlange( "I", &N, &nrhs, h_B, &ldb, h_workd);
            error = Rnorm / Anorm;
            
            //=====================================================================
            //                 Double Precision Factor
            //=====================================================================
            magma_zsetmatrix( N, N, h_A, lda, d_A, lda );
            
            gpu_time = magma_wtime();
            magma_zpotrf_gpu(opts.uplo, N, d_A, lda, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfdf = gflopsF / gpu_time;
            if (info != 0)
                printf("magma_zpotrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Double Precision Solve
            //=====================================================================
            magma_zsetmatrix( N, N,    h_A, lda, d_A, lda );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, ldb );
            
            gpu_time = magma_wtime();
            magma_zpotrf_gpu(opts.uplo, N, d_A, lda, &info);
            magma_zpotrs_gpu(opts.uplo, N, nrhs, d_A, lda, d_B, ldb, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfds = gflopsS / gpu_time;
            if (info != 0)
                printf("magma_zpotrs returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Single Precision Factor
            //=====================================================================
            d_As = d_works;
            d_Bs = d_works + lda*N;
            magma_zsetmatrix( N, N,    h_A, lda, d_A, lda );
            magma_zsetmatrix( N, nrhs, h_B, ldb, d_B, ldb );
            magmablas_zlag2c( N, N,    d_A, lda, d_As, N, &info );
            magmablas_zlag2c( N, nrhs, d_B, ldb, d_Bs, N, &info );
            
            gpu_time = magma_wtime();
            magma_cpotrf_gpu(opts.uplo, N, d_As, N, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfsf = gflopsF / gpu_time;
            if (info != 0)
                printf("magma_cpotrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //=====================================================================
            //                 Single Precision Solve
            //=====================================================================
            magmablas_zlag2c(N, N,    d_A, lda, d_As, N, &info );
            magmablas_zlag2c(N, nrhs, d_B, ldb, d_Bs, N, &info );
            
            gpu_time = magma_wtime();
            magma_cpotrf_gpu(opts.uplo, N, d_As, lda, &info);
            magma_cpotrs_gpu(opts.uplo, N, nrhs, d_As, N, d_Bs, N, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perfss = gflopsS / gpu_time;
            if (info != 0)
                printf("magma_cpotrs returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            printf("%5d %5d   %7.2f   %7.2f   %7.2f   %7.2f   %7.2f    %4d   %8.2e   %s\n",
                   (int) N, (int) nrhs,
                   gpu_perfdf, gpu_perfds, gpu_perfsf, gpu_perfss, gpu_perf,
                   (int) posv_iter, error, (error < tol ? "ok" : "failed"));
            status += ! (error < tol);
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( h_X );
            TESTING_FREE_CPU( h_workd );
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            TESTING_FREE_DEV( d_X );
            TESTING_FREE_DEV( d_works );
            TESTING_FREE_DEV( d_workd );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
