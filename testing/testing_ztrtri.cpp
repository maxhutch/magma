/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015
  
       @precisions normal z -> c d s
       
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrtri
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t N, n2, lda, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double      work[1], error, norm;
    magma_int_t status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||A||_F\n");
    printf("%%================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            gflops = FLOPS_ZTRTRI( N ) / 1e9;
            
            TESTING_MALLOC_CPU( h_A, magmaDoubleComplex, n2 );
            TESTING_MALLOC_PIN( h_R, magmaDoubleComplex, n2 );
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            magma_zmake_hpd( N, h_A, lda );
            lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            if ( opts.warmup ) {
                magma_zpotrf( opts.uplo, N, h_R, lda, &info );
                magma_ztrtri( opts.uplo, opts.diag, N, h_R, lda, &info );
                lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            }
            
            /* factorize matrix */
            magma_zpotrf( opts.uplo, N, h_R, lda, &info );
            
            // check for exact singularity
            //h_R[ 10 + 10*lda ] = MAGMA_Z_MAKE( 0.0, 0.0 );
            
            gpu_time = magma_wtime();
            magma_ztrtri( opts.uplo, opts.diag, N, h_R, lda, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_ztrtri returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                
                cpu_time = magma_wtime();
                lapackf77_ztrtri( lapack_uplo_const(opts.uplo), lapack_diag_const(opts.diag), &N, h_A, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_ztrtri returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                norm  = lapackf77_zlantr("f", lapack_uplo_const(opts.uplo), MagmaNonUnitStr, &N, &N, h_A, &N, work);
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                error = lapackf77_zlantr("f", lapack_uplo_const(opts.uplo), MagmaNonUnitStr, &N, &N, h_R, &N, work) / norm;
                printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (error < tol ? "ok" : "failed") );
                status += ! (error < tol);
            }
            else {
                printf("%5d     ---   (  ---  )   %7.2f (%7.2f)     ---\n",
                       (int) N, gpu_perf, gpu_time );
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
