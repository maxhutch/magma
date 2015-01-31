/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zgetri_gpu.cpp normal z -> s, Fri Jan 30 19:00:24 2015
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
   -- Testing sgetrf
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float *h_A, *h_R, *work;
    magmaFloat_ptr d_A, dwork;
    float c_neg_one = MAGMA_S_NEG_ONE;
    magma_int_t N, n2, lda, ldda, info, lwork, ldwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    float tmp;
    float error, rwork[1];
    magma_int_t *ipiv;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    // need looser bound (3000*eps instead of 30*eps) for tests
    // TODO: should compute ||I - A*A^{-1}|| / (n*||A||*||A^{-1}||)
    opts.tolerance = max( 3000., opts.tolerance );
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("    N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / (N*||A||_F)\n");
    printf("=================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            ldda   = ((N+31)/32)*32;
            ldwork = N * magma_get_sgetri_nb( N );
            gflops = FLOPS_SGETRI( N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_sgetri( &N, NULL, &lda, NULL, &tmp, &lwork, &info );
            if (info != 0)
                printf("lapackf77_sgetri returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            lwork = int( MAGMA_S_REAL( tmp ));
            
            TESTING_MALLOC_CPU( ipiv,  magma_int_t,        N      );
            TESTING_MALLOC_CPU( work,  float, lwork  );
            TESTING_MALLOC_CPU( h_A,   float, n2     );
            
            TESTING_MALLOC_PIN( h_R,   float, n2     );
            
            TESTING_MALLOC_DEV( d_A,   float, ldda*N );
            TESTING_MALLOC_DEV( dwork, float, ldwork );
            
            /* Initialize the matrix */
            lapackf77_slarnv( &ione, ISEED, &n2, h_A );
            error = lapackf77_slange( "f", &N, &N, h_A, &lda, rwork );  // norm(A)
            
            /* Factor the matrix. Both MAGMA and LAPACK will use this factor. */
            magma_ssetmatrix( N, N, h_A, lda, d_A, ldda );
            magma_sgetrf_gpu( N, N, d_A, ldda, ipiv, &info );
            magma_sgetmatrix( N, N, d_A, ldda, h_A, lda );
            if ( info != 0 )
                printf("magma_sgetrf_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            // check for exact singularity
            //h_A[ 10 + 10*lda ] = MAGMA_S_MAKE( 0.0, 0.0 );
            //magma_ssetmatrix( N, N, h_A, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_sgetri_gpu( N, d_A, ldda, ipiv, dwork, ldwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_sgetri_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            magma_sgetmatrix( N, N, d_A, ldda, h_R, lda );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_sgetri( &N, h_A, &lda, ipiv, work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_sgetri returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                blasf77_saxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
                error = lapackf77_slange( "f", &N, &N, h_R, &lda, rwork ) / (N*error);
                
                printf( "%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf( "%5d     ---   (  ---  )   %7.2f (%7.2f)     ---\n",
                        (int) N, gpu_perf, gpu_time );
            }
            
            TESTING_FREE_CPU( ipiv  );
            TESTING_FREE_CPU( work  );
            TESTING_FREE_CPU( h_A   );
            
            TESTING_FREE_PIN( h_R   );
            
            TESTING_FREE_DEV( d_A   );
            TESTING_FREE_DEV( dwork );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
