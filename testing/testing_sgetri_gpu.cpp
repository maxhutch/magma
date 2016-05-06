/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zgetri_gpu.cpp normal z -> s, Mon May  2 23:31:13 2016
       @author Mark Gates
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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgetri
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    // constants
    const float c_zero    = MAGMA_S_ZERO;
    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float *h_A, *h_Ainv, *h_R, *work;
    magmaFloat_ptr d_A, dwork;
    magma_int_t N, n2, lda, ldda, info, lwork, ldwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    float tmp;
    float error, rwork[1];
    magma_int_t *ipiv;
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("%%   N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||I - A*A^{-1}||_1 / (N*cond(A))\n");
    printf("%%===============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            ldwork = N * magma_get_sgetri_nb( N );
            gflops = FLOPS_SGETRI( N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_sgetri( &N, NULL, &lda, NULL, &tmp, &lwork, &info );
            if (info != 0) {
                printf("lapackf77_sgetri returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            lwork = magma_int_t( MAGMA_S_REAL( tmp ));
            
            TESTING_MALLOC_CPU( ipiv,   magma_int_t,        N      );
            TESTING_MALLOC_CPU( work,   float, lwork  );
            TESTING_MALLOC_CPU( h_A,    float, n2     );
            TESTING_MALLOC_CPU( h_Ainv, float, n2     );
            TESTING_MALLOC_CPU( h_R,    float, n2     );
            
            TESTING_MALLOC_DEV( d_A,    float, ldda*N );
            TESTING_MALLOC_DEV( dwork,  float, ldwork );
            
            /* Initialize the matrix */
            lapackf77_slarnv( &ione, ISEED, &n2, h_A );
            
            /* Factor the matrix. Both MAGMA and LAPACK will use this factor. */
            magma_ssetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            magma_sgetrf_gpu( N, N, d_A, ldda, ipiv, &info );
            magma_sgetmatrix( N, N, d_A, ldda, h_Ainv, lda, opts.queue );
            if (info != 0) {
                printf("magma_sgetrf_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            // check for exact singularity
            //h_Ainv[ 10 + 10*lda ] = MAGMA_S_MAKE( 0.0, 0.0 );
            //magma_ssetmatrix( N, N, h_Ainv, lda, d_A, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_sgetri_gpu( N, d_A, ldda, ipiv, dwork, ldwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_sgetri_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_sgetri( &N, h_Ainv, &lda, ipiv, work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_sgetri returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                printf( "%5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                        (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf( "%5d     ---   (  ---  )   %7.2f (%7.2f)",
                        (int) N, gpu_perf, gpu_time );
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.check ) {
                magma_sgetmatrix( N, N, d_A, ldda, h_Ainv, lda, opts.queue );
                
                // compute 1-norm condition number estimate, following LAPACK's zget03
                float normA, normAinv, rcond;
                normA    = lapackf77_slange( "1", &N, &N, h_A,    &lda, rwork );
                normAinv = lapackf77_slange( "1", &N, &N, h_Ainv, &lda, rwork );
                if ( normA <= 0 || normAinv <= 0 ) {
                    rcond = 0;
                    error = 1 / (tol/opts.tolerance);  // == 1/eps
                }
                else {
                    rcond = (1 / normA) / normAinv;
                    // R = I
                    // R -= A*A^{-1}
                    // err = ||I - A*A^{-1}|| / ( N ||A||*||A^{-1}|| ) = ||R|| * rcond / N, using 1-norm
                    lapackf77_slaset( "full", &N, &N, &c_zero, &c_one, h_R, &lda );
                    blasf77_sgemm( "no", "no", &N, &N, &N,
                                   &c_neg_one, h_A,    &lda,
                                               h_Ainv, &lda,
                                   &c_one,     h_R,    &lda );
                    error = lapackf77_slange( "1", &N, &N, h_R, &lda, rwork );
                    error = error * rcond / N;
                }
                
                bool okay = (error < tol);
                status += ! okay;
                printf( "   %8.2e   %s\n",
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "\n" );
            }
            
            TESTING_FREE_CPU( ipiv   );
            TESTING_FREE_CPU( work   );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_Ainv );
            TESTING_FREE_CPU( h_R    );
            
            TESTING_FREE_DEV( d_A    );
            TESTING_FREE_DEV( dwork  );
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
