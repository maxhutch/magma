/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zgetri_gpu.cpp, normal z -> c, Sun Nov 20 20:20:36 2016
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
   -- Testing cgetri
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    // constants
    const magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaFloatComplex *h_A, *h_Ainv, *h_R, *work, unused[1];
    magmaFloatComplex_ptr d_A, dwork;
    magma_int_t N, n2, lda, ldda, info, lwork, ldwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magmaFloatComplex tmp;
    float error, rwork[1];
    magma_int_t *ipiv, iunused[1];
    int status = 0;
    
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
            ldwork = N * magma_get_cgetri_nb( N );
            gflops = FLOPS_CGETRI( N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_cgetri( &N, unused, &lda, iunused, &tmp, &lwork, &info );
            if (info != 0) {
                printf("lapackf77_cgetri returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            lwork = magma_int_t( MAGMA_C_REAL( tmp ));
            
            TESTING_CHECK( magma_imalloc_cpu( &ipiv,   N      ));
            TESTING_CHECK( magma_cmalloc_cpu( &work,   lwork  ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_A,    n2     ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_Ainv, n2     ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_R,    n2     ));
            
            TESTING_CHECK( magma_cmalloc( &d_A,    ldda*N ));
            TESTING_CHECK( magma_cmalloc( &dwork,  ldwork ));
            
            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            
            /* Factor the matrix. Both MAGMA and LAPACK will use this factor. */
            magma_csetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            magma_cgetrf_gpu( N, N, d_A, ldda, ipiv, &info );
            magma_cgetmatrix( N, N, d_A, ldda, h_Ainv, lda, opts.queue );
            if (info != 0) {
                printf("magma_cgetrf_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            // check for exact singularity
            //h_Ainv[ 10 + 10*lda ] = MAGMA_C_MAKE( 0.0, 0.0 );
            //magma_csetmatrix( N, N, h_Ainv, lda, d_A, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_cgetri_gpu( N, d_A, ldda, ipiv, dwork, ldwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_cgetri_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_cgetri( &N, h_Ainv, &lda, ipiv, work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_cgetri returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                printf( "%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                        (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf( "%5lld     ---   (  ---  )   %7.2f (%7.2f)",
                        (long long) N, gpu_perf, gpu_time );
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.check ) {
                magma_cgetmatrix( N, N, d_A, ldda, h_Ainv, lda, opts.queue );
                
                // compute 1-norm condition number estimate, following LAPACK's zget03
                float normA, normAinv, rcond;
                normA    = lapackf77_clange( "1", &N, &N, h_A,    &lda, rwork );
                normAinv = lapackf77_clange( "1", &N, &N, h_Ainv, &lda, rwork );
                if ( normA <= 0 || normAinv <= 0 ) {
                    rcond = 0;
                    error = 1 / (tol/opts.tolerance);  // == 1/eps
                }
                else {
                    rcond = (1 / normA) / normAinv;
                    // R = I
                    // R -= A*A^{-1}
                    // err = ||I - A*A^{-1}|| / ( N ||A||*||A^{-1}|| ) = ||R|| * rcond / N, using 1-norm
                    lapackf77_claset( "full", &N, &N, &c_zero, &c_one, h_R, &lda );
                    blasf77_cgemm( "no", "no", &N, &N, &N,
                                   &c_neg_one, h_A,    &lda,
                                               h_Ainv, &lda,
                                   &c_one,     h_R,    &lda );
                    error = lapackf77_clange( "1", &N, &N, h_R, &lda, rwork );
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
            
            magma_free_cpu( ipiv   );
            magma_free_cpu( work   );
            magma_free_cpu( h_A    );
            magma_free_cpu( h_Ainv );
            magma_free_cpu( h_R    );
            
            magma_free( d_A    );
            magma_free( dwork  );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
