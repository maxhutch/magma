/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Stan Tomov
       @author Mark Gates

       @generated from testing/testing_zhetrd_mgpu.cpp, normal z -> c, Sun Nov 20 20:20:37 2016

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

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing chetrd
*/

int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gflops, gpu_perf, cpu_perf, gpu_time, cpu_time;
    magmaFloatComplex *h_A, *h_R, *h_Q, *h_work, *work;
    magmaFloatComplex *tau;
    float *diag, *offdiag;
    #ifdef COMPLEX
    float *rwork = NULL;
    #endif
    float result[2] = {0., 0.};
    magma_int_t N, n2, lda, lwork, info, nb;
    magma_int_t ione     = 1;
    magma_int_t itwo     = 2;
    magma_int_t ithree   = 3;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t k = 1;  // TODO: UNKNOWN, UNDOCUMENTED VARIABLE (number of streams?)

    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    float eps = lapackf77_slamch( "E" );

    /* To avoid uninitialized variable warning */
    h_Q   = NULL;
    work  = NULL;

    printf("%% uplo = %s, ngpu %lld\n", lapack_uplo_const(opts.uplo), (long long) opts.ngpu );
    printf("%% N     CPU Gflop/s (sec)   GPU Gflop/s (sec)   |A-QHQ^H|/N|A|   |I-QQ^H|/N\n");
    printf("%%==========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N      = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            nb     = magma_get_chetrd_nb(N);
            /* We suppose the magma nb is bigger than lapack nb */
            lwork  = N*nb;
            gflops = FLOPS_CHETRD( N ) / 1e9;
            
            /* Allocate host memory for the matrix */
            TESTING_CHECK( magma_cmalloc_pinned( &h_R,     lda*N ));
            TESTING_CHECK( magma_cmalloc_pinned( &h_work,  lwork ));
            
            TESTING_CHECK( magma_cmalloc_cpu( &h_A,     lda*N ));
            TESTING_CHECK( magma_cmalloc_cpu( &tau,     N     ));
            TESTING_CHECK( magma_smalloc_cpu( &diag,    N   ));
            TESTING_CHECK( magma_smalloc_cpu( &offdiag, N-1 ));
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            magma_cmake_hermitian( N, h_A, lda );
            
            lapackf77_clacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.ngpu == 0 ) {
                magma_chetrd(opts.uplo, N, h_R, lda, diag, offdiag,
                             tau, h_work, lwork, &info);
            } else {
                magma_chetrd_mgpu(opts.ngpu, k, opts.uplo, N, h_R, lda, diag, offdiag,
                                  tau, h_work, lwork, &info);
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_chetrd_mgpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            gpu_perf = gflops / gpu_time;
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                TESTING_CHECK( magma_cmalloc_cpu( &h_Q,  lda*N ));
                TESTING_CHECK( magma_cmalloc_cpu( &work, 2*N*N ));
                #ifdef COMPLEX
                TESTING_CHECK( magma_smalloc_cpu( &rwork, N ));
                #endif
                
                lapackf77_clacpy( lapack_uplo_const(opts.uplo), &N, &N, h_R, &lda, h_Q, &lda);
                lapackf77_cungtr( lapack_uplo_const(opts.uplo), &N, h_Q, &lda, tau, h_work, &lwork, &info);

                lapackf77_chet21( &itwo, lapack_uplo_const(opts.uplo), &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &result[0] );
                
                lapackf77_chet21( &ithree, lapack_uplo_const(opts.uplo), &N, &ione,
                                  h_A, &lda, diag, offdiag,
                                  h_Q, &lda, h_R, &lda,
                                  tau, work,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &result[1] );
                result[0] *= eps;
                result[1] *= eps;

                magma_free_cpu( h_Q );
                magma_free_cpu( work );
                #ifdef COMPLEX
                magma_free_cpu( rwork );
                #endif
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_chetrd(lapack_uplo_const(opts.uplo), &N, h_A, &lda, diag, offdiag, tau,
                                 h_work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_chetrd returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld   %7.2f (%7.2f)", (long long) N, cpu_perf, cpu_time );
            }
            else {
                printf("%5lld     ---   (  ---  )", (long long) N );
            }
            printf("   %7.2f (%7.2f)", gpu_perf, gpu_time );
            if ( opts.check ) {
                printf("     %8.2e        %8.2e   %s\n",
                       result[0], result[1], (result[0] < tol && result[1] < tol ? "ok" : "failed") );
                status += ! (result[0] < tol && result[1] < tol);
            }
            else {
                printf("       ---  \n");
            }

            magma_free_pinned( h_R );
            magma_free_pinned( h_work );

            magma_free_cpu( h_A );
            magma_free_cpu( tau );
            magma_free_cpu( diag );
            magma_free_cpu( offdiag );
            
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
