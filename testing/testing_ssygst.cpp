/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Mark Gates

       @generated from testing/testing_zhegst.cpp, normal z -> s, Sun Nov 20 20:20:36 2016

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

#define REAL

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ssygst
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    // Constants
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const magma_int_t ione = 1;

    // Local variables
    real_Double_t gpu_time, cpu_time;
    float *h_A, *h_B, *h_R;
    float      Anorm, error, work[1];
    magma_int_t N, n2, lda, info;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    float tol = opts.tolerance * lapackf77_slamch("E");

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% itype   N   CPU time (sec)   GPU time (sec)   |R|     \n");
    printf("%%=======================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = N*lda;
            
            TESTING_CHECK( magma_smalloc_cpu( &h_A,     lda*N ));
            TESTING_CHECK( magma_smalloc_cpu( &h_B,     lda*N ));
            
            TESTING_CHECK( magma_smalloc_pinned( &h_R,     lda*N ));
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            lapackf77_slarnv( &ione, ISEED, &n2, h_A );
            lapackf77_slarnv( &ione, ISEED, &n2, h_B );
            magma_smake_symmetric( N, h_A, lda );
            magma_smake_hpd(       N, h_B, lda );
            magma_spotrf( opts.uplo, N, h_B, lda, &info );
            if (info != 0) {
                printf("magma_spotrf returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            lapackf77_slacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_ssygst( opts.itype, opts.uplo, N, h_R, lda, h_B, lda, &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_ssygst returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_ssygst( &opts.itype, lapack_uplo_const(opts.uplo),
                                  &N, h_A, &lda, h_B, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0) {
                    printf("lapackf77_ssygst returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                blasf77_saxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
                Anorm = safe_lapackf77_slansy("f", lapack_uplo_const(opts.uplo), &N, h_A, &lda, work );
                error = safe_lapackf77_slansy("f", lapack_uplo_const(opts.uplo), &N, h_R, &lda, work )
                      / Anorm;
                
                bool okay = (error < tol);
                status += ! okay;
                printf("%3lld   %5lld   %7.2f          %7.2f          %8.2e   %s\n",
                       (long long) opts.itype, (long long) N, cpu_time, gpu_time,
                       error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%3lld   %5lld     ---            %7.2f\n",
                       (long long) opts.itype, (long long) N, gpu_time );
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            
            magma_free_pinned( h_R );
            
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
