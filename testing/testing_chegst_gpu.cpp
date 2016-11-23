/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Mark Gates

       @generated from testing/testing_zhegst_gpu.cpp, normal z -> c, Sun Nov 20 20:20:36 2016

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
   -- Testing chegst
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    // Constants
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    const magma_int_t ione = 1;

    // Local variables
    real_Double_t gpu_time, cpu_time;
    magmaFloatComplex *h_A, *h_B, *h_R;
    magmaFloatComplex_ptr d_A, d_B;
    float      Anorm, error, work[1];
    magma_int_t N, n2, lda, ldda, info;
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
            ldda   = magma_roundup( lda, opts.align );
            n2     = N*lda;
            
            TESTING_CHECK( magma_cmalloc_cpu( &h_A,     lda*N ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_B,     lda*N ));
            
            TESTING_CHECK( magma_cmalloc_pinned( &h_R,     lda*N ));
            
            TESTING_CHECK( magma_cmalloc( &d_A,     ldda*N ));
            TESTING_CHECK( magma_cmalloc( &d_B,     ldda*N ));
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            lapackf77_clarnv( &ione, ISEED, &n2, h_B );
            magma_cmake_hermitian( N, h_A, lda );
            magma_cmake_hpd(       N, h_B, lda );
            magma_cpotrf( opts.uplo, N, h_B, lda, &info );
            if (info != 0) {
                printf("magma_cpotrf returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            magma_csetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            magma_csetmatrix( N, N, h_B, lda, d_B, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_chegst_gpu( opts.itype, opts.uplo, N, d_A, ldda, d_B, ldda, &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_chegst_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_chegst( &opts.itype, lapack_uplo_const(opts.uplo),
                                  &N, h_A, &lda, h_B, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0) {
                    printf("lapackf77_chegst returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                magma_cgetmatrix( N, N, d_A, ldda, h_R, lda, opts.queue );
                
                blasf77_caxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
                Anorm = safe_lapackf77_clanhe("f", lapack_uplo_const(opts.uplo), &N, h_A, &lda, work );
                error = safe_lapackf77_clanhe("f", lapack_uplo_const(opts.uplo), &N, h_R, &lda, work )
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
            
            magma_free( d_A );
            magma_free( d_B );
            
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
