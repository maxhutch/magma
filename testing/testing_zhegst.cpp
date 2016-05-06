/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates

       @precisions normal z -> s d c

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
   -- Testing zhegst
*/
int main( int argc, char** argv)
{
    TESTING_INIT();
    
    // Constants
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;

    // Local variables
    real_Double_t gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_B, *h_R;
    double      Anorm, error, work[1];
    magma_int_t N, n2, lda, info;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% itype   N   CPU time (sec)   GPU time (sec)   |R|     \n");
    printf("%%=======================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = N*lda;
            
            TESTING_MALLOC_CPU( h_A,     magmaDoubleComplex, lda*N );
            TESTING_MALLOC_CPU( h_B,     magmaDoubleComplex, lda*N );
            
            TESTING_MALLOC_PIN( h_R,     magmaDoubleComplex, lda*N );
            
            /* ====================================================================
               Initialize the matrix
               =================================================================== */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );
            lapackf77_zlarnv( &ione, ISEED, &n2, h_B );
            magma_zmake_hermitian( N, h_A, lda );
            magma_zmake_hpd(       N, h_B, lda );
            magma_zpotrf( opts.uplo, N, h_B, lda, &info );
            if (info != 0) {
                printf("magma_zpotrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_zhegst( opts.itype, opts.uplo, N, h_R, lda, h_B, lda, &info );
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_zhegst returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_zhegst( &opts.itype, lapack_uplo_const(opts.uplo),
                                  &N, h_A, &lda, h_B, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0) {
                    printf("lapackf77_zhegst returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                
                blasf77_zaxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
                Anorm = safe_lapackf77_zlanhe("f", lapack_uplo_const(opts.uplo), &N, h_A, &lda, work );
                error = safe_lapackf77_zlanhe("f", lapack_uplo_const(opts.uplo), &N, h_R, &lda, work )
                      / Anorm;
                
                bool okay = (error < tol);
                status += ! okay;
                printf("%3d   %5d   %7.2f          %7.2f          %8.2e   %s\n",
                       (int) opts.itype, (int) N, cpu_time, gpu_time,
                       error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%3d   %5d     ---            %7.2f\n",
                       (int) opts.itype, (int) N, gpu_time );
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            
            TESTING_FREE_PIN( h_R );
            
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
