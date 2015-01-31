/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zgebrd.cpp normal z -> c, Fri Jan 30 19:00:26 2015

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

#define PRECISION_c

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgebrd
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaFloatComplex *h_A, *h_Q, *h_PT, *h_work;
    magmaFloatComplex *taup, *tauq;
    float      *diag, *offdiag;
    float      eps, result[3] = {0., 0., 0.};
    magma_int_t M, N, n2, lda, lhwork, info, minmn, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    eps = lapackf77_slamch( "E" );
    magma_int_t status = 0;

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("    M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)   |A-QBP'|/N|A|  |I-QQ'|/N  |I-PP'|/N\n");
    printf("=========================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            minmn  = min(M, N);
            nb     = magma_get_cgebrd_nb(N);
            lda    = M;
            n2     = lda*N;
            lhwork = (M + N)*nb;
            gflops = FLOPS_CGEBRD( M, N ) / 1e9;

            TESTING_MALLOC_CPU( h_A,     magmaFloatComplex, lda*N );
            TESTING_MALLOC_CPU( tauq,    magmaFloatComplex, minmn );
            TESTING_MALLOC_CPU( taup,    magmaFloatComplex, minmn );
            TESTING_MALLOC_CPU( diag,    float, minmn   );
            TESTING_MALLOC_CPU( offdiag, float, minmn-1 );
            
            TESTING_MALLOC_PIN( h_Q,     magmaFloatComplex, lda*N  );
            TESTING_MALLOC_PIN( h_work,  magmaFloatComplex, lhwork );
            
            /* Initialize the matrices */
            lapackf77_clarnv( &ione, ISEED, &n2, h_A );
            lapackf77_clacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_Q, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_cgebrd( M, N, h_Q, lda,
                          diag, offdiag, tauq, taup,
                          h_work, lhwork, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_cgebrd returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.check ) {
                // cungbr prefers minmn*NB
                // cbdt01 needs M+N
                // cunt01 prefers minmn*(minmn+1) to check Q and P
                magma_int_t lwork_err;
                magmaFloatComplex *h_work_err;
                lwork_err = max( minmn * nb, M+N );
                lwork_err = max( lwork_err, minmn*(minmn+1) );
                TESTING_MALLOC_CPU( h_PT,       magmaFloatComplex, lda*N     );
                TESTING_MALLOC_CPU( h_work_err, magmaFloatComplex, lwork_err );
                
                // cbdt01 needs M
                // cunt01 needs minmn
                #if defined(PRECISION_z) || defined(PRECISION_c)
                float *rwork_err;
                TESTING_MALLOC_CPU( rwork_err, float, M );
                #endif

                lapackf77_clacpy(MagmaUpperLowerStr, &M, &N, h_Q, &lda, h_PT, &lda);
                
                // generate Q & P'
                lapackf77_cungbr("Q", &M, &minmn, &N, h_Q,  &lda, tauq, h_work_err, &lwork_err, &info);
                if (info != 0)
                    printf("lapackf77_cungbr #1 returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                lapackf77_cungbr("P", &minmn, &N, &M, h_PT, &lda, taup, h_work_err, &lwork_err, &info);
                if (info != 0)
                    printf("lapackf77_cungbr #2 returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                // Test 1:  Check the decomposition A := Q * B * PT
                //      2:  Check the orthogonality of Q
                //      3:  Check the orthogonality of PT
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_cbdt01(&M, &N, &ione,
                                 h_A, &lda, h_Q, &lda,
                                 diag, offdiag, h_PT, &lda,
                                 h_work_err, rwork_err, &result[0]);
                lapackf77_cunt01("Columns", &M, &minmn, h_Q,  &lda, h_work_err, &lwork_err, rwork_err, &result[1]);
                lapackf77_cunt01("Rows",    &minmn, &N, h_PT, &lda, h_work_err, &lwork_err, rwork_err, &result[2]);
                #else
                lapackf77_cbdt01(&M, &N, &ione,
                                 h_A, &lda, h_Q, &lda,
                                 diag, offdiag, h_PT, &lda,
                                 h_work_err, &result[0]);
                lapackf77_cunt01("Columns", &M, &minmn, h_Q,  &lda, h_work_err, &lwork_err, &result[1]);
                lapackf77_cunt01("Rows",    &minmn, &N, h_PT, &lda, h_work_err, &lwork_err, &result[2]);
                #endif
                
                TESTING_FREE_CPU( h_PT );
                TESTING_FREE_CPU( h_work_err );
                #if defined(PRECISION_z) || defined(PRECISION_c)
                TESTING_FREE_CPU( rwork_err );
                #endif
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_cgebrd(&M, &N, h_A, &lda,
                                 diag, offdiag, tauq, taup,
                                 h_work, &lhwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_cgebrd returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Print performance and error.
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (int) M, (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)",
                       (int) M, (int) N, gpu_perf, gpu_time );
            }
            if ( opts.check ) {
                printf("   %8.2e       %8.2e   %8.2e   %s\n",
                       result[0]*eps, result[1]*eps, result[2]*eps,
                       (result[0]*eps < tol && result[1]*eps < tol && result[2]*eps < tol ? "ok" : "failed") );
                status += ! (result[0]*eps < tol && result[1]*eps < tol && result[2]*eps < tol);
            } else {
                printf("     ---            --- \n");
            }
            
            TESTING_FREE_CPU( h_A     );
            TESTING_FREE_CPU( tauq    );
            TESTING_FREE_CPU( taup    );
            TESTING_FREE_CPU( diag    );
            TESTING_FREE_CPU( offdiag );
            
            TESTING_FREE_PIN( h_Q    );
            TESTING_FREE_PIN( h_work );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
