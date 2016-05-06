/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zgels.cpp normal z -> d, Mon May  2 23:31:16 2016

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
   -- Testing dgels
*/
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           gpu_error, cpu_error, error, Anorm, work[1];
    double  c_one     = MAGMA_D_ONE;
    double  c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_A2, *h_B, *h_B2, *h_R, *tau, *h_work, tmp[1];
    magma_int_t M, N, size, nrhs, lda, ldb, min_mn, max_mn, nb, info;
    magma_int_t lhwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    opts.parse_opts( argc, argv );
 
    magma_int_t status = 0;
    double tol = opts.tolerance * lapackf77_dlamch("E");

    nrhs = opts.nrhs;
    
    printf("%%                                                           ||b-Ax|| / (N||A||)   ||dx-x||/(N||A||)\n");
    printf("%%   M     N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   CPU        GPU                         \n");
    printf("%%==================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            if ( M < N ) {
                printf( "%5d %5d %5d   skipping because M < N is not yet supported.\n", (int) M, (int) N, (int) nrhs );
                continue;
            }
            min_mn = min(M, N);
            max_mn = max(M, N);
            lda    = M;
            ldb    = max_mn;
            nb     = magma_get_dgeqrf_nb( M, N );
            gflops = (FLOPS_DGEQRF( M, N ) + FLOPS_DGEQRS( M, N, nrhs )) / 1e9;
            
            // query for workspace size
            lhwork = -1;
            lapackf77_dgels( MagmaNoTransStr, &M, &N, &nrhs,
                             NULL, &lda, NULL, &ldb, tmp, &lhwork, &info );
            lhwork = (magma_int_t) MAGMA_D_REAL( tmp[0] );
            lhwork = max(lhwork, N*nb);
            lhwork = max(lhwork, 2*nb*nb );
            
            TESTING_MALLOC_CPU( tau,    double, min_mn    );
            TESTING_MALLOC_CPU( h_A,    double, lda*N     );
            TESTING_MALLOC_PIN( h_A2,   double, lda*N     );
            TESTING_MALLOC_CPU( h_B,    double, ldb*nrhs  );
            TESTING_MALLOC_CPU( h_B2,   double, ldb*nrhs  );
            TESTING_MALLOC_CPU( h_R,    double, ldb*nrhs  );
            TESTING_MALLOC_CPU( h_work, double, lhwork    );
            
            /* Initialize the matrices */
            size = lda*N;
            lapackf77_dlarnv( &ione, ISEED, &size, h_A );
            lapackf77_dlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_A2, &lda );
            
            // make random RHS
            size = ldb*nrhs;
            lapackf77_dlarnv( &ione, ISEED, &size, h_B );
            lapackf77_dlacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_R , &ldb );
            lapackf77_dlacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_B2, &ldb );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            magma_dgels( MagmaNoTrans, M, N, nrhs, h_A2, lda,
                         h_B2, ldb, h_work, lhwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_dgels_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            // compute the residual
            blasf77_dgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A, &lda,
                                       h_B2, &ldb,
                           &c_one,     h_R, &ldb );
            Anorm = lapackf77_dlange("f", &M, &N, h_A, &lda, work);
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            lapackf77_dlacpy( MagmaFullStr, &M, &N, h_A, &lda, h_A2, &lda );
            lapackf77_dlacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_B2, &ldb );
            
            cpu_time = magma_wtime();
            lapackf77_dgels( MagmaNoTransStr, &M, &N, &nrhs,
                             h_A2, &lda, h_B2, &ldb, h_work, &lhwork, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0) {
                printf("lapackf77_dgels returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            blasf77_dgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A, &lda,
                                       h_B2,  &ldb,
                           &c_one,     h_B,  &ldb );
            
            cpu_error = lapackf77_dlange("f", &M, &nrhs, h_B, &ldb, work) / (min_mn*Anorm);
            gpu_error = lapackf77_dlange("f", &M, &nrhs, h_R, &ldb, work) / (min_mn*Anorm);
            
            // error relative to LAPACK
            size = M*nrhs;
            blasf77_daxpy( &size, &c_neg_one, h_B, &ione, h_R, &ione );
            error = lapackf77_dlange("f", &M, &nrhs, h_R, &ldb, work) / (min_mn*Anorm);
            
            printf("%5d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e",
                   (int) M, (int) N, (int) nrhs,
                   cpu_perf, cpu_time, gpu_perf, gpu_time, cpu_error, gpu_error, error );
            
            if ( M == N ) {
                printf( "   %s\n", (gpu_error < tol && error < tol ? "ok" : "failed"));
                status += ! (gpu_error < tol && error < tol);
            }
            else {
                printf( "   %s\n", (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }

            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_PIN( h_A2   );
            TESTING_FREE_CPU( h_B    );
            TESTING_FREE_CPU( h_B2   );
            TESTING_FREE_CPU( h_R    );
            TESTING_FREE_CPU( h_work );
            
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
