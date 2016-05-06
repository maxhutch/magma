/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zgels_gpu.cpp normal z -> c, Mon May  2 23:31:14 2016

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
   -- Testing cgels
*/
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float           gpu_error, cpu_error, error, Anorm, work[1];
    magmaFloatComplex  c_one     = MAGMA_C_ONE;
    magmaFloatComplex  c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex *h_A, *h_A2, *h_B, *h_X, *h_R, *tau, *h_work, tmp[1];
    magmaFloatComplex_ptr d_A, d_B;
    magma_int_t M, N, size, nrhs, lda, ldb, ldda, lddb, min_mn, max_mn, nb, info;
    magma_int_t lworkgpu, lhwork;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    opts.parse_opts( argc, argv );
 
    magma_int_t status = 0;
    float tol = opts.tolerance * lapackf77_slamch("E");

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
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            lddb   = magma_roundup( max_mn, opts.align );  // multiple of 32 by default
            nb     = magma_get_cgeqrf_nb( M, N );
            gflops = (FLOPS_CGEQRF( M, N ) + FLOPS_CGEQRS( M, N, nrhs )) / 1e9;
            
            lworkgpu = (M - N + nb)*(nrhs + nb) + nrhs*nb;
            
            // query for workspace size
            lhwork = -1;
            lapackf77_cgels( MagmaNoTransStr, &M, &N, &nrhs,
                             NULL, &lda, NULL, &ldb, tmp, &lhwork, &info );
            lhwork = (magma_int_t) MAGMA_C_REAL( tmp[0] );
            lhwork = max( lhwork, lworkgpu );
            
            TESTING_MALLOC_CPU( tau,    magmaFloatComplex, min_mn    );
            TESTING_MALLOC_CPU( h_A,    magmaFloatComplex, lda*N     );
            TESTING_MALLOC_CPU( h_A2,   magmaFloatComplex, lda*N     );
            TESTING_MALLOC_CPU( h_B,    magmaFloatComplex, ldb*nrhs  );
            TESTING_MALLOC_CPU( h_X,    magmaFloatComplex, ldb*nrhs  );
            TESTING_MALLOC_CPU( h_R,    magmaFloatComplex, ldb*nrhs  );
            TESTING_MALLOC_CPU( h_work, magmaFloatComplex, lhwork    );
            
            TESTING_MALLOC_DEV( d_A,    magmaFloatComplex, ldda*N    );
            TESTING_MALLOC_DEV( d_B,    magmaFloatComplex, lddb*nrhs );
            
            /* Initialize the matrices */
            size = lda*N;
            lapackf77_clarnv( &ione, ISEED, &size, h_A );
            lapackf77_clacpy( MagmaFullStr, &M, &N, h_A, &lda, h_A2, &lda );
            
            // make random RHS
            size = ldb*nrhs;
            lapackf77_clarnv( &ione, ISEED, &size, h_B );
            lapackf77_clacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_R, &ldb );
            
            // make consistent RHS
            //size = N*nrhs;
            //lapackf77_clarnv( &ione, ISEED, &size, h_X );
            //blasf77_cgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
            //               &c_one,  h_A, &lda,
            //                        h_X, &ldb,
            //               &c_zero, h_B, &ldb );
            //lapackf77_clacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_R, &ldb );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_csetmatrix( M, N,    h_A, lda, d_A, ldda, opts.queue );
            magma_csetmatrix( M, nrhs, h_B, ldb, d_B, lddb, opts.queue );
            
            gpu_time = magma_wtime();
            magma_cgels_gpu( MagmaNoTrans, M, N, nrhs, d_A, ldda,
                             d_B, lddb, h_work, lworkgpu, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_cgels_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            // compute the residual
            magma_cgetmatrix( N, nrhs, d_B, lddb, h_X, ldb, opts.queue );
            blasf77_cgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A, &lda,
                                       h_X, &ldb,
                           &c_one,     h_R, &ldb );
            Anorm = lapackf77_clange("f", &M, &N, h_A, &lda, work);
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            lapackf77_clacpy( MagmaFullStr, &M, &nrhs, h_B, &ldb, h_X, &ldb );
            
            cpu_time = magma_wtime();
            lapackf77_cgels( MagmaNoTransStr, &M, &N, &nrhs,
                             h_A, &lda, h_X, &ldb, h_work, &lhwork, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0) {
                printf("lapackf77_cgels returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            blasf77_cgemm( MagmaNoTransStr, MagmaNoTransStr, &M, &nrhs, &N,
                           &c_neg_one, h_A2, &lda,
                                       h_X,  &ldb,
                           &c_one,     h_B,  &ldb );
            
            cpu_error = lapackf77_clange("f", &M, &nrhs, h_B, &ldb, work) / (min_mn*Anorm);
            gpu_error = lapackf77_clange("f", &M, &nrhs, h_R, &ldb, work) / (min_mn*Anorm);
            
            // error relative to LAPACK
            size = M*nrhs;
            blasf77_caxpy( &size, &c_neg_one, h_B, &ione, h_R, &ione );
            error = lapackf77_clange("f", &M, &nrhs, h_R, &ldb, work) / (min_mn*Anorm);
            
            printf("%5d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e",
                   (int) M, (int) N, (int) nrhs,
                   cpu_perf, cpu_time, gpu_perf, gpu_time, cpu_error, gpu_error, error );
            
            bool okay;
            if ( M == N ) {
                okay = (gpu_error < tol && error < tol);
            }
            else {
                okay = (error < tol);
            }
            status += ! okay;
            printf( "   %s\n", (okay ? "ok" : "failed"));

            TESTING_FREE_CPU( tau    );
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_A2   );
            TESTING_FREE_CPU( h_B    );
            TESTING_FREE_CPU( h_X    );
            TESTING_FREE_CPU( h_R    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_DEV( d_A    );
            TESTING_FREE_DEV( d_B    );
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
