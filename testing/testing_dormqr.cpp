/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Mark Gates
       @generated d Tue Dec 17 13:18:57 2013
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dormqr
*/
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double error, work[1];
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t m, n, k, size, info;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t nb, ldc, lda, lwork, lwork_max;
    double *C, *R, *A, *W, *tau;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    // test all combinations of input parameters
    const char* side[]   = { MagmaLeftStr,      MagmaRightStr   };
    const char* trans[]  = { MagmaTransStr, MagmaNoTransStr };

    printf("    M     N     K  side   trans      CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R||_F / ||QC||_F\n");
    printf("===============================================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iside = 0; iside < 2; ++iside ) {
        for( int itran = 0; itran < 2; ++itran ) {
            m = opts.msize[i];
            n = opts.nsize[i];
            k = opts.ksize[i];
            nb  = magma_get_dgeqrf_nb( m );
            ldc = ((m + 31)/32)*32;
            lda = ((max(m,n) + 31)/32)*32;
            gflops = FLOPS_DORMQR( m, n, k, *side[iside] ) / 1e9;
            
            if ( *side[iside] == 'L' && m < k ) {
                printf( "%5d %5d %5d  %-5s  %-9s   skipping because side=left and m < k\n",
                        (int) m, (int) n, (int) k, side[iside], trans[itran] );
                continue;
            }
            if ( *side[iside] == 'R' && n < k ) {
                printf( "%5d %5d %5d  %-5s  %-9s   skipping because side=right and n < k\n",
                        (int) m, (int) n, (int) k, side[iside], trans[itran] );
                continue;
            }
            
            // need at least 2*nb*nb for geqrf
            lwork_max = max( max( m*nb, n*nb ), 2*nb*nb );
            
            TESTING_MALLOC_CPU( C,   double, ldc*n );
            TESTING_MALLOC_CPU( R,   double, ldc*n );
            TESTING_MALLOC_CPU( A,   double, lda*k );
            TESTING_MALLOC_CPU( W,   double, lwork_max );
            TESTING_MALLOC_CPU( tau, double, k );
            
            // C is full, m x n
            size = ldc*n;
            lapackf77_dlarnv( &ione, ISEED, &size, C );
            lapackf77_dlacpy( "Full", &m, &n, C, &ldc, R, &ldc );
            //magma_dsetmatrix( m,   n, C, ldc, dC, ldc );
            
            // A is m x k (left) or n x k (right)
            lda = (*side[iside] == 'L' ? m : n);
            size = lda*k;
            lapackf77_dlarnv( &ione, ISEED, &size, A );
            
            // compute QR factorization to get Householder vectors in A, tau
            magma_dgeqrf( lda, k, A, lda, tau, W, lwork_max, &info );
            if (info != 0)
                printf("magma_dgeqrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_dormqr( side[iside], trans[itran],
                              &m, &n, &k,
                              A, &lda, tau, C, &ldc, W, &lwork_max, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            if (info != 0)
                printf("lapackf77_dormqr returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            // query for workspace size
            lwork = -1;
            magma_dormqr( *side[iside], *trans[itran],
                          m, n, k,
                          A, lda, tau, R, ldc, W, lwork, &info );
            if (info != 0)
                printf("magma_dormqr (lwork query) returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            lwork = (magma_int_t) MAGMA_D_REAL( W[0] );
            if ( lwork < 0 || lwork > lwork_max )
                printf("invalid lwork %d, lwork_max %d\n", (int) lwork, (int) lwork_max );
            
            gpu_time = magma_wtime();
            magma_dormqr( *side[iside], *trans[itran],
                          m, n, k,
                          A, lda, tau, R, ldc, W, lwork, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_dormqr returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            //magma_dgetmatrix( m, n, dC, ldc, R, ldc );
            
            /* =====================================================================
               compute relative error |QC_magma - QC_lapack| / |QC_lapack|
               =================================================================== */
            error = lapackf77_dlange( "Fro", &m, &n, C, &ldc, work );
            size = ldc*n;
            blasf77_daxpy( &size, &c_neg_one, C, &ione, R, &ione );
            error = lapackf77_dlange( "Fro", &m, &n, R, &ldc, work ) / error;
            
            printf( "%5d %5d %5d  %-5s  %-9s  %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e\n",
                    (int) m, (int) n, (int) k, side[iside], trans[itran],
                    cpu_perf, cpu_time, gpu_perf, gpu_time, error );
            
            TESTING_FREE_CPU( C );
            TESTING_FREE_CPU( R );
            TESTING_FREE_CPU( A );
            TESTING_FREE_CPU( W );
            TESTING_FREE_CPU( tau );
        }}  // end iside, itran
        printf( "\n" );
    }
    
    TESTING_FINALIZE();
    return 0;
}
