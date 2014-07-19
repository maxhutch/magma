/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from testing_zlaset.cpp normal z -> d, Fri Jul 18 17:34:23 2014
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dlaset
   Code is very similar to testing_dlacpy.cpp
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    double  c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_R;
    double *d_A;
    double offdiag = MAGMA_D_MAKE( 1.2000, 6.7000 );
    double diag    = MAGMA_D_MAKE( 3.1415, 2.7183 );
    magma_int_t M, N, size, lda, ldb, ldda;
    magma_int_t ione     = 1;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );

    magma_uplo_t uplo[] = { MagmaLower, MagmaUpper, MagmaFull };
    
    printf("uplo       M     N   CPU GByte/s (ms)    GPU GByte/s (ms)    check\n");
    printf("==================================================================\n");
    for( int iuplo = 0; iuplo < 3; ++iuplo ) {
      for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            //M += 2;  // space for insets
            //N += 2;
            lda    = M;
            ldb    = lda;
            ldda   = ((M+31)/32)*32;
            size   = lda*N;
            if ( uplo[iuplo] == MagmaLower || uplo[iuplo] == MagmaUpper ) {
                // save triangle (with diagonal)
                // TODO wrong for trapezoid
                gbytes = sizeof(double) * 0.5*N*(N+1) / 1e9;
            }
            else {
                // save entire matrix
                gbytes = sizeof(double) * 1.*M*N / 1e9;
            }
    
            TESTING_MALLOC_CPU( h_A, double, size   );
            TESTING_MALLOC_CPU( h_R, double, size   );
            
            TESTING_MALLOC_DEV( d_A, double, ldda*N );
            
            /* Initialize the matrix */
            for( int j = 0; j < N; ++j ) {
                for( int i = 0; i < M; ++i ) {
                    h_A[i + j*lda] = MAGMA_D_MAKE( i + j/10000., j );
                }
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_dsetmatrix( M, N, h_A, lda, d_A, ldda );
            
            gpu_time = magma_sync_wtime( 0 );
            //magmablas_dlaset( uplo[iuplo], M-2, N-2, offdiag, diag, d_A+1+ldda, ldda );  // inset by 1 row & col
            magmablas_dlaset( uplo[iuplo], M, N, offdiag, diag, d_A, ldda );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            //magma_int_t M2 = M-2;  // inset by 1 row & col
            //magma_int_t N2 = N-2;
            //lapackf77_dlaset( lapack_uplo_const( uplo[iuplo] ), &M2, &N2, &offdiag, &diag, h_A+1+lda, &lda );
            lapackf77_dlaset( lapack_uplo_const( uplo[iuplo] ), &M, &N, &offdiag, &diag, h_A, &lda );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_dgetmatrix( M, N, d_A, ldda, h_R, lda );
            
            blasf77_daxpy(&size, &c_neg_one, h_A, &ione, h_R, &ione);
            error = lapackf77_dlange("f", &M, &N, h_R, &lda, work);

            printf("%4c   %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %s\n",
                   lapacke_uplo_const( uplo[iuplo] ), (int) M, (int) N,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   (error == 0. ? "ok" : "failed") );
            status += ! (error == 0.);
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_R );
            
            TESTING_FREE_DEV( d_A );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }
      printf( "\n" );
    }

    TESTING_FINALIZE();
    return status;
}
