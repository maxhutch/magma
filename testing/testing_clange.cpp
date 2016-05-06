/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zlange.cpp normal z -> c, Mon May  2 23:31:08 2016
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing clange
*/
int main( int argc, char** argv)
{
    #define h_A(i_, j_) (h_A + (i_) + (j_)*lda)
    
    #ifdef HAVE_clBLAS
    #define d_A(i_, j_)  d_A, ((i_) + (j_)*ldda)
    #else
    #define d_A(i_, j_) (d_A + (i_) + (j_)*ldda)
    #endif
    
    TESTING_INIT();
    
    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaFloatComplex *h_A;
    float *h_work;
    magmaFloatComplex_ptr d_A;
    magmaFloat_ptr d_work;
    magma_int_t i, j, M, N, n2, lda, ldda, lwork;
    magma_int_t idist    = 3;  // normal distribution (otherwise max norm is always ~ 1)
    magma_int_t ISEED[4] = {0,0,0,1};
    float      error, norm_magma, norm_lapack;
    magma_int_t status = 0;
    magma_int_t lapack_nan_fail = 0;
    magma_int_t lapack_inf_fail = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    float tol2;
    
    // Frobenius norm not currently supported, but leave this here for future support
    // of different norms. See similar code in testing_clanhe.cpp.
    magma_norm_t norm[] = { MagmaMaxNorm, MagmaOneNorm, MagmaInfNorm, MagmaFrobeniusNorm };
    
    printf("%%   M     N   norm   CPU GByte/s (ms)    GPU GByte/s (ms)        error               nan      inf\n");
    printf("%%================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      for( int inorm = 0; inorm < 3; ++inorm ) {  /* < 4 for Frobenius */
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M   = opts.msize[itest];
            N   = opts.nsize[itest];
            lda = M;
            n2  = lda*N;
            ldda = magma_roundup( M, opts.align );
            if ( norm[inorm] == MagmaOneNorm )
                lwork = N;
            else
                lwork = M;
            // read whole matrix
            gbytes = M*N*sizeof(magmaFloatComplex) / 1e9;
            
            TESTING_MALLOC_CPU( h_A,    magmaFloatComplex, n2 );
            TESTING_MALLOC_CPU( h_work, float, M );
            
            TESTING_MALLOC_DEV( d_A,    magmaFloatComplex, ldda*N );
            TESTING_MALLOC_DEV( d_work, float, lwork );
            
            /* Initialize the matrix */
            lapackf77_clarnv( &idist, ISEED, &n2, h_A );
            magma_csetmatrix( M, N, h_A, lda, d_A, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            norm_magma = magmablas_clange( norm[inorm], M, N, d_A, ldda, d_work, lwork, opts.queue );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (norm_magma == -1) {
                printf( "%5d   %4c   skipped because %s norm isn't supported\n",
                        (int) N, lapacke_norm_const( norm[inorm] ), lapack_norm_const( norm[inorm] ));
                goto cleanup;
            }
            else if (norm_magma < 0) {
                printf("magmablas_clange returned error %f: %s.\n",
                       norm_magma, magma_strerror( (int) norm_magma ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            norm_lapack = lapackf77_clange( lapack_norm_const(norm[inorm]), &M, &N, h_A, &lda, h_work );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (norm_lapack < 0) {
                printf("lapackf77_clange returned error %f: %s.\n",
                       norm_lapack, magma_strerror( (int) norm_lapack ));
            }
            
            /* =====================================================================
               Check the result compared to LAPACK
               Max norm should be identical; others should be within tolerance.
               =================================================================== */
            error = fabs( norm_magma - norm_lapack ) / norm_lapack;
            tol2 = tol;
            if ( norm[inorm] == MagmaMaxNorm ) {
                // max-norm depends on only one element, so for Real precisions,
                // MAGMA and LAPACK should exactly agree (tol2 = 0),
                // while Complex precisions incur roundoff in cuCabsf.
                #ifdef REAL
                tol2 = 0;
                #endif
            }
            
            bool okay; okay = (error <= tol2);
            status += ! okay;
            
            /* ====================================================================
               Check for NAN and INF propagation
               =================================================================== */
            i = rand() % M;
            j = rand() % N;
            *h_A(i,j) = MAGMA_C_NAN;
            magma_csetvector( 1, h_A(i,j), 1, d_A(i,j), 1, opts.queue );
            norm_magma  = magmablas_clange( norm[inorm], M, N, d_A, ldda, d_work, lwork, opts.queue );
            norm_lapack = lapackf77_clange( lapack_norm_const( norm[inorm] ),
                                            &M, &N, h_A, &lda, h_work );
            bool nan_okay;    nan_okay    = isnan(norm_magma);
            bool la_nan_okay; la_nan_okay = isnan(norm_lapack);
            lapack_nan_fail += ! la_nan_okay;
            status          += !    nan_okay;
            
            *h_A(i,j) = MAGMA_C_INF;
            magma_csetvector( 1, h_A(i,j), 1, d_A(i,j), 1, opts.queue );
            norm_magma  = magmablas_clange( norm[inorm], M, N, d_A, ldda, d_work, lwork, opts.queue );
            norm_lapack = lapackf77_clange( lapack_norm_const( norm[inorm] ),
                                            &M, &N, h_A, &lda, h_work );
            bool inf_okay;    inf_okay    = isinf(norm_magma);
            bool la_inf_okay; la_inf_okay = isinf(norm_lapack);
            lapack_inf_fail += ! la_inf_okay;
            status          += !    inf_okay;
            
            printf("%5d %5d   %4c   %7.2f (%7.2f)   %7.2f (%7.2f)   %#9.3g   %-6s   %6s%1s  %6s%1s\n",
                   (int) M, (int) N,
                   lapacke_norm_const( norm[inorm] ),
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   error,
                   (okay     ? "ok" : "failed"),
                   (nan_okay ? "ok" : "failed"), (la_nan_okay ? " " : "*"),
                   (inf_okay ? "ok" : "failed"), (la_inf_okay ? " " : "*"));
            
        cleanup:
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_DEV( d_A    );
            TESTING_FREE_DEV( d_work );
            fflush( stdout );
        } // end iter
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      } // end inorm
      printf( "\n" );
    }
    
    if ( lapack_nan_fail ) {
        printf( "* Warning: LAPACK did not pass NAN propagation test; upgrade to LAPACK version >= 3.4.2 (Sep. 2012)\n" );
    }
    if ( lapack_inf_fail ) {
        printf( "* Warning: LAPACK did not pass INF propagation test\n" );
    }
    
    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
