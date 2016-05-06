/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> s d c
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

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlascl
   Code is very similar to testing_zlacpy.cpp
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           error, work[1];
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_R;
    magmaDoubleComplex_ptr d_A;
    double cto, cfrom;
    magma_int_t M, N, size, lda, ldda, info;
    magma_int_t ione     = 1;
    magma_int_t status = 0;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magma_opts opts;
    opts.parse_opts( argc, argv );

    magma_uplo_t uplo[] = { MagmaLower, MagmaUpper, MagmaFull };
    
    double sfmin = lapackf77_dlamch("sfmin");
    double bignum = 1 / sfmin;
    
    printf("%% uplo    M     N    CPU GByte/s (ms)    GPU GByte/s (ms)   check\n");
    printf("%%===================================================================\n");
    for( int iuplo = 0; iuplo < 3; ++iuplo ) {
      for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            //M += 2;  // space for insets
            //N += 2;
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            size   = lda*N;
            if ( uplo[iuplo] == MagmaLower ) {
                // load & save lower trapezoid (with diagonal)
                if ( M > N ) {
                    gbytes = 2. * sizeof(magmaDoubleComplex) * (1.*M*N - 0.5*N*(N-1)) / 1e9;
                } else {
                    gbytes = 2. * sizeof(magmaDoubleComplex) * 0.5*M*(M+1) / 1e9;
                }
            }
            else if ( uplo[iuplo] == MagmaUpper ) {
                // load & save upper trapezoid (with diagonal)
                if ( N > M ) {
                    gbytes = 2. * sizeof(magmaDoubleComplex) * (1.*M*N - 0.5*M*(M-1)) / 1e9;
                } else {
                    gbytes = 2. * sizeof(magmaDoubleComplex) * 0.5*N*(N+1) / 1e9;
                }
            }
            else {
                // load & save entire matrix
                gbytes = 2. * sizeof(magmaDoubleComplex) * 1.*M*N / 1e9;
            }
    
            TESTING_MALLOC_CPU( h_A, magmaDoubleComplex, size   );
            TESTING_MALLOC_CPU( h_R, magmaDoubleComplex, size   );
            
            TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, ldda*N );
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &size, h_A );
            
            if ( 0 ) {
                // test that the over/underflow algorithm is working (but slower)
                // scale by 1e-4
                cto   = 1e-4;
                cfrom = 1;
                lapackf77_zlascl( "G", &ione, &ione, &cfrom, &cto, &M, &N, h_A, &lda, &info );
                assert( info == 0 );
                
                // this (cto/cfrom) is inf,
                // but (1e-4 * cto/cfrom) is ~ 1e308 in double and ~ 1e37 in float
                cto   = 100.*sqrt( bignum );
                cfrom = 1/cto;
            }
            else {
                cto   = 1.2345;
                cfrom = 3.1415;
            }
                        
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zsetmatrix( M, N, h_A, lda, d_A, ldda, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            //magmablas_zlascl( uplo[iuplo], 1, 1, cfrom, cto, M-2, N-2, d_A+1+ldda, ldda, opts.queue, &info );  // inset by 1 row & col
            magmablas_zlascl( uplo[iuplo], 1, 1, cfrom, cto, M, N, d_A, ldda, opts.queue, &info );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (info != 0) {
                printf("magmablas_zlascl returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            //magma_int_t M2 = M-2;  // inset by 1 row & col
            //magma_int_t N2 = N-2;
            //lapackf77_zlascl( lapack_uplo_const( uplo[iuplo] ), &ione, &ione, &cfrom, &cto, &M2, &N2, h_A+1+lda, &lda, &info );
            lapackf77_zlascl( lapack_uplo_const( uplo[iuplo] ), &ione, &ione, &cfrom, &cto, &M, &N, h_A, &lda, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (info != 0) {
                printf("lapackf77_zlascl returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            magma_zgetmatrix( M, N, d_A, ldda, h_R, lda, opts.queue );
            //magma_zprint( M, N, h_R, lda );
            
            blasf77_zaxpy(&size, &c_neg_one, h_A, &ione, h_R, &ione);
            error = lapackf77_zlange("f", &M, &N, h_R, &lda, work);

            printf("%5s %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                   lapack_uplo_const( uplo[iuplo] ), (int) M, (int) N,
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   error, (error == 0. ? "ok" : "failed") );
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

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
