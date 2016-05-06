/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zungbr.cpp normal z -> c, Mon May  2 23:31:20 2016

       @author Stan Tomov
       @author Mathieu Faverge
       @author Mark Gates
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cungbr
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float           Anorm, error, work[1];
    magmaFloatComplex  c_neg_one = MAGMA_C_NEG_ONE;
    float *d, *e;
    magmaFloatComplex *hA, *hR, *tauq, *taup, *h_work;
    magma_int_t m, n, k;
    magma_int_t n2, lda, lwork, min_mn, nb, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magma_vect_t vect;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );

    float tol = opts.tolerance * lapackf77_slamch("E");
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    magma_vect_t vects[] = { MagmaQ, MagmaP };
    
    printf("%% Q/P   m     n     k   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R|| / ||A||\n");
    printf("%%============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      for( int ivect = 0; ivect < 2; ++ivect ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m = opts.msize[itest];
            n = opts.nsize[itest];
            k = opts.ksize[itest];
            vect = vects[ivect];
            
            if ( (vect == MagmaQ && (m < n || n < min(m,k))) ||
                 (vect == MagmaP && (n < m || m < min(n,k))) )
            {
                printf( "%3c %5d %5d %5d   skipping invalid dimensions\n",
                        lapacke_vect_const(vect), (int) m, (int) n, (int) k );
                continue;
            }
            
            lda = m;
            n2 = lda*n;
            min_mn = min(m, n);
            nb = max( magma_get_cgelqf_nb( m, n ),
                      magma_get_cgebrd_nb( m, n ));
            lwork  = max( nb*nb, (m + n)*nb );
            
            if (vect == MagmaQ) {
                if (m >= k) {
                    gflops = FLOPS_CUNGQR( m, n, k ) / 1e9;
                } else {
                    gflops = FLOPS_CUNGQR( m-1, m-1, m-1 ) / 1e9;
                }
            }
            else {
                if (k < n) {
                    gflops = FLOPS_CUNGLQ( m, n, k ) / 1e9;
                } else {
                    gflops = FLOPS_CUNGLQ( n-1, n-1, n-1 ) / 1e9;
                }
            }
            
            TESTING_MALLOC_PIN( h_work, magmaFloatComplex, lwork  );
            TESTING_MALLOC_PIN( hR,     magmaFloatComplex, lda*n  );
            
            TESTING_MALLOC_CPU( hA,     magmaFloatComplex, lda*n  );
            TESTING_MALLOC_CPU( tauq,   magmaFloatComplex, min_mn );
            TESTING_MALLOC_CPU( taup,   magmaFloatComplex, min_mn );
            TESTING_MALLOC_CPU( d,      float, min_mn   );
            TESTING_MALLOC_CPU( e,      float, min_mn-1 );
            
            lapackf77_clarnv( &ione, ISEED, &n2, hA );
            lapackf77_clacpy( MagmaFullStr, &m, &n, hA, &lda, hR, &lda );
            
            Anorm = lapackf77_clange("f", &m, &n, hA, &lda, work );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            // first, get GEBRD factors in both hA and hR
            magma_cgebrd( m, n, hA, lda, d, e, tauq, taup, h_work, lwork, &info );
            if (info != 0) {
                printf("magma_cgelqf_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            lapackf77_clacpy( MagmaFullStr, &m, &n, hA, &lda, hR, &lda );
            
            gpu_time = magma_wtime();
            if (vect == MagmaQ) {
                magma_cungbr( vect, m, n, k, hR, lda, tauq, h_work, lwork, &info );
            }
            else {
                magma_cungbr( vect, m, n, k, hR, lda, taup, h_work, lwork, &info );
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_cungbr returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                if (vect == MagmaQ) {
                    lapackf77_cungbr( lapack_vect_const(vect), &m, &n, &k,
                                      hA, &lda, tauq, h_work, &lwork, &info );
                }
                else {
                    lapackf77_cungbr( lapack_vect_const(vect), &m, &n, &k,
                                      hA, &lda, taup, h_work, &lwork, &info );
                }
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_cungbr returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                
                if ( opts.verbose ) {
                    printf( "R=" );  magma_cprint( m, n, hR, lda );
                    printf( "A=" );  magma_cprint( m, n, hA, lda );
                }
                
                // compute relative error |R|/|A| := |Q_magma - Q_lapack|/|A|
                blasf77_caxpy( &n2, &c_neg_one, hA, &ione, hR, &ione );
                error = lapackf77_clange("f", &m, &n, hR, &lda, work) / Anorm;
                
                if ( opts.verbose ) {
                    printf( "diff=" );  magma_cprint( m, n, hR, lda );
                }
                
                bool okay = (error < tol);
                status += ! okay;
                printf("%3c %5d %5d %5d   %7.1f (%7.2f)   %7.1f (%7.2f)   %8.2e   %s\n",
                       lapacke_vect_const(vect), (int) m, (int) n, (int) k,
                       cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%3c %5d %5d %5d     ---   (  ---  )   %7.1f (%7.2f)     ---  \n",
                       lapacke_vect_const(vect), (int) m, (int) n, (int) k,
                       gpu_perf, gpu_time );
            }
            
            TESTING_FREE_PIN( h_work );
            TESTING_FREE_PIN( hR     );
            
            TESTING_FREE_CPU( hA   );
            TESTING_FREE_CPU( tauq );
            TESTING_FREE_CPU( taup );
            TESTING_FREE_CPU( d );
            TESTING_FREE_CPU( e );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }
    }
    
    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
