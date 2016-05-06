/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from testing/testing_zungqr.cpp normal z -> c, Mon May  2 23:31:17 2016

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
   -- Testing cungqr
*/
int main( int argc, char** argv )
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    float           Anorm, error, work[1];
    magmaFloatComplex  c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex *hA, *hR, *tau, *h_work, *hT;
    magmaFloatComplex_ptr dA, dT;
    magma_int_t m, n, k;
    magma_int_t n2, lda, ldda, lwork, min_mn, nb, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );

    float tol = opts.tolerance * lapackf77_slamch("E");
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% version %d, ngpu %d\n", int(opts.version), int(abs_ngpu) );
    printf("%% Available versions:\n");
    printf("%%   1 - uses precomputed clarft matrices (default)\n");
    printf("%%   2 - recomputes the clarft matrices on the fly\n\n");

    printf("%%   m     n     k   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R|| / ||A||\n");
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m = opts.msize[itest];
            n = opts.nsize[itest];
            k = opts.ksize[itest];
            if ( m < n || n < k ) {
                printf( "%5d %5d %5d   skipping because m < n or n < k\n", (int) m, (int) n, (int) k );
                continue;
            }
            
            lda  = m;
            ldda = magma_roundup( m, opts.align );  // multiple of 32 by default
            n2 = lda*n;
            min_mn = min(m, n);
            nb = magma_get_cgeqrf_nb( m, n );
            lwork  = n*nb;
            gflops = FLOPS_CUNGQR( m, n, k ) / 1e9;
            
            TESTING_MALLOC_PIN( hR,     magmaFloatComplex, lda*n  );
            
            TESTING_MALLOC_CPU( hA,     magmaFloatComplex, lda*n  );
            TESTING_MALLOC_CPU( tau,    magmaFloatComplex, min_mn );
            TESTING_MALLOC_CPU( h_work, magmaFloatComplex, lwork  );
            TESTING_MALLOC_CPU( hT,     magmaFloatComplex, min_mn*nb );
            
            TESTING_MALLOC_DEV( dA,     magmaFloatComplex, ldda*n );
            TESTING_MALLOC_DEV( dT,     magmaFloatComplex, ( 2*min_mn + magma_roundup( n, 32 ) )*nb );
            
            lapackf77_clarnv( &ione, ISEED, &n2, hA );
            lapackf77_clacpy( MagmaFullStr, &m, &n, hA, &lda, hR, &lda );
            
            Anorm = lapackf77_clange("f", &m, &n, hA, &lda, work );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            // first, get QR factors in both hA and hR
            // okay that magma_cgeqrf_gpu has special structure for R; R isn't used here.
            magma_csetmatrix( m, n, hA, lda, dA, ldda, opts.queue );
            magma_cgeqrf_gpu( m, n, dA, ldda, tau, dT, &info );
            if (info != 0) {
                printf("magma_cgeqrf_gpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            magma_cgetmatrix( m, n, dA, ldda, hA, lda, opts.queue );
            lapackf77_clacpy( MagmaFullStr, &m, &n, hA, &lda, hR, &lda );
            magma_cgetmatrix( nb, min_mn, dT, nb, hT, nb, opts.queue );  // for multi GPU
            
            gpu_time = magma_wtime();
            if (opts.version == 1) {
                if (opts.ngpu == 1) {
                    magma_cungqr( m, n, k, hR, lda, tau, dT, nb, &info );
                }
                else {
                    magma_cungqr_m( m, n, k, hR, lda, tau, hT, nb, &info );
                }
            }
            else {
                magma_cungqr2( m, n, k, hR, lda, tau, &info );
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_cungqr returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_cungqr( &m, &n, &k, hA, &lda, tau, h_work, &lwork, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_cungqr returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                
                // compute relative error |R|/|A| := |Q_magma - Q_lapack|/|A|
                blasf77_caxpy( &n2, &c_neg_one, hA, &ione, hR, &ione );
                error = lapackf77_clange("f", &m, &n, hR, &lda, work) / Anorm;
                
                bool okay = (error < tol);
                status += ! okay;
                printf("%5d %5d %5d   %7.1f (%7.2f)   %7.1f (%7.2f)   %8.2e   %s\n",
                       (int) m, (int) n, (int) k,
                       cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%5d %5d %5d     ---   (  ---  )   %7.1f (%7.2f)     ---  \n",
                       (int) m, (int) n, (int) k,
                       gpu_perf, gpu_time );
            }
            
            TESTING_FREE_PIN( hR     );
            
            TESTING_FREE_CPU( hA  );
            TESTING_FREE_CPU( tau );
            TESTING_FREE_CPU( h_work );
            TESTING_FREE_CPU( hT  );
            
            TESTING_FREE_DEV( dA );
            TESTING_FREE_DEV( dT );
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
