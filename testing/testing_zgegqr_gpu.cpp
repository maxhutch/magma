/*
    -- MAGMA (version 1.5.0-beta1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date April 2014

       @precisions normal z -> s d c
       @author Stan Tomov

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgegqr
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double           e1, e2, e3, e4, e5, *work;
    magmaDoubleComplex  c_neg_one = MAGMA_Z_NEG_ONE, one = MAGMA_Z_ONE, zero = MAGMA_Z_ZERO;
    magmaDoubleComplex *h_A, *h_R, *tau, *dtau, *h_work, *h_rwork, tmp[1];

    magmaDoubleComplex *d_A, *dwork, *ddA, *d_T;
    magma_int_t M, N, n2, lda, ldda, lwork, info, min_mn;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    printf("  M     N     CPU GFlop/s (ms)    GPU GFlop/s (ms)        ||I-Q'Q||_F     ||I-Q'Q||_I     ||A-Q R||_I \n");
    printf("=====================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = ((M+31)/32)*32;
            gflops = FLOPS_ZGEQRF( M, N ) / 1e9 +  FLOPS_ZUNGQR( M, N, N ) / 1e9;
            
            // query for workspace size
            lwork = -1;
            lapackf77_zgeqrf(&M, &N, NULL, &M, NULL, tmp, &lwork, &info);
            lwork = (magma_int_t)MAGMA_Z_REAL( tmp[0] );
            lwork = max(lwork, 3*N*N);
            
            TESTING_MALLOC_PIN( tau,    magmaDoubleComplex, min_mn );
            TESTING_MALLOC_PIN( h_work, magmaDoubleComplex, lwork  );
            TESTING_MALLOC_PIN(h_rwork, magmaDoubleComplex, lwork  );            

            TESTING_MALLOC_CPU( h_A,   magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( h_R,   magmaDoubleComplex, n2     );
            TESTING_MALLOC_CPU( work, double,             M       ); 
            
            TESTING_MALLOC_DEV( d_A,   magmaDoubleComplex, ldda*N );
            TESTING_MALLOC_DEV( dtau,  magmaDoubleComplex, min_mn );
            TESTING_MALLOC_DEV( dwork, magmaDoubleComplex, N*N    );
            TESTING_MALLOC_DEV( ddA,   magmaDoubleComplex, N*N    );
            TESTING_MALLOC_DEV( d_T,   magmaDoubleComplex, N*N    );
            
            magmablas_zlaset( MagmaFull, N, N, ddA, N );
            magmablas_zlaset( MagmaFull, N, N, d_T, N );

            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &n2, h_A );

            lapackf77_zlacpy( MagmaUpperLowerStr, &M, &N, h_A, &lda, h_R, &lda );
            magma_zsetmatrix( M, N, h_R, lda, d_A, ldda );
            
            // warmup
            magma_zgegqr_gpu( 1, M, N, d_A, ldda, dwork, h_work, &info );
            magma_zsetmatrix( M, N, h_R, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_sync_wtime( 0 );
            if (opts.version == 2) {
                int min_mn = min(M, N);
                int     nb = N;

                magmaDoubleComplex *dtau = dwork;
                
                magma_zgeqr2x3_gpu(&M, &N, d_A, &ldda, dtau, d_T, ddA, 
                                   (double *)(dwork+min_mn), &info);
                magma_zgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn);  
                magma_zungqr_gpu( M, N, N, d_A, ldda, tau, d_T, nb, &info );
            }
            else
                magma_zgegqr_gpu( 1, M, N, d_A, ldda, dwork, h_rwork, &info );
            gpu_time = magma_sync_wtime( 0 ) - gpu_time;

            gpu_perf = gflops / gpu_time;
            if (info != 0)
                printf("magma_zgegqr returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));

            magma_zgetmatrix( M, N, d_A, ldda, h_R, M );

            // Regenerate R
            // blasf77_zgemm("t", "n", &N, &N, &M, &one, h_R, &M, h_A, &M, &zero, h_work, &N); 
            
            //magma_zprint(N, N, h_work, N);

            blasf77_ztrmm("r", "u", "n", "n", &M, &N, &one, h_rwork, &N, h_R, &M);
            blasf77_zaxpy( &n2, &c_neg_one, h_A, &ione, h_R, &ione );
            e5 = lapackf77_zlange("i", &M, &N, h_R, &M, work) / 
            lapackf77_zlange("i", &M, &N, h_A, &lda, work);
            magma_zgetmatrix( M, N, d_A, ldda, h_R, M );
 
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();

                /* Orthogonalize on the CPU */
                lapackf77_zgeqrf(&M, &N, h_A, &lda, tau, h_work, &lwork, &info);
                lapackf77_zungqr(&M, &N, &N, h_A, &lda, tau, h_work, &lwork, &info );

                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_zungqr returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                blasf77_zgemm("t", "n", &N, &N, &M, &one, h_R, &M, h_R, &M, &zero, h_work, &N);
                for(int ii=0; ii<N*N; ii+=(N+1)) h_work[ii] = MAGMA_Z_SUB(h_work[ii], one);

                e1    = lapackf77_zlange("f", &N, &N, h_work, &N, work);
                e3    = lapackf77_zlange("i", &N, &N, h_work, &N, work);

                blasf77_zgemm("t", "n", &N, &N, &M, &one, h_A, &M, h_A, &M, &zero, h_work, &N);
                for(int ii=0; ii<N*N; ii+=(N+1)) h_work[ii] = MAGMA_Z_SUB(h_work[ii], one);
                e2    = lapackf77_zlange("f", &N, &N, h_work, &N, work);
                e4    = lapackf77_zlange("i", &N, &N, h_work, &N, work);

                printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2e/%7.2e  %7.2e/%7.2e  %7.2e\n",
                       (int) M, (int) N, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time, e1, e2,
                       e3, e4, e5);
            }
            else {
                printf("%5d %5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (int) M, (int) N, gpu_perf, 1000.*gpu_time );
            }
            
            TESTING_FREE_PIN( tau    );
            TESTING_FREE_PIN( h_work );
            TESTING_FREE_PIN( h_rwork );
           
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_R  );
            TESTING_FREE_CPU( work );            

            TESTING_FREE_DEV( d_A   );
            TESTING_FREE_DEV( dtau  );
            TESTING_FREE_DEV( dwork );
            TESTING_FREE_DEV( ddA   );
            TESTING_FREE_DEV( d_T   );

            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return 0;
}
