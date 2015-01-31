/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zlarfg.cpp normal z -> c, Fri Jan 30 19:00:23 2015
       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_d

int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaFloatComplex *h_x, *h_x2, *h_tau, *h_tau2;
    magmaFloatComplex_ptr d_x, d_tau;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    float      error, error2, work[1];
    magma_int_t N, nb, lda, ldda, size;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};    magma_int_t status = 0;


    magma_opts opts;
    parse_opts( argc, argv, &opts );

    float tol = opts.tolerance * lapackf77_slamch("E");
    
    // does larfg on nb columns, one after another
    nb = (opts.nb > 0 ? opts.nb : 64);
    
    magma_queue_t queue = 0;

    printf("    N    nb    CPU GFLop/s (ms)    GPU GFlop/s (ms)   error      tau error\n");
    printf("==========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda  = N;
            ldda = ((N+31)/32)*32;
            gflops = FLOPS_CLARFG( N ) / 1e9 * nb;
    
            TESTING_MALLOC_CPU( h_x,    magmaFloatComplex, N*nb );
            TESTING_MALLOC_CPU( h_x2,   magmaFloatComplex, N*nb );
            TESTING_MALLOC_CPU( h_tau,  magmaFloatComplex, nb   );
            TESTING_MALLOC_CPU( h_tau2, magmaFloatComplex, nb   );
        
            TESTING_MALLOC_DEV( d_x,   magmaFloatComplex, ldda*nb );
            TESTING_MALLOC_DEV( d_tau, magmaFloatComplex, nb      );
            
            /* Initialize the vectors */
            size = N*nb;
            lapackf77_clarnv( &ione, ISEED, &size, h_x );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_csetmatrix( N, nb, h_x, N, d_x, ldda );
    
            gpu_time = magma_sync_wtime( queue );
            for( int j = 0; j < nb; ++j ) {
                magmablas_clarfg( N, &d_x[0+j*ldda], &d_x[1+j*ldda], ione, &d_tau[j] );
            }
            gpu_time = magma_sync_wtime( queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            
            magma_cgetmatrix( N, nb, d_x, ldda, h_x2, N );
            magma_cgetvector( nb, d_tau, 1, h_tau2, 1 );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            for( int j = 0; j < nb; ++j ) {
                lapackf77_clarfg( &N, &h_x[0+j*lda], &h_x[1+j*lda], &ione, &h_tau[j] );
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Error Computation and Performance Comparison
               =================================================================== */
            blasf77_caxpy( &size, &c_neg_one, h_x, &ione, h_x2, &ione );
            error = lapackf77_clange( "F", &N, &nb, h_x2, &N, work )
                  / lapackf77_clange( "F", &N, &nb, h_x,  &N, work );
            
            // tau can be 0
            blasf77_caxpy( &nb, &c_neg_one, h_tau, &ione, h_tau2, &ione );
            error2 = lapackf77_clange( "F", &nb, &ione, h_tau,  &nb, work );
            if ( error2 != 0 ) {
                error2 = lapackf77_clange( "F", &nb, &ione, h_tau2, &nb, work ) / error2;
            }

            printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %s\n",
                   (int) N, (int) nb, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time,
                   error, error2,
                   (error < tol && error2 < tol ? "ok" : "failed") );
            status += ! (error < tol && error2 < tol);
            
            TESTING_FREE_CPU( h_x   );
            TESTING_FREE_CPU( h_x2  );
            TESTING_FREE_CPU( h_tau );
        
            TESTING_FREE_DEV( d_x   );
            TESTING_FREE_DEV( d_tau );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
