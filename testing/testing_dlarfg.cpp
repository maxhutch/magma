/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal d -> s
       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_d

int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double *h_x, *h_x1, *h_x2, *h_tau;
    double *d_x, *d_tau;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double      error, work[1];
    magma_int_t N, size, nb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    // does larfg on nb columns, one after another
    nb = (opts.nb > 0 ? opts.nb : 64);
    
    magma_queue_t queue = 0;

    printf("    N    nb    CPU GFLop/s (ms)    GPU GFlop/s (ms)    error  \n");
    printf("==============================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[i];
            gflops = FLOPS_DLARFG( N ) / 1e9 * nb;
    
            TESTING_MALLOC_CPU( h_x,   double, N*nb );
            TESTING_MALLOC_CPU( h_x1,  double, N*nb );
            TESTING_MALLOC_CPU( h_x2,  double, N*nb );
            TESTING_MALLOC_CPU( h_tau, double, nb   );
        
            TESTING_MALLOC_DEV( d_x,   double, N*nb );
            TESTING_MALLOC_DEV( d_tau, double, nb   );
            
            /* Initialize the vector */
            size = N*nb;
            lapackf77_dlarnv( &ione, ISEED, &size, h_x );
            blasf77_dcopy( &size, h_x, &ione, h_x1, &ione );
            
            /* =====================================================================
               Performs operation using MAGMA-BLAS
               =================================================================== */
            magma_dsetvector( size, h_x, ione, d_x, ione );
    
            gpu_time = magma_sync_wtime( queue );
            for( int j = 0; j < nb; ++j ) {
                magma_dlarfg( N, &d_x[0+j*N], &d_x[1+j*N], ione, &d_tau[j] );
            }
            gpu_time = magma_sync_wtime( queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            
            magma_dgetvector( size, d_x, ione, h_x2, ione );
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            for( int j = 0; j < nb; ++j ) {
                lapackf77_dlarfg( &N, &h_x1[0+j*N], &h_x1[1+j*N], &ione, &h_tau[j] );
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Error Computation and Performance Compariosn
               =================================================================== */
            blasf77_daxpy( &size, &c_neg_one, h_x1, &ione, h_x2, &ione);
            error = lapackf77_dlange( "F", &N, &nb, h_x2, &N, work );
            printf("%5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2g\n",
                   (int) N, (int) nb, cpu_perf, 1000.*cpu_time, gpu_perf, 1000.*gpu_time, error );
            
            TESTING_FREE_CPU( h_x   );
            TESTING_FREE_CPU( h_x1  );
            TESTING_FREE_CPU( h_x2  );
            TESTING_FREE_CPU( h_tau );
        
            TESTING_FREE_DEV( d_x   );
            TESTING_FREE_DEV( d_tau );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return 0;
}
