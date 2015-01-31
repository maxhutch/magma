/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zgemv_batched.cpp normal z -> d, Fri Jan 30 19:00:26 2015
       @author Mark Gates
       @author Azzam Haidar
       @author Tingxing Dong
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "testings.h"  // before magma.h, to include cublas_v2
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgemm_batched
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    double          magma_error, magma_err, Ynorm, work[1];
    magma_int_t M, N, Xm, Ym, lda, ldda;
    magma_int_t sizeA, sizeX, sizeY;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magma_int_t batchCount;

    double *h_A, *h_X, *h_Y, *h_Ymagma;
    double *d_A, *d_X, *d_Y;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double alpha = MAGMA_D_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    double **A_array = NULL;
    double **X_array = NULL;
    double **Y_array = NULL;


    magma_opts opts;
    parse_opts( argc, argv, &opts );
    batchCount = opts.batchcount;
    opts.lapack |= opts.check; 

    //double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("trans = %s\n", lapack_trans_const(opts.transA) );

    printf("BatchCount    M     N     MAGMA Gflop/s (ms)  CPU Gflop/s (ms)  MAGMA error\n");

    printf("===================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = ((M+31)/32)*32;
            gflops = FLOPS_DGEMV( M, N ) / 1e9 * batchCount;

            if ( opts.transA == MagmaNoTrans ) {
                Xm = N;
                Ym = M;
            } else {
                Xm = M;
                Ym = N;
            }

            sizeA = lda*N*batchCount;
            sizeX = incx*Xm*batchCount;
            sizeY = incy*Ym*batchCount;

            ldda = ((lda+31)/32)*32;

            TESTING_MALLOC_CPU( h_A,  double, sizeA );
            TESTING_MALLOC_CPU( h_X,  double, sizeX );
            TESTING_MALLOC_CPU( h_Y,  double, sizeY  );
            TESTING_MALLOC_CPU( h_Ymagma,  double, sizeY  );


            TESTING_MALLOC_DEV( d_A, double, ldda*N*batchCount );
            TESTING_MALLOC_DEV( d_X, double, sizeX );
            TESTING_MALLOC_DEV( d_Y, double, sizeY );

            magma_malloc((void**)&A_array, batchCount * sizeof(*A_array));
            magma_malloc((void**)&X_array, batchCount * sizeof(*X_array));
            magma_malloc((void**)&Y_array, batchCount * sizeof(*Y_array));

            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_dlarnv( &ione, ISEED, &sizeX, h_X );
            lapackf77_dlarnv( &ione, ISEED, &sizeY, h_Y );
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_dsetmatrix( M, N*batchCount, h_A, lda, d_A, ldda );
            magma_dsetvector( Xm*batchCount, h_X, incx, d_X, incx );
            magma_dsetvector( Ym*batchCount, h_Y, incy, d_Y, incy );
            
            dset_pointer(A_array, d_A, ldda, 0, 0, ldda*N, batchCount, magma_stream);
            dset_pointer(X_array, d_X, 1, 0, 0, incx*Xm, batchCount, magma_stream);
            dset_pointer(Y_array, d_Y, 1, 0, 0, incy*Ym, batchCount, magma_stream);

            magma_time = magma_sync_wtime( NULL );
            magmablas_dgemv_batched(opts.transA, M, N,
                             alpha, A_array, ldda,
                                    X_array, incx,
                             beta,  Y_array, incy, batchCount, magma_stream);
            magma_time = magma_sync_wtime( NULL ) - magma_time;
            magma_perf = gflops / magma_time;            
            magma_dgetvector( Ym*batchCount, d_Y, incy, h_Ymagma, incy );
                      
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                for(int i=0; i<batchCount; i++)
                {
                   blasf77_dgemv(
                               lapack_trans_const(opts.transA), 
                               &M, &N,
                               &alpha, h_A + i*lda*N, &lda,
                                       h_X + i*Xm, &incx,
                               &beta,  h_Y + i*Ym, &incy );
                }
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for both magma  relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                magma_error = 0.0;

                for(int s=0; s<batchCount; s++)
                {

                    Ynorm = lapackf77_dlange( "M", &M, &ione, h_Y + s*Ym, &incy, work );

                    blasf77_daxpy( &Ym, &c_neg_one, h_Y + s*Ym, &ione, h_Ymagma + s*Ym, &ione );
                    magma_err = lapackf77_dlange( "M", &M, &ione, h_Ymagma + s*Ym, &incy, work ) / Ynorm; 

                    if ( isnan(magma_err) || isinf(magma_err) ) {
                      magma_error = magma_err;
                      break;
                    }
                    magma_error = max(fabs(magma_err), magma_error); 

                }

                    printf("%10d %5d %5d  %7.2f (%7.2f)    %7.2f (%7.2f)   %8.2e  \n",
                       (int) batchCount, (int) M, (int) N, 
                       magma_perf,  1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error);
            }
            else {

                    printf("%10d %5d %5d  %7.2f (%7.2f)    ---   (  ---  )    ---\n",
                       (int) batchCount, (int) M, (int) N, 
                       magma_perf,  1000.*magma_time);
            }
            
            TESTING_FREE_CPU( h_A  );
            TESTING_FREE_CPU( h_X  );
            TESTING_FREE_CPU( h_Y  );
            TESTING_FREE_CPU( h_Ymagma  );


            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_X );
            TESTING_FREE_DEV( d_Y );
            TESTING_FREE_DEV( A_array );
            TESTING_FREE_DEV( X_array );
            TESTING_FREE_DEV( Y_array );

            
            fflush( stdout);

        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
