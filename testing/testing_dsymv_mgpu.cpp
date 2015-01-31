/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zhemv_mgpu.cpp normal z -> d, Fri Jan 30 19:00:23 2015
       
       @author Mark Gates
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

#include "magma_operators.h"

#define PRECISION_d


// --------------------
int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t gflops, cpu_time=0, cpu_perf=0, gpu_time, gpu_perf, mgpu_time, mgpu_perf, cuda_time, cuda_perf;
    double      error=0, error2=0, work[1];
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t n_local[MagmaMaxGPUs];

    magma_int_t N, Noffset, lda, ldda, blocks, lhwork, ldwork, matsize, vecsize;
    magma_int_t incx = 1;

    double alpha = MAGMA_D_MAKE(  1.5, -2.3 );
    double beta  = MAGMA_D_MAKE( -0.6,  0.8 );
    double *A, *X, *Y, *Ylapack, *Ycublas, *Ymagma, *Ymagma1, *hwork;
    magmaDouble_ptr dA, dX, dY;
    magmaDouble_ptr d_lA[MagmaMaxGPUs], dwork[MagmaMaxGPUs];

    magma_device_t dev;
    magma_queue_t queues[MagmaMaxGPUs];
    magma_int_t     status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    magma_int_t nb = 64;  // required by magmablas_dsymv_mgpu implementation

    for( dev=0; dev < opts.ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_queue_create( &queues[dev] );
    }
    
    // currently, tests all offsets in the offsets array;
    // comment out loop below to test a specific offset.
    magma_int_t offset = opts.offset;
    magma_int_t offsets[] = { 0, 1, 31, 32, 33, 63, 64, 65, 100, 200 };
    magma_int_t noffsets = sizeof(offsets) / sizeof(*offsets);
    
    printf("uplo = %s, ngpu %d, block size = %d, offset %d\n",
            lapack_uplo_const(opts.uplo), (int) opts.ngpu, (int) nb, (int) offset );
    printf( "                  BLAS                CUBLAS              MAGMA 1 GPU         MAGMA MGPU       Error rel  Error rel\n"
            "    N  offset     Gflop/s (msec)      Gflop/s (msec)      Gflop/s (msec)      Gflop/s (msec)   to CUBLAS  to LAPACK\n"
            "===================================================================================================================\n" );
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      
      // comment out these two lines & end of loop to test a specific offset
      for( int ioffset=0; ioffset < noffsets; ioffset += 1 ) {
        offset = offsets[ioffset];
        
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N       = opts.nsize[itest];
            Noffset = N + offset;
            lda     = Noffset;
            ldda    = ((Noffset+31)/32)*32;
            matsize = Noffset*ldda;
            vecsize = (Noffset-1)*incx + 1;
            gflops  = FLOPS_DSYMV( N ) / 1e9;
            
            blocks = (N + (offset % nb) - 1)/nb + 1;
            lhwork = N*opts.ngpu;
            ldwork = ldda*(blocks + 1);

            TESTING_MALLOC_CPU( A,       double, matsize );
            TESTING_MALLOC_CPU( Y,       double, vecsize );
            TESTING_MALLOC_CPU( Ycublas, double, vecsize );
            TESTING_MALLOC_CPU( Ymagma,  double, vecsize );
            TESTING_MALLOC_CPU( Ymagma1, double, vecsize );
            TESTING_MALLOC_CPU( Ylapack, double, vecsize );

            TESTING_MALLOC_PIN( X,       double, vecsize );
            TESTING_MALLOC_PIN( hwork,   double, lhwork  );
            
            magma_setdevice( opts.device );
            TESTING_MALLOC_DEV( dA, double, matsize );
            TESTING_MALLOC_DEV( dX, double, vecsize );
            TESTING_MALLOC_DEV( dY, double, vecsize );
            
            // TODO make magma_dmalloc_bcyclic helper function?
            for( dev=0; dev < opts.ngpu; dev++ ) {
                n_local[dev] = ((Noffset/nb)/opts.ngpu)*nb;
                if (dev < (Noffset/nb) % opts.ngpu)
                    n_local[dev] += nb;
                else if (dev == (Noffset/nb) % opts.ngpu)
                    n_local[dev] += Noffset % nb;
                
                magma_setdevice( dev );
                TESTING_MALLOC_DEV( d_lA[dev],  double, ldda*n_local[dev] );
                TESTING_MALLOC_DEV( dwork[dev], double, ldwork );
            }
            
            //////////////////////////////////////////////////////////////////////////
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &matsize, A );
            magma_dmake_symmetric( Noffset, A, lda );
            
            lapackf77_dlarnv( &ione, ISEED, &vecsize, X );
            lapackf77_dlarnv( &ione, ISEED, &vecsize, Y );
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_setdevice( opts.device );
            magma_dsetmatrix( Noffset, Noffset, A, lda, dA, ldda );
            magma_dsetvector( Noffset, X, incx, dX, incx );
            magma_dsetvector( Noffset, Y, incx, dY, incx );
            
            cuda_time = magma_sync_wtime(0);
            cublasDsymv( opts.handle, cublas_uplo_const(opts.uplo), N,
                         &alpha, dA + offset + offset*ldda, ldda,
                                 dX + offset, incx,
                         &beta,  dY + offset, incx );
            cuda_time = magma_sync_wtime(0) - cuda_time;
            cuda_perf = gflops / cuda_time;
            
            magma_dgetvector( Noffset, dY, incx, Ycublas, incx );
            
            /* =====================================================================
               Performs operation using MAGMABLAS (1 GPU)
               =================================================================== */
            magma_setdevice( opts.device );
            magma_dsetvector( Noffset, Y, incx, dY, incx );
            
            gpu_time = magma_sync_wtime( opts.queue );
            
            magmablas_dsymv_work( opts.uplo, N,
                                  alpha, dA + offset + offset*ldda, ldda,
                                         dX + offset, incx,
                                  beta,  dY + offset, incx, dwork[ opts.device ], ldwork,
                                  opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            magma_dgetvector( Noffset, dY, incx, Ymagma1, incx );
            
            /* =====================================================================
               Performs operation using MAGMABLAS (multi-GPU)
               =================================================================== */
            magma_dsetmatrix_1D_col_bcyclic( Noffset, Noffset, A, lda, d_lA, ldda, opts.ngpu, nb );
            blasf77_dcopy( &Noffset, Y, &incx, Ymagma, &incx );
            
            // workspaces do NOT need to be zero -- set to NAN to prove
            for( dev=0; dev < opts.ngpu; ++dev ) {
                magma_setdevice( dev );
                magmablas_dlaset( MagmaFull, ldwork, 1, MAGMA_D_NAN, MAGMA_D_NAN, dwork[dev], ldwork );
            }
            lapackf77_dlaset( "Full", &lhwork, &ione, &MAGMA_D_NAN, &MAGMA_D_NAN, hwork, &lhwork );
            
            mgpu_time = magma_sync_wtime(0);
            
            magma_int_t info;
            info = magmablas_dsymv_mgpu(
                opts.uplo, N,
                alpha,
                d_lA, ldda, offset,
                X + offset, incx,
                beta,
                Ymagma + offset, incx,
                hwork, lhwork,
                dwork, ldwork,
                opts.ngpu, nb, queues );
            if ( info != 0 )
                printf("magmablas_dsymv_mgpu returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            info = magmablas_dsymv_mgpu_sync(
                opts.uplo, N,
                alpha,
                d_lA, ldda, offset,
                X + offset, incx,
                beta,
                Ymagma + offset, incx,
                hwork, lhwork,
                dwork, ldwork,
                opts.ngpu, nb, queues );
            if ( info != 0 )
                printf("magmablas_dsymv_sync returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            mgpu_time = magma_sync_wtime(0) - mgpu_time;
            mgpu_perf = gflops / mgpu_time;
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                blasf77_dcopy( &Noffset, Y, &incx, Ylapack, &incx );
                
                cpu_time = magma_wtime();
                blasf77_dsymv( lapack_uplo_const(opts.uplo), &N,
                               &alpha, A + offset + offset*lda, &lda,
                                       X + offset, &incx,
                               &beta,  Ylapack + offset, &incx );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
    
                /* =====================================================================
                   Compute the Difference LAPACK vs. Magma
                   =================================================================== */
                error2 = lapackf77_dlange( "F", &Noffset, &ione, Ylapack, &Noffset, work );
                blasf77_daxpy( &Noffset, &c_neg_one, Ymagma, &incx, Ylapack, &incx );
                error2 = lapackf77_dlange( "F", &Noffset, &ione, Ylapack, &Noffset, work ) / error2;
            }
            
            /* =====================================================================
               Compute the Difference Cublas vs. Magma
               =================================================================== */            
            error = lapackf77_dlange( "F", &Noffset, &ione, Ycublas, &Noffset, work );
            blasf77_daxpy( &Noffset, &c_neg_one, Ymagma, &incx, Ycublas, &incx );
            error = lapackf77_dlange( "F", &Noffset, &ione, Ycublas, &Noffset, work ) / error;
            
            bool okay = (error < tol && error2 < tol);
            status += ! okay;
            if ( opts.lapack ) {
                printf( "%5d  %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %s\n",
                        (int) N, (int) offset,
                         cpu_perf,  cpu_time*1000.,
                        cuda_perf, cuda_time*1000.,
                         gpu_perf,  gpu_time*1000.,
                        mgpu_perf, mgpu_time*1000.,
                        error, error2, (okay ? "ok" : "failed") );
            }
            else {
                printf( "%5d  %5d     ---   (  ---  )   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e     ---      %s\n",
                        (int) N, (int) offset,
                        cuda_perf, cuda_time*1000.,
                         gpu_perf,  gpu_time*1000.,
                        mgpu_perf, mgpu_time*1000.,
                        error, (okay ? "ok" : "failed") );
            }
            
            /* Free Memory */
            TESTING_FREE_CPU( A );
            TESTING_FREE_CPU( Y );
            TESTING_FREE_CPU( Ycublas );
            TESTING_FREE_CPU( Ymagma  );
            TESTING_FREE_CPU( Ymagma1 );
            TESTING_FREE_CPU( Ylapack );

            TESTING_FREE_PIN( X );
            TESTING_FREE_PIN( hwork   );
            
            magma_setdevice( opts.device );
            TESTING_FREE_DEV( dA );
            TESTING_FREE_DEV( dX );
            TESTING_FREE_DEV( dY );
            
            for( dev=0; dev < opts.ngpu; dev++ ) {
                magma_setdevice( dev );
                TESTING_FREE_DEV( d_lA[dev]  );
                TESTING_FREE_DEV( dwork[dev] );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
        
      // comment out these two lines line & top of loop test a specific offset
      }  // end for ioffset
      printf( "\n" );
      
    }
    
    for( dev=0; dev < opts.ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_queue_destroy( queues[dev] );
    }
    
    TESTING_FINALIZE();
    return status;
}
