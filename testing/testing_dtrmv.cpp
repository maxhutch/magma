/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016
 
       @generated from testing/testing_ztrmv.cpp, normal z -> d, Sun Nov 20 20:20:33 2016
       @author Chongxiao Cao
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dtrmv
*/
int main( int argc, char** argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dx(i_)      dx, ((i_))
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dx(i_)     (dx + (i_))
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, dev_perf, dev_time, cpu_perf, cpu_time;
    double          dev_error, Cnorm, work[1];
    magma_int_t N;
    magma_int_t Ak;
    magma_int_t sizeA;
    magma_int_t lda, ldda;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    double *hA, *hx, *hxdev;
    magmaDouble_ptr dA, dx;
    double c_neg_one = MAGMA_D_NEG_ONE;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% If running lapack (option --lapack), CUBLAS error is computed\n"
           "%% relative to CPU BLAS result.\n\n");
    printf("%% uplo = %s, transA = %s, diag = %s \n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA),
           lapack_diag_const(opts.diag) );
    printf("%%   N   CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  CUBLAS error\n");
    printf("%%=================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            gflops = FLOPS_DTRMM(opts.side, N, 1) / 1e9;

            lda = N;
            Ak = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            
            sizeA = lda*Ak;
            
            TESTING_CHECK( magma_dmalloc_cpu( &hA,    lda*Ak ));
            TESTING_CHECK( magma_dmalloc_cpu( &hx,    N      ));
            TESTING_CHECK( magma_dmalloc_cpu( &hxdev, N      ));
            
            TESTING_CHECK( magma_dmalloc( &dA, ldda*Ak ));
            TESTING_CHECK( magma_dmalloc( &dx, N       ));
            
            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_dlarnv( &ione, ISEED, &N, hx );
            
            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_dsetmatrix( Ak, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            magma_dsetvector( N, hx, 1, dx(0), 1, opts.queue );
            
            dev_time = magma_sync_wtime( opts.queue );
            magma_dtrmv( opts.uplo, opts.transA, opts.diag,
                         N,
                         dA(0,0), ldda,
                         dx(0),   1, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_dgetvector( N, dx(0), 1, hxdev, 1, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_dtrmv( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &N,
                               hA, &lda,
                               hx, &ione );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute relative error for both magma & cuBLAS/clBLAS, relative to lapack,
                // |C_magma - C_lapack| / |C_lapack|
                Cnorm = lapackf77_dlange( "M", &N, &ione, hx, &N, work );
                
                blasf77_daxpy( &N, &c_neg_one, hx, &ione, hxdev, &ione );
                dev_error = lapackf77_dlange( "M", &N, &ione, hxdev, &N, work ) / Cnorm;
                
                bool okay = (dev_error < tol);
                status += ! okay;
                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                       (long long) N,
                       dev_perf, 1000.*dev_time,
                       cpu_perf, 1000.*cpu_time,
                       dev_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%5lld   %7.2f (%7.2f)    ---   (  ---  )    ---     ---\n",
                       (long long) N,
                       dev_perf, 1000.*dev_time);
            }
            
            magma_free_cpu( hA );
            magma_free_cpu( hx );
            magma_free_cpu( hxdev );
            
            magma_free( dA );
            magma_free( dx );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
