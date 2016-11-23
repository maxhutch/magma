/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_ztrmm.cpp, normal z -> d, Sun Nov 20 20:20:33 2016
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
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dtrmm
*/
int main( int argc, char** argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dB(i_, j_)  dB, ((i_) + (j_)*lddb)
    #define dC(i_, j_)  dC, ((i_) + (j_)*lddc)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dC(i_, j_) (dC + (i_) + (j_)*lddc)
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, dev_perf, dev_time, oop_perf=0, oop_time=0, cpu_perf, cpu_time;
    double          dev_error, oop_error=0, work[1];
    magma_int_t M, N;
    magma_int_t Ak;
    magma_int_t sizeA, sizeB;
    magma_int_t lda, ldb, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    double *hA, *hB, *hBdev, *hBoop;
    magmaDouble_ptr dA, dB, dC;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double alpha = MAGMA_D_MAKE(  0.29, -0.86 );
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    // See testing_dgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    printf("%% If running lapack (option --lapack), %s error is computed\n"
           "%% relative to CPU BLAS result.\n\n", g_platform_str );
    printf("%% side = %s, uplo = %s, transA = %s, diag = %s \n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%%   M     N   %s Gflop/s (ms)   Out-of-place Gflop/s (ms)   CPU Gflop/s (ms)  %s error   out-of-place\n",
            g_platform_str, g_platform_str );
    printf("%%====================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_DTRMM(opts.side, M, N) / 1e9;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                Ak = M;
            } else {
                lda = N;
                Ak = N;
            }
            
            ldb = M;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = lddb;
            
            sizeA = lda*Ak;
            sizeB = ldb*N;
            
            TESTING_CHECK( magma_dmalloc_cpu( &hA,    lda*Ak ));
            TESTING_CHECK( magma_dmalloc_cpu( &hB,    ldb*N  ));
            TESTING_CHECK( magma_dmalloc_cpu( &hBdev, ldb*N  ));
            TESTING_CHECK( magma_dmalloc_cpu( &hBoop, ldb*N  ));
            
            TESTING_CHECK( magma_dmalloc( &dA, ldda*Ak ));
            TESTING_CHECK( magma_dmalloc( &dB, lddb*N  ));
            TESTING_CHECK( magma_dmalloc( &dC, lddc*N  ));
            
            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_dlarnv( &ione, ISEED, &sizeB, hB );
            
            // for error checks
            double Anorm = lapackf77_dlantr( "F", lapack_uplo_const(opts.uplo),
                                                  lapack_diag_const(opts.diag),
                                                  &Ak, &Ak, hA, &lda, work );
            double Bnorm = lapackf77_dlange( "F", &M,  &N,  hB, &ldb, work );
            
            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_dsetmatrix( Ak, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            magma_dsetmatrix( M,  N,  hB, ldb, dB(0,0), lddb, opts.queue );
            
            // note cublas does trmm out-of-place (i.e., adds output matrix C),
            // but allows C=B to do in-place.
            dev_time = magma_sync_wtime( opts.queue );
            magma_dtrmm( opts.side, opts.uplo, opts.transA, opts.diag,
                         M, N,
                         alpha, dA(0,0), ldda,
                                dB(0,0), lddb, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_dgetmatrix( M, N, dB(0,0), lddb, hBdev, ldb, opts.queue );
            
            /* =====================================================================
               Performs operation using cuBLAS (out-of-place)
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_dsetmatrix( Ak, Ak, hA, lda, dA(0,0), ldda, opts.queue );
                magma_dsetmatrix( M,  N,  hB, ldb, dB(0,0), lddb, opts.queue );
                
                // cuBLAS does trmm out-of-place (i.e., adds output matrix C),
                // but allows C=B to do in-place.
                oop_time = magma_sync_wtime( opts.queue );
                cublasDtrmm( opts.handle, cublas_side_const(opts.side), cublas_uplo_const(opts.uplo),
                             cublas_trans_const(opts.transA), cublas_diag_const(opts.diag),
                             int(M), int(N),
                             &alpha, dA(0,0), int(ldda),
                                     dB(0,0), int(lddb),
                                     dC(0,0), int(lddc) );  // output C; differs from BLAS standard
                oop_time = magma_sync_wtime( opts.queue ) - oop_time;
                oop_perf = gflops / oop_time;
                
                magma_dgetmatrix( M, N, dC(0,0), lddc, hBoop, ldb, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_dtrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &alpha, hA, &lda,
                                       hB, &ldb );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // See testing_dgemm for formula. Here K = Ak.
                blasf77_daxpy( &sizeB, &c_neg_one, hB, &ione, hBdev, &ione );
                dev_error = lapackf77_dlange( "M", &M, &N, hBdev, &ldb, work )
                            / (sqrt(double(Ak+2))*fabs(alpha)*Anorm*Bnorm);
                
                #ifdef HAVE_CUBLAS
                blasf77_daxpy( &sizeB, &c_neg_one, hB, &ione, hBoop, &ione );
                oop_error = lapackf77_dlange( "M", &M, &N, hBoop, &ldb, work )
                            / (sqrt(double(Ak+2))*fabs(alpha)*Anorm*Bnorm);
                #endif
                
                bool okay = (dev_error < tol && oop_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %8.2e   %s\n",
                       (long long) M, (long long) N,
                       dev_perf, 1000.*dev_time,
                       oop_perf, 1000.*oop_time,
                       cpu_perf, 1000.*cpu_time,
                       dev_error, oop_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    ---   (  ---  )    ---     ---     ---\n",
                       (long long) M, (long long) N,
                       dev_perf, 1000.*dev_time,
                       oop_perf, 1000.*oop_time);
            }
            
            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hBdev );
            magma_free_cpu( hBoop );
            
            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
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
