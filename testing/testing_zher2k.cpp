/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @precisions normal z -> c d s
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

#define COMPLEX


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zher2k
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

    real_Double_t   gflops, dev_perf, dev_time, cpu_perf, cpu_time;
    double          dev_error, work[1];
    magma_int_t N, K;
    magma_int_t Ak, An, Bk, Bn;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magmaDoubleComplex *hA, *hB, *hC, *hCdev;
    magmaDoubleComplex_ptr dA, dB, dC;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    // See testing_zgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    #ifdef COMPLEX
    if (opts.transA == MagmaTrans) {
        opts.transA = MagmaConjTrans; 
        printf("%% WARNING: transA = MagmaTrans changed to MagmaConjTrans\n");
    }
    #endif
    
    printf("%% If running lapack (option --lapack), CUBLAS error is computed\n"
           "%% relative to CPU BLAS result.\n\n");
    printf("%% uplo = %s, transA = %s\n",
           lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA) );
    printf("%%   N     K   CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  CUBLAS error\n");
    printf("%%=================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.msize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_ZHER2K(K, N) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                Ak = K;
                ldb = Bn = N;
                Bk = K;
            } else {
                lda = An = K;
                Ak = N;
                ldb = Bn = K;
                Bk = N;
            }
            
            ldc = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            
            sizeA = lda*Ak;
            sizeB = ldb*Ak;
            sizeC = ldc*N;
            
            TESTING_CHECK( magma_zmalloc_cpu( &hA,    lda*Ak ));
            TESTING_CHECK( magma_zmalloc_cpu( &hB,    ldb*Bk ));
            TESTING_CHECK( magma_zmalloc_cpu( &hC,    ldc*N  ));
            TESTING_CHECK( magma_zmalloc_cpu( &hCdev, ldc*N  ));
            
            TESTING_CHECK( magma_zmalloc( &dA, ldda*Ak ));
            TESTING_CHECK( magma_zmalloc( &dB, lddb*Bk ));
            TESTING_CHECK( magma_zmalloc( &dC, lddc*N  ));
            
            /* Initialize the matrices */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, hB );
            lapackf77_zlarnv( &ione, ISEED, &sizeC, hC );
            
            // for error checks
            double Anorm = lapackf77_zlange( "F", &An, &Ak, hA, &lda, work );
            double Bnorm = lapackf77_zlange( "F", &Bn, &Bk, hB, &ldb, work );
            double Cnorm = safe_lapackf77_zlanhe( "F", lapack_uplo_const(opts.uplo), &N, hC, &ldc, work );
            
            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_zsetmatrix( An, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            magma_zsetmatrix( Bn, Bk, hB, ldb, dB(0,0), lddb, opts.queue );
            magma_zsetmatrix( N, N, hC, ldc, dC(0,0), lddc, opts.queue );
            
            dev_time = magma_sync_wtime( opts.queue );
            magma_zher2k( opts.uplo, opts.transA, N, K,
                          alpha, dA(0,0), ldda,
                                 dB(0,0), lddb,
                          beta,  dC(0,0), lddc, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_zgetmatrix( N, N, dC(0,0), lddc, hCdev, ldc, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_zher2k( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), &N, &K,
                               &alpha, hA, &lda,
                                       hB, &ldb,
                               &beta,  hC, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // See testing_zgemm for formula.
                // There are two multiplies, A*B^H + B*A^H, so double it.
                blasf77_zaxpy( &sizeC, &c_neg_one, hC, &ione, hCdev, &ione );
                dev_error = safe_lapackf77_zlanhe( "F", lapack_uplo_const(opts.uplo), &N, hCdev, &ldc, work )
                            / (2*sqrt(double(K+2))*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);
                
                bool okay = (dev_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                       (long long) N, (long long) K,
                       dev_perf, 1000.*dev_time,
                       cpu_perf, 1000.*cpu_time,
                       dev_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("%5lld %5lld   %7.2f (%7.2f)    ---   (  ---  )    ---     ---\n",
                       (long long) N, (long long) K,
                       dev_perf, 1000.*dev_time);
            }
            
            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hC );
            magma_free_cpu( hCdev );
            
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
