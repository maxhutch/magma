/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zherk.cpp, normal z -> s, Sun Nov 20 20:20:33 2016
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

#define REAL


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ssyrk
*/
int main( int argc, char** argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dC(i_, j_)  dC, ((i_) + (j_)*lddc)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dC(i_, j_) (dC + (i_) + (j_)*lddc)
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, dev_perf, dev_time, cpu_perf, cpu_time;
    float          dev_error, work[1];
    magma_int_t N, K;
    magma_int_t Ak, An;
    magma_int_t sizeA, sizeC;
    magma_int_t lda, ldc, ldda, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    float *hA, *hC, *hCdev;
    magmaFloat_ptr dA, dC;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_D_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    // See testing_sgemm about tolerance.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;
    
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
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = FLOPS_SSYRK(K, N) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                Ak = K;
            } else {
                lda = An = K;
                Ak = N;
            }
            
            ldc = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default
            
            sizeA = lda*Ak;
            sizeC = ldc*N;
            
            TESTING_CHECK( magma_smalloc_cpu( &hA,    lda*Ak ));
            TESTING_CHECK( magma_smalloc_cpu( &hC,    ldc*N  ));
            TESTING_CHECK( magma_smalloc_cpu( &hCdev, ldc*N  ));
            
            TESTING_CHECK( magma_smalloc( &dA, ldda*Ak ));
            TESTING_CHECK( magma_smalloc( &dC, lddc*N  ));
            
            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_slarnv( &ione, ISEED, &sizeC, hC );
            
            // for error checks
            float Anorm = lapackf77_slange( "F", &An, &Ak, hA, &lda, work );
            float Cnorm = safe_lapackf77_slansy( "F", lapack_uplo_const(opts.uplo), &N, hC, &ldc, work );
            
            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_ssetmatrix( An, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            magma_ssetmatrix( N, N, hC, ldc, dC(0,0), lddc, opts.queue );

            dev_time = magma_sync_wtime( opts.queue );
            magma_ssyrk( opts.uplo, opts.transA, N, K,
                         alpha, dA(0,0), ldda,
                         beta,  dC(0,0), lddc, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_sgetmatrix( N, N, dC(0,0), lddc, hCdev, ldc, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ssyrk( lapack_uplo_const(opts.uplo), lapack_trans_const(opts.transA), &N, &K,
                               &alpha, hA, &lda,
                               &beta,  hC, &ldc );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // See testing_sgemm for formula.
                blasf77_saxpy( &sizeC, &c_neg_one, hC, &ione, hCdev, &ione );
                dev_error = safe_lapackf77_slansy( "F", lapack_uplo_const(opts.uplo), &N, hCdev, &ldc, work )
                            / (sqrt(float(K+2))*fabs(alpha)*Anorm*Anorm + 2*fabs(beta)*Cnorm);
                
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
            magma_free_cpu( hC );
            magma_free_cpu( hCdev );
            
            magma_free( dA );
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
