/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_ztrsm.cpp, normal z -> c, Sun Nov 20 20:20:33 2016
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
#include "magma_operators.h"  // for MAGMA_C_DIV
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ctrsm
*/
int main( int argc, char** argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dB(i_, j_)  dB, ((i_) + (j_)*lddb)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #endif
    
    #define hA(i_, j_) (hA + (i_) + (j_)*lda)

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf=0, magma_time=0, dev_perf, dev_time, cpu_perf=0, cpu_time=0;
    float          magma_error=0, dev_error, lapack_error, work[1];
    magma_int_t M, N, info;
    magma_int_t Ak;
    magma_int_t sizeA, sizeB;
    magma_int_t lda, ldb, ldda, lddb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t *ipiv;
    
    magmaFloatComplex *hA, *hB, *hBdev, *hBmagma, *hBlapack, *hX;
    magmaFloatComplex_ptr dA, dB;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex c_one = MAGMA_C_ONE;
    magmaFloatComplex alpha = MAGMA_C_MAKE(  0.29, -0.86 );
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");

    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% side = %s, uplo = %s, transA = %s, diag = %s, ngpu = %lld\n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag), (long long) abs_ngpu);
    
    printf("%%   M     N  MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)      MAGMA     CUBLAS   LAPACK error\n");
    printf("%%============================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_CTRSM(opts.side, M, N) / 1e9;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                Ak  = M;
            } else {
                lda = N;
                Ak  = N;
            }
            
            ldb = M;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            
            sizeA = lda*Ak;
            sizeB = ldb*N;
            
            TESTING_CHECK( magma_cmalloc_cpu( &hA,       lda*Ak  ));
            TESTING_CHECK( magma_cmalloc_cpu( &hB,       ldb*N   ));
            TESTING_CHECK( magma_cmalloc_cpu( &hX,       ldb*N   ));
            TESTING_CHECK( magma_cmalloc_cpu( &hBlapack, ldb*N   ));
            TESTING_CHECK( magma_cmalloc_cpu( &hBdev,    ldb*N   ));
            TESTING_CHECK( magma_cmalloc_cpu( &hBmagma,  ldb*N   ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv,      Ak      ));
            
            TESTING_CHECK( magma_cmalloc( &dA,       ldda*Ak ));
            TESTING_CHECK( magma_cmalloc( &dB,       lddb*N  ));
            
            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            lapackf77_clarnv( &ione, ISEED, &sizeA, hA );
            lapackf77_cgetrf( &Ak, &Ak, hA, &lda, ipiv, &info );
            for( int j = 0; j < Ak; ++j ) {
                for( int i = 0; i < j; ++i ) {
                    *hA(i,j) = *hA(j,i);
                }
            }
            
            lapackf77_clarnv( &ione, ISEED, &sizeB, hB );
            memcpy( hBlapack, hB, sizeB*sizeof(magmaFloatComplex) );
            magma_csetmatrix( Ak, Ak, hA, lda, dA(0,0), ldda, opts.queue );
            
            /* =====================================================================
               Performs operation using MAGMABLAS (only with CUDA)
               =================================================================== */
            #if defined(HAVE_CUBLAS)
                magma_csetmatrix( M, N, hB, ldb, dB(0,0), lddb, opts.queue );
                
                magma_time = magma_sync_wtime( opts.queue );
                if (opts.ngpu == 1) {
                    magmablas_ctrsm( opts.side, opts.uplo, opts.transA, opts.diag,
                                     M, N,
                                     alpha, dA(0,0), ldda,
                                            dB(0,0), lddb, opts.queue );
                }
                else {
                    magma_ctrsm_m( abs_ngpu, opts.side, opts.uplo, opts.transA, opts.diag,
                                   M, N,
                                   alpha, dA(0,0), ldda,
                                          dB(0,0), lddb );
                }                            
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_cgetmatrix( M, N, dB(0,0), lddb, hBmagma, ldb, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_csetmatrix( M, N, hB, ldb, dB(0,0), lddb, opts.queue );
            
            dev_time = magma_sync_wtime( opts.queue );
            magma_ctrsm( opts.side, opts.uplo, opts.transA, opts.diag,
                         M, N,
                         alpha, dA(0,0), ldda,
                                dB(0,0), lddb, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_cgetmatrix( M, N, dB(0,0), lddb, hBdev, ldb, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                blasf77_ctrsm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &alpha, hA, &lda,
                                       hBlapack, &ldb );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            // ||b - 1/alpha*A*x|| / (||A||*||x||)
            magmaFloatComplex inv_alpha = MAGMA_C_DIV( c_one, alpha );
            float normR, normX, normA;
            normA = lapackf77_clantr( "M",
                                      lapack_uplo_const(opts.uplo),
                                      lapack_diag_const(opts.diag),
                                      &Ak, &Ak, hA, &lda, work );
            
            #if defined(HAVE_CUBLAS)
                // check magma
                memcpy( hX, hBmagma, sizeB*sizeof(magmaFloatComplex) );
                blasf77_ctrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &inv_alpha, hA, &lda,
                                           hX, &ldb );
                
                blasf77_caxpy( &sizeB, &c_neg_one, hB, &ione, hX, &ione );
                normR = lapackf77_clange( "M", &M, &N, hX,      &ldb, work );
                normX = lapackf77_clange( "M", &M, &N, hBmagma, &ldb, work );
                magma_error = normR/(normX*normA);
            #endif

            // check cuBLAS / clBLAS
            memcpy( hX, hBdev, sizeB*sizeof(magmaFloatComplex) );
            blasf77_ctrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                           &M, &N,
                           &inv_alpha, hA, &lda,
                                       hX, &ldb );

            blasf77_caxpy( &sizeB, &c_neg_one, hB, &ione, hX, &ione );
            normR = lapackf77_clange( "M", &M, &N, hX,    &ldb, work );
            normX = lapackf77_clange( "M", &M, &N, hBdev, &ldb, work );
            dev_error = normR/(normX*normA);

            bool okay = (magma_error < tol && dev_error < tol);
            status += ! okay;
            if ( opts.lapack ) {
                // check lapack
                // this verifies that the matrix wasn't so bad that it couldn't be solved accurately.
                memcpy( hX, hBlapack, sizeB*sizeof(magmaFloatComplex) );
                blasf77_ctrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                               lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                               &M, &N,
                               &inv_alpha, hA, &lda,
                                           hX, &ldb );
    
                blasf77_caxpy( &sizeB, &c_neg_one, hB, &ione, hX, &ione );
                normR = lapackf77_clange( "M", &M, &N, hX,       &ldb, work );
                normX = lapackf77_clange( "M", &M, &N, hBlapack, &ldb, work );
                lapack_error = normR/(normX*normA);
                
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %8.2e   %8.2e   %s\n",
                        (long long) M, (long long) N,
                        magma_perf,  1000.*magma_time,
                        dev_perf,    1000.*dev_time,
                        cpu_perf,    1000.*cpu_time,
                        magma_error, dev_error, lapack_error,
                        (okay ? "ok" : "failed"));
            }
            else {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)     ---   (  ---  )   %8.2e   %8.2e     ---      %s\n",
                        (long long) M, (long long) N,
                        magma_perf,  1000.*magma_time,
                        dev_perf,    1000.*dev_time,
                        magma_error, dev_error,
                        (okay ? "ok" : "failed"));
            }
            
            magma_free_cpu( hA );
            magma_free_cpu( hB );
            magma_free_cpu( hX );
            magma_free_cpu( hBlapack );
            magma_free_cpu( hBdev    );
            magma_free_cpu( hBmagma  );
            magma_free_cpu( ipiv );
            
            magma_free( dA );
            magma_free( dB );
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
