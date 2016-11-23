/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Mark Gates
       
       Note: [ds] precisions generated from testing_chemv.cu
       
       @generated from testing/testing_zsymv.cpp, normal z -> c, Sun Nov 20 20:20:33 2016
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing csymv
*/
int main(int argc, char **argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dX(i_)      dX, ((i_))
    #define dY(i_)      dY, ((i_))
    #define dwork(i_)   dwork, ((i_))
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dX(i_)     (dX + (i_))
    #define dY(i_)     (dY + (i_))
    #define dwork(i_)  (dwork + (i_))
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    const magma_int_t        ione      = 1;
    
    real_Double_t   gflops, magma_perf=0, magma_time=0, cpu_perf, cpu_time;
    float          magma_error=0, work[1];
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t N, lda, ldda, sizeA, sizeX, sizeY, blocks, ldwork;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t nb   = 64;
    magmaFloatComplex alpha = MAGMA_C_MAKE(  1.5, -2.3 );
    magmaFloatComplex beta  = MAGMA_C_MAKE( -0.6,  0.8 );
    magmaFloatComplex *A, *X, *Y, *Ymagma;
    magmaFloatComplex_ptr dA, dX, dY, dwork;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    // See testing_cgemm about tolerance.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;

    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%%   N   MAGMA Gflop/s (ms)  CPU Gflop/s (ms)  MAGMA error\n");
    printf("%%========================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            sizeA  = N*lda;
            sizeX  = N*incx;
            sizeY  = N*incy;
            gflops = FLOPS_CSYMV( N ) / 1e9;
            
            TESTING_CHECK( magma_cmalloc_cpu( &A,       sizeA ));
            TESTING_CHECK( magma_cmalloc_cpu( &X,       sizeX ));
            TESTING_CHECK( magma_cmalloc_cpu( &Y,       sizeY ));
            TESTING_CHECK( magma_cmalloc_cpu( &Ymagma,  sizeY ));
            
            TESTING_CHECK( magma_cmalloc( &dA, ldda*N ));
            TESTING_CHECK( magma_cmalloc( &dX, sizeX ));
            TESTING_CHECK( magma_cmalloc( &dY, sizeY ));
            
            blocks = magma_ceildiv( N, nb );
            ldwork = ldda*blocks;
            TESTING_CHECK( magma_cmalloc( &dwork, ldwork ));
            
            magmablas_claset( MagmaFull, ldwork, 1, MAGMA_C_NAN, MAGMA_C_NAN, dwork(0), ldwork, opts.queue );
            magmablas_claset( MagmaFull, ldda,   N, MAGMA_C_NAN, MAGMA_C_NAN, dA(0,0),  ldda,   opts.queue );
            
            /* Initialize the matrix */
            lapackf77_clarnv( &ione, ISEED, &sizeA, A );
            magma_cmake_hermitian( N, A, lda );
            
            // should not use data from the opposite triangle -- fill with NAN to check
            magma_int_t N1 = N-1;
            if ( opts.uplo == MagmaUpper ) {
                lapackf77_claset( "Lower", &N1, &N1, &MAGMA_C_NAN, &MAGMA_C_NAN, &A[1], &lda );
            }
            else {
                lapackf77_claset( "Upper", &N1, &N1, &MAGMA_C_NAN, &MAGMA_C_NAN, &A[lda], &lda );
            }
            
            lapackf77_clarnv( &ione, ISEED, &sizeX, X );
            lapackf77_clarnv( &ione, ISEED, &sizeY, Y );
            
            // for error checks
            // lanhe and lansy should be same
            float Anorm = safe_lapackf77_clanhe( "F", lapack_uplo_const(opts.uplo), &N, A, &lda, work );
            float Xnorm = lapackf77_clange( "F", &N, &ione, X, &N, work );
            float Ynorm = lapackf77_clange( "F", &N, &ione, Y, &N, work );
            
            /* Note: CUBLAS does not implement csymv */
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_csetmatrix( N, N, A, lda, dA(0,0), ldda, opts.queue );
                magma_csetvector( N, X, incx, dX(0), incx, opts.queue );
                magma_csetvector( N, Y, incy, dY(0), incy, opts.queue );
                
                magma_time = magma_sync_wtime( opts.queue );
                if ( opts.version == 1 ) {
                    magmablas_csymv_work( opts.uplo, N,
                                          alpha, dA(0,0), ldda,
                                                 dX(0),   incx,
                                          beta,  dY(0),   incy,
                                          dwork(0), ldwork, opts.queue );
                }
                else {
                    // non-work interface (has added overhead)
                    magmablas_csymv( opts.uplo, N,
                                     alpha, dA(0,0), ldda,
                                            dX(0),   incx,
                                     beta,  dY(0),   incy, opts.queue );
                }
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_cgetvector( N, dY(0), incy, Ymagma, incy, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_csymv( lapack_uplo_const(opts.uplo), &N, &alpha, A, &lda, X, &incx, &beta, Y, &incy );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            // See testing_cgemm for formula. Here K = N.
            blasf77_caxpy( &N, &c_neg_one, Y, &incy, Ymagma, &incy );
            magma_error = lapackf77_clange( "M", &N, &ione, Ymagma, &N, work )
                            / (sqrt(float(N+2))*fabs(alpha)*Anorm*Xnorm + 2*fabs(beta)*Ynorm);
            
            bool okay = (magma_error < tol);
            status += ! okay;
            printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                   (long long) N,
                   magma_perf,  1000.*magma_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, (okay ? "ok" : "failed"));
            
            magma_free_cpu( A );
            magma_free_cpu( X );
            magma_free_cpu( Y );
            magma_free_cpu( Ymagma  );
            
            magma_free( dA );
            magma_free( dX );
            magma_free( dY );
            magma_free( dwork );
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
