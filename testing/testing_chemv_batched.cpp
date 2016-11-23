/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zhemv_batched.cpp, normal z -> c, Sun Nov 20 20:20:38 2016
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
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing chemv_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    float          error, magma_error, normalize, work[1];
    magma_int_t N, lda, ldda;
    magma_int_t sizeA, sizeX, sizeY;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    magmaFloatComplex *h_A, *h_X, *h_Y, *h_Ymagma;
    magmaFloatComplex *d_A, *d_X, *d_Y;
    magmaFloatComplex **d_A_array = NULL;
    magmaFloatComplex **d_X_array = NULL;
    magmaFloatComplex **d_Y_array = NULL;
    
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex alpha = MAGMA_C_MAKE(  0.29, -0.86 );
    magmaFloatComplex beta  = MAGMA_C_MAKE( -0.48,  0.38 );
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;
    batchCount = opts.batchcount;
    
    float *Anorm, *Xnorm, *Ynorm;
    TESTING_CHECK( magma_smalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Xnorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Ynorm, batchCount ));
    
    TESTING_CHECK( magma_malloc( (void**) &d_A_array, batchCount * sizeof(magmaFloatComplex*) ));
    TESTING_CHECK( magma_malloc( (void**) &d_X_array, batchCount * sizeof(magmaFloatComplex*) ));
    TESTING_CHECK( magma_malloc( (void**) &d_Y_array, batchCount * sizeof(magmaFloatComplex*) ));
    
    // See testing_cgemm about tolerance.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;
    
    printf("%% uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("%% BatchCount     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=======================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            gflops = FLOPS_CHEMV( N ) / 1e9 * batchCount;

            sizeA = lda*N*batchCount;
            sizeX = incx*N*batchCount;
            sizeY = incy*N*batchCount;

            TESTING_CHECK( magma_cmalloc_cpu( &h_A,  sizeA ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_X,  sizeX ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_Y,  sizeY  ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_Ymagma,  sizeY  ));

            TESTING_CHECK( magma_cmalloc( &d_A, ldda*N*batchCount ));
            TESTING_CHECK( magma_cmalloc( &d_X, sizeX ));
            TESTING_CHECK( magma_cmalloc( &d_Y, sizeY ));

            /* Initialize the matrices */
            lapackf77_clarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_clarnv( &ione, ISEED, &sizeX, h_X );
            lapackf77_clarnv( &ione, ISEED, &sizeY, h_Y );
            
            /* set the opposite triangular part to NAN to check */
            magma_int_t N1 = N-1;
            for(magma_int_t i = 0; i < batchCount; i++){
                magmaFloatComplex* Ai = h_A + i * N * lda;
                if ( opts.uplo == MagmaUpper ) {
                    lapackf77_claset( "Lower", &N1, &N1, &MAGMA_C_NAN, &MAGMA_C_NAN, &Ai[1], &lda );
                }
                else {
                    lapackf77_claset( "Upper", &N1, &N1, &MAGMA_C_NAN, &MAGMA_C_NAN, &Ai[lda], &lda );
                }
            }

            // Compute norms for error computation
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = safe_lapackf77_clanhe( "F", lapack_uplo_const(opts.uplo), &N, &h_A[s*lda*N], &lda, work );
                Xnorm[s] = lapackf77_clange( "F", &ione, &N, &h_X[s*N*incx], &incx, work );
                Ynorm[s] = lapackf77_clange( "F", &ione, &N, &h_Y[s*N*incy], &incy, work );
            }

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_csetmatrix( N, N*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_csetvector( N*batchCount, h_X, incx, d_X, incx, opts.queue );
            magma_csetvector( N*batchCount, h_Y, incy, d_Y, incy, opts.queue );
            
            magma_cset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_cset_pointer( d_X_array, d_X, 1, 0, 0, incx*N, batchCount, opts.queue );
            magma_cset_pointer( d_Y_array, d_Y, 1, 0, 0, incy*N, batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_chemv_batched(opts.uplo, N,
                             alpha, d_A_array, ldda,
                                    d_X_array, incx,
                             beta,  d_Y_array, incy, batchCount, opts.queue);
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_cgetvector( N*batchCount, d_Y, incy, h_Ymagma, incy, opts.queue );
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (int i=0; i < batchCount; i++)
                {
                    blasf77_chemv( lapack_uplo_const(opts.uplo), &N,
                                   &alpha, h_A + i*lda*N, &lda,
                                           h_X + i*N*incx, &incx,
                                   &beta,  h_Y + i*N*incy, &incy );
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.lapack ) {
                // compute error compared lapack
                // error = |dY - Y| / (gamma_{k+2}|A||X| + gamma_2|Yin|); k = m
                magma_error = 0;
                
                for (int s=0; s < batchCount; s++){
                    normalize = sqrt(float(N+2))*Anorm[s]*Xnorm[s] + 2*Ynorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    blasf77_caxpy( &N, &c_neg_one, &h_Y[s*N*incy], &incy, &h_Ymagma[s*N*incy], &incy );
                    error = lapackf77_clange( "F", &ione, &N, &h_Ymagma[s*N*incy], &incy, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                       (long long) batchCount, (long long) N,
                       magma_perf,  1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                       (long long) batchCount,(long long) N,
                       magma_perf,  1000.*magma_time);
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_X );
            magma_free_cpu( h_Y );
            magma_free_cpu( h_Ymagma );

            magma_free( d_A );
            magma_free( d_X );
            magma_free( d_Y );
            
            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_free_cpu( Anorm );
    magma_free_cpu( Xnorm );
    magma_free_cpu( Ynorm );

    magma_free( d_A_array );
    magma_free( d_X_array );
    magma_free( d_Y_array );
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
