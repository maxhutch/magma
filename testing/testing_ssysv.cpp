/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zhesv.cpp, normal z -> s, Sun Nov 20 20:20:35 2016
       @author Ichitaro Yamazaki
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

/* ================================================================================================== */

// Initialize matrix to random & symmetrize.
// Having this in separate function ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix( magma_int_t m, magma_int_t n, float *h_A, magma_int_t lda )
{
    assert( m == n );
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t n2 = lda*n;
    lapackf77_slarnv( &ione, ISEED, &n2, h_A );
    magma_smake_symmetric( n, h_A, lda );
}


// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
float get_residual(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    float *A, magma_int_t lda, magma_int_t *ipiv,
    float *x, magma_int_t ldx,
    float *b, magma_int_t ldb)
{
    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const magma_int_t ione = 1;
    
    // reset to original A
    init_matrix( n, n, A, lda );
    
    // compute r = Ax - b, saved in b
    blasf77_sgemv( "Notrans", &n, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_slange( MagmaFullStr, &n, &n, A, &lda, work );
    norm_r = lapackf77_slange( MagmaFullStr, &n, &ione, b, &n, work );
    norm_x = lapackf77_slange( MagmaFullStr, &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_sprint( 1, n, b, 1 );
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%lld\n", norm_r, norm_A, norm_x, (long long) n );
    return norm_r / (n * norm_A * norm_x);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ssysv
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    float *h_A, *h_B, *h_X, *work, temp;
    real_Double_t   gflops, gpu_perf, gpu_time = 0.0, cpu_perf=0, cpu_time=0;
    float          error, error_lapack = 0.0;
    magma_int_t     *ipiv;
    magma_int_t     N, n2, lda, ldb, sizeB, lwork, info;
    magma_int_t     ione = 1;
    magma_int_t     ISEED[4] = {0,0,0,1};
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");

    printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            ldb    = N;
            lda    = N;
            n2     = lda*N;
            sizeB  = ldb*opts.nrhs;
            gflops = ( FLOPS_SPOTRF( N ) + FLOPS_SPOTRS( N, opts.nrhs ) ) / 1e9;
            
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, N ));
            TESTING_CHECK( magma_smalloc_pinned( &h_A,  n2 ));
            TESTING_CHECK( magma_smalloc_pinned( &h_B,  sizeB ));
            TESTING_CHECK( magma_smalloc_pinned( &h_X,  sizeB ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                lwork = -1;
                lapackf77_ssysv(lapack_uplo_const(opts.uplo), &N, &opts.nrhs,
                                h_A, &lda, ipiv, h_X, &ldb, &temp, &lwork, &info);
                lwork = (magma_int_t)MAGMA_S_REAL(temp);
                TESTING_CHECK( magma_smalloc_cpu( &work, lwork ));

                init_matrix( N, N, h_A, lda );
                lapackf77_slarnv( &ione, ISEED, &sizeB, h_B );
                lapackf77_slacpy( MagmaFullStr, &N, &opts.nrhs, h_B, &ldb, h_X, &ldb );

                cpu_time = magma_wtime();
                lapackf77_ssysv(lapack_uplo_const(opts.uplo), &N, &opts.nrhs,
                                h_A, &lda, ipiv, h_X, &ldb, work, &lwork, &info);
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_ssysv returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                error_lapack = get_residual( opts.uplo, N, opts.nrhs, h_A, lda, ipiv, h_X, ldb, h_B, ldb );

                magma_free_cpu( work );
            }
           
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            init_matrix( N, N, h_A, lda );
            lapackf77_slarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_slacpy( MagmaFullStr, &N, &opts.nrhs, h_B, &ldb, h_X, &ldb );

            magma_setdevice(0);
            gpu_time = magma_wtime();
            magma_ssysv( opts.uplo, N, opts.nrhs, h_A, lda, ipiv, h_X, ldb, &info);
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_ssysv returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) N, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) N, (long long) N, gpu_perf, gpu_time );
            }
            if ( opts.check == 0 ) {
                printf("     ---   \n");
            } else {
                error = get_residual( opts.uplo, N, opts.nrhs, h_A, lda, ipiv, h_X, ldb, h_B, ldb );
                printf("   %8.2e   %s", error, (error < tol ? "ok" : "failed"));
                if (opts.lapack)
                    printf(" (lapack rel.res. = %8.2e)", error_lapack);
                printf("\n");
                status += ! (error < tol);
            }
            
            magma_free_cpu( ipiv );
            magma_free_pinned( h_X  );
            magma_free_pinned( h_B  );
            magma_free_pinned( h_A  );
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
