/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zgesv_nopiv_batched.cpp, normal z -> c, Sun Nov 20 20:20:38 2016
       @author Mark Gates
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgesv_gpu
*/
int main(int argc, char **argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    float          error, Rnorm, Anorm, Xnorm, *work;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex *h_A, *h_B, *h_X;
    magmaFloatComplex *d_A, *d_B;
    magmaFloatComplex **d_A_array, **d_B_array;
    magma_int_t *ipiv, *cpu_info;
    magma_int_t N, n2, nrhs, lda, ldb, ldda, lddb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t     *dinfo_magma;
    magma_int_t batchCount;
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    
    float tol = opts.tolerance * lapackf77_slamch("E");

    batchCount = opts.batchcount;
    magma_int_t columns;
    nrhs = opts.nrhs;
    
    printf("%% Batchcount   N  NRHS   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%==========================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldb    = lda;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            n2     = lda*N * batchCount;
            lddb   = ldda;
            gflops = ( FLOPS_CGETRF( N, N ) + FLOPS_CGETRS( N, nrhs ) ) / 1e9 * batchCount;
            
            TESTING_CHECK( magma_cmalloc_cpu( &h_A, n2    ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_B, ldb*nrhs*batchCount ));
            TESTING_CHECK( magma_cmalloc_cpu( &h_X, ldb*nrhs*batchCount ));
            TESTING_CHECK( magma_smalloc_cpu( &work, N ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, N ));
            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));
            
            TESTING_CHECK( magma_imalloc( &dinfo_magma, batchCount ));
            
            TESTING_CHECK( magma_cmalloc( &d_A, ldda*N*batchCount    ));
            TESTING_CHECK( magma_cmalloc( &d_B, lddb*nrhs*batchCount ));
            
            TESTING_CHECK( magma_malloc( (void**) &d_A_array, batchCount * sizeof(magmaFloatComplex*) ));
            TESTING_CHECK( magma_malloc( (void**) &d_B_array, batchCount * sizeof(magmaFloatComplex*) ));

            /* Initialize the matrices */
            sizeA = n2;
            sizeB = ldb*nrhs*batchCount;
            lapackf77_clarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_clarnv( &ione, ISEED, &sizeB, h_B );
            // make A diagonally dominant, to not need pivoting
            for( int s=0; s < batchCount; ++s ) {
                for( int i=0; i < N; ++i ) {
                    h_A[ i + i*lda + s*lda*N ] = MAGMA_C_MAKE(
                        MAGMA_C_REAL( h_A[ i + i*lda + s*lda*N ] ) + N,
                        MAGMA_C_IMAG( h_A[ i + i*lda + s*lda*N ] ));
                }
            }
            columns = N * batchCount;
            magma_csetmatrix( N, columns,    h_A, lda, d_A, ldda, opts.queue );
            magma_csetmatrix( N, nrhs*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_cset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_cset_pointer( d_B_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            info = magma_cgesv_nopiv_batched( N, nrhs, d_A_array, ldda, d_B_array, lddb, dinfo_magma, batchCount, opts.queue );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1, opts.queue );
            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_cgesv_nopiv_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) cpu_info[i] );
                }
            }
            if (info != 0) {
                printf("magma_cgesv_nopiv_batched returned argument error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            //=====================================================================
            // Residual
            //=====================================================================
            error = 0;
            magma_cgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, ldb, opts.queue );
            for (magma_int_t s = 0; s < batchCount; s++)
            {
                Anorm = lapackf77_clange("I", &N, &N,    h_A+s*lda*N, &lda, work);
                Xnorm = lapackf77_clange("I", &N, &nrhs, h_X+s*ldb*nrhs, &ldb, work);

                blasf77_cgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                        &c_one,     h_A+s*lda*N, &lda,
                        h_X+s*ldb*nrhs, &ldb,
                        &c_neg_one, h_B+s*ldb*nrhs, &ldb);

                Rnorm = lapackf77_clange("I", &N, &nrhs, h_B+s*ldb*nrhs, &ldb, work);
                float err = Rnorm/(N*Anorm*Xnorm);
                if ( isnan(err) || isinf(err) ) {
                    error = err;
                    break;
                }
                error = max( err, error );
            }
            bool okay = (error < tol);
            status += ! okay;
            
            /* ====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                lapackf77_cgesv( &N, &nrhs, h_A, &lda, ipiv, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_cgesv returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                printf( "%10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "%10lld %5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) batchCount, (long long) N, (long long) nrhs, gpu_perf, gpu_time,
                        error, (okay ? "ok" : "failed"));
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_X );
            magma_free_cpu( work );
            magma_free_cpu( ipiv );
            magma_free_cpu( cpu_info );
            
            magma_free( dinfo_magma );
            magma_free( d_A );
            magma_free( d_B );
            
            magma_free( d_A_array );
            magma_free( d_B_array );
            
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
