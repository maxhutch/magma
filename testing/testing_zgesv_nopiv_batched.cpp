/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> c d s
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
   -- Testing zgesv_gpu
*/
int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    double          error, Rnorm, Anorm, Xnorm, *work;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *h_A, *h_B, *h_X;
    magmaDoubleComplex *d_A, *d_B;
    magmaDoubleComplex **d_A_array, **d_B_array;
    magma_int_t *ipiv, *cpu_info;
    magma_int_t N, n2, nrhs, lda, ldb, ldda, lddb, info, sizeA, sizeB;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magma_int_t     *dinfo_magma;
    magma_int_t batchCount;
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

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
            gflops = ( FLOPS_ZGETRF( N, N ) + FLOPS_ZGETRS( N, nrhs ) ) / 1e9 * batchCount;
            
            TESTING_MALLOC_CPU( h_A, magmaDoubleComplex, n2    );
            TESTING_MALLOC_CPU( h_B, magmaDoubleComplex, ldb*nrhs*batchCount );
            TESTING_MALLOC_CPU( h_X, magmaDoubleComplex, ldb*nrhs*batchCount );
            TESTING_MALLOC_CPU( work, double,      N );
            TESTING_MALLOC_CPU( ipiv, magma_int_t, N );
            TESTING_MALLOC_CPU( cpu_info, magma_int_t, batchCount);
            
            TESTING_MALLOC_DEV( dinfo_magma, magma_int_t, batchCount );
            
            TESTING_MALLOC_DEV( d_A, magmaDoubleComplex, ldda*N*batchCount    );
            TESTING_MALLOC_DEV( d_B, magmaDoubleComplex, lddb*nrhs*batchCount );
            
            TESTING_MALLOC_DEV( d_A_array, magmaDoubleComplex*, batchCount );
            TESTING_MALLOC_DEV( d_B_array, magmaDoubleComplex*, batchCount );

            /* Initialize the matrices */
            sizeA = n2;
            sizeB = ldb*nrhs*batchCount;
            lapackf77_zlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_zlarnv( &ione, ISEED, &sizeB, h_B );
            // make A diagonally dominant, to not need pivoting
            for( int s=0; s < batchCount; ++s ) {
                for( int i=0; i < N; ++i ) {
                    h_A[ i + i*lda + s*lda*N ] = MAGMA_Z_MAKE(
                        MAGMA_Z_REAL( h_A[ i + i*lda + s*lda*N ] ) + N,
                        MAGMA_Z_IMAG( h_A[ i + i*lda + s*lda*N ] ));
                }
            }
            columns = N * batchCount;
            magma_zsetmatrix( N, columns,    h_A, lda, d_A, ldda, opts.queue );
            magma_zsetmatrix( N, nrhs*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_zset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_zset_pointer( d_B_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            info = magma_zgesv_nopiv_batched( N, nrhs, d_A_array, ldda, d_B_array, lddb, dinfo_magma, batchCount, opts.queue );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1, opts.queue );
            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_zgesv_nopiv_batched matrix %d returned internal error %d\n",i, (int)cpu_info[i] );
                }
            }
            if (info != 0) {
                printf("magma_zgesv_nopiv_batched returned argument error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            }
            
            //=====================================================================
            // Residual
            //=====================================================================
            error = 0;
            magma_zgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, ldb, opts.queue );
            for (magma_int_t s = 0; s < batchCount; s++)
            {
                Anorm = lapackf77_zlange("I", &N, &N,    h_A+s*lda*N, &lda, work);
                Xnorm = lapackf77_zlange("I", &N, &nrhs, h_X+s*ldb*nrhs, &ldb, work);

                blasf77_zgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                        &c_one,     h_A+s*lda*N, &lda,
                        h_X+s*ldb*nrhs, &ldb,
                        &c_neg_one, h_B+s*ldb*nrhs, &ldb);

                Rnorm = lapackf77_zlange("I", &N, &nrhs, h_B+s*ldb*nrhs, &ldb, work);
                double err = Rnorm/(N*Anorm*Xnorm);
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
                lapackf77_zgesv( &N, &nrhs, h_A, &lda, ipiv, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zgesv returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                }
                
                printf( "%10d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                        (int)batchCount, (int) N, (int) nrhs, cpu_perf, cpu_time, gpu_perf, gpu_time,
                        error, (okay ? "ok" : "failed"));
            }
            else {
                printf( "%10d %5d %5d     ---   (  ---  )   %7.2f (%7.2f)   %8.2e   %s\n",
                        (int) batchCount, (int) N, (int) nrhs, gpu_perf, gpu_time,
                        error, (okay ? "ok" : "failed"));
            }
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_CPU( h_B );
            TESTING_FREE_CPU( h_X );
            TESTING_FREE_CPU( work );
            TESTING_FREE_CPU( ipiv );
            TESTING_FREE_CPU( cpu_info );
            
            TESTING_FREE_DEV( dinfo_magma );
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            
            TESTING_FREE_DEV( d_A_array );
            TESTING_FREE_DEV( d_B_array );
            
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_FINALIZE();
    return status;
}
