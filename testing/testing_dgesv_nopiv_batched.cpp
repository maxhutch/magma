/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from testing_zgesv_nopiv_batched.cpp normal z -> d, Fri Sep 11 18:29:39 2015
       @author Mark Gates
*/
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgesv_gpu
*/
int main(int argc, char **argv)
{
    TESTING_INIT();

    real_Double_t   gflops, cpu_perf, cpu_time, gpu_perf, gpu_time;
    double          error, Rnorm, Anorm, Xnorm, *work;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double *h_A, *h_B, *h_X;
    double *d_A, *d_B;
    double **d_A_array, **d_B_array;
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
    
    printf("%% Batchcount   N  NRHS   CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||B - AX|| / N*||A||*||X||\n");
    printf("%%==========================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            ldb    = lda;
            ldda   = magma_roundup( N, opts.align );  // multiple of 32 by default
            n2     = lda*N * batchCount;
            lddb   = ldda;
            gflops = ( FLOPS_DGETRF( N, N ) + FLOPS_DGETRS( N, nrhs ) ) / 1e9 * batchCount;
            
            TESTING_MALLOC_CPU( h_A, double, n2    );
            TESTING_MALLOC_CPU( h_B, double, ldb*nrhs*batchCount );
            TESTING_MALLOC_CPU( h_X, double, ldb*nrhs*batchCount );
            TESTING_MALLOC_CPU( work, double,      N );
            TESTING_MALLOC_CPU( ipiv, magma_int_t, N );
            TESTING_MALLOC_CPU( cpu_info, magma_int_t, batchCount);
            
            TESTING_MALLOC_DEV(  dinfo_magma,  magma_int_t, batchCount);
            
            TESTING_MALLOC_DEV( d_A, double, ldda*N*batchCount    );
            TESTING_MALLOC_DEV( d_B, double, lddb*nrhs*batchCount );
            
            magma_malloc((void**)&d_A_array, batchCount * sizeof(*d_A_array));
            magma_malloc((void**)&d_B_array, batchCount * sizeof(*d_B_array));

            /* Initialize the matrices */
            sizeA = n2;
            sizeB = ldb*nrhs*batchCount;
            lapackf77_dlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_dlarnv( &ione, ISEED, &sizeB, h_B );
            // make A diagonally dominant, to not need pivoting
            for( int s=0; s < batchCount; ++s ) {
                for( int i=0; i < N; ++i ) {
                    h_A[ i + i*lda + s*lda*N ] = MAGMA_D_MAKE(
                        MAGMA_D_REAL( h_A[ i + i*lda + s*lda*N ] ) + N,
                        MAGMA_D_IMAG( h_A[ i + i*lda + s*lda*N ] ));
                }
            }
            columns = N * batchCount;
            magma_dsetmatrix( N, columns,    h_A, lda, d_A, ldda );
            magma_dsetmatrix( N, nrhs*batchCount, h_B, ldb, d_B, lddb );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            dset_pointer(d_A_array, d_A, ldda, 0, 0, ldda*N, batchCount, opts.queue);
            dset_pointer(d_B_array, d_B, lddb, 0, 0, lddb*nrhs, batchCount, opts.queue);
            
            gpu_time = magma_sync_wtime( opts.queue );
            info = magma_dgesv_nopiv_batched( N, nrhs, d_A_array, ldda, d_B_array, lddb, dinfo_magma, batchCount, opts.queue );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gflops / gpu_time;
            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1);
            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_dgesv_nopiv_batched matrix %d returned internal error %d\n",i, (int)cpu_info[i] );
                }
            }
            if (info != 0)
                printf("magma_dgesv_nopiv_batched returned argument error %d: %s.\n", (int) info, magma_strerror( info ));
            
            //=====================================================================
            // Residual
            //=====================================================================
            error = 0;
            magma_dgetmatrix( N, nrhs*batchCount, d_B, lddb, h_X, ldb );
            for (magma_int_t s = 0; s < batchCount; s++)
            {
                Anorm = lapackf77_dlange("I", &N, &N,    h_A+s*lda*N, &lda, work);
                Xnorm = lapackf77_dlange("I", &N, &nrhs, h_X+s*ldb*nrhs, &ldb, work);

                blasf77_dgemm( MagmaNoTransStr, MagmaNoTransStr, &N, &nrhs, &N,
                        &c_one,     h_A+s*lda*N, &lda,
                        h_X+s*ldb*nrhs, &ldb,
                        &c_neg_one, h_B+s*ldb*nrhs, &ldb);

                Rnorm = lapackf77_dlange("I", &N, &nrhs, h_B+s*ldb*nrhs, &ldb, work);
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
                lapackf77_dgesv( &N, &nrhs, h_A, &lda, ipiv, h_B, &ldb, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0)
                    printf("lapackf77_dgesv returned error %d: %s.\n",
                           (int) info, magma_strerror( info ));
                
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
            
            TESTING_FREE_DEV( d_A );
            TESTING_FREE_DEV( d_B );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    TESTING_FINALIZE();
    return status;
}
