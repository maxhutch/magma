/*
   -- MAGMA (version 2.2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date November 2016

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from testing/testing_zgetrf_nopiv_batched.cpp, normal z -> d, Sun Nov 20 20:20:38 2016
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

double get_LU_error(magma_int_t M, magma_int_t N,
                    double *A,  magma_int_t lda,
                    double *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    double alpha = MAGMA_D_ONE;
    double beta  = MAGMA_D_ZERO;
    double *L, *U;
    double work[1], matnorm, residual;
    
    TESTING_CHECK( magma_dmalloc_cpu( &L, M*min_mn ));
    TESTING_CHECK( magma_dmalloc_cpu( &U, min_mn*N ));
    memset( L, 0, M*min_mn*sizeof(double) );
    memset( U, 0, min_mn*N*sizeof(double) );

    lapackf77_dlaswp( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    lapackf77_dlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_dlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for (j=0; j < min_mn; j++)
        L[j+j*M] = MAGMA_D_MAKE( 1., 0. );
    
    matnorm = lapackf77_dlange("f", &M, &N, A, &lda, work);

    blasf77_dgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_D_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_dlange("f", &M, &N, LU, &lda, work);

    magma_free_cpu( L );
    magma_free_cpu( U );

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cublas_perf=0., cublas_time=0., cpu_perf=0, cpu_time=0;
    double          error;
    magma_int_t cublas_enable = 0;
    double *h_A, *h_R;
    double *dA_magma;
    double **dA_array = NULL;

    magma_int_t     **dipiv_array = NULL;
    magma_int_t     *ipiv, *cpu_info;
    magma_int_t     *dipiv_magma, *dinfo_magma;
    
    magma_int_t M, N, n2, lda, ldda, min_mn, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t batchCount;
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    //opts.lapack |= opts.check;

    batchCount = opts.batchcount;
    magma_int_t columns;
    
    double tol = opts.tolerance * lapackf77_dlamch("E");

    printf("%% BatchCount   M     N    CPU Gflop/s (ms)   MAGMA Gflop/s (ms)   CUBLAS Gflop/s (ms)   ||PA-LU||/(||A||*N)\n");
    printf("%%==========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N * batchCount;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_DGETRF( M, N ) / 1e9 * batchCount;
            
            TESTING_CHECK( magma_imalloc_cpu( &cpu_info, batchCount ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, min_mn * batchCount ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_A,  n2 ));
            TESTING_CHECK( magma_dmalloc_cpu( &h_R,  n2 ));
            
            TESTING_CHECK( magma_dmalloc( &dA_magma,  ldda*N * batchCount ));
            TESTING_CHECK( magma_imalloc( &dipiv_magma,  min_mn * batchCount ));
            TESTING_CHECK( magma_imalloc( &dinfo_magma,  batchCount ));

            TESTING_CHECK( magma_malloc( (void**) &dA_array,    batchCount * sizeof(double*) ));
            TESTING_CHECK( magma_malloc( (void**) &dipiv_array, batchCount * sizeof(magma_int_t*) ));

            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            // make A diagonally dominant, to not need pivoting
            for( int s=0; s < batchCount; ++s ) {
                for( int i=0; i < min_mn; ++i ) {
                    h_A[ i + i*lda + s*lda*N ] = MAGMA_D_MAKE(
                        MAGMA_D_REAL( h_A[ i + i*lda + s*lda*N ] ) + N,
                        MAGMA_D_IMAG( h_A[ i + i*lda + s*lda*N ] ));
                }
            }
            columns = N * batchCount;
            lapackf77_dlacpy( MagmaFullStr, &M, &columns, h_A, &lda, h_R, &lda );
            magma_dsetmatrix( M, columns, h_R, lda, dA_magma, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_dset_pointer( dA_array, dA_magma, ldda, 0, 0, ldda*N, batchCount, opts.queue );
            magma_time = magma_sync_wtime( opts.queue );
            info = magma_dgetrf_nopiv_batched( M, N, dA_array, ldda, dinfo_magma, batchCount, opts.queue);
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            // check correctness of results throught "dinfo_magma" and correctness of argument throught "info"
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_magma, 1, cpu_info, 1, opts.queue );
            for (int i=0; i < batchCount; i++)
            {
                if (cpu_info[i] != 0 ) {
                    printf("magma_dgetrf_batched matrix %lld returned internal error %lld\n", (long long) i, (long long) cpu_info[i] );
                }
            }
            if (info != 0) {
                printf("magma_dgetrf_batched returned argument error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                for (int i=0; i < batchCount; i++) {
                    lapackf77_dgetrf(&M, &N, h_A + i*lda*N, &lda, ipiv + i * min_mn, &info);
                    assert( info == 0 );
                }
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_dgetrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%10lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)     %7.2f (%7.2f)",
                       (long long) batchCount, (long long) M, (long long) N,
                       cpu_perf, cpu_time*1000.,
                       magma_perf, magma_time*1000.,
                       cublas_perf*cublas_enable, cublas_time*1000.*cublas_enable  );
            }
            else {
                printf("%10lld %5lld %5lld     ---   (  ---  )    %7.2f (%7.2f)     %7.2f (%7.2f)",
                       (long long) batchCount, (long long) M, (long long) N,
                       magma_perf, magma_time*1000.,
                       cublas_perf*cublas_enable, cublas_time*1000.*cublas_enable );
            }

            if ( opts.check ) {
                // initialize ipiv to 1, 2, 3, ...
                for (int i=0; i < batchCount; i++)
                {
                    for (int k=0; k < min_mn; k++) {
                        ipiv[i*min_mn+k] = k+1;
                    }
                }

                magma_dgetmatrix( M, N*batchCount, dA_magma, ldda, h_A, lda, opts.queue );
                error = 0;
                for (int i=0; i < batchCount; i++)
                {
                    double err;
                    err = get_LU_error( M, N, h_R + i * lda*N, lda, h_A + i * lda*N, ipiv + i * min_mn);
                    if ( isnan(err) || isinf(err) ) {
                        error = err;
                        break;
                    }
                    error = max( err, error );
                }
                bool okay = (error < tol);
                status += ! okay;
                printf("   %8.2e  %s\n", error, (okay ? "ok" : "failed") );
            }
            else {
                printf("     ---  \n");
            }
            
            magma_free_cpu( cpu_info );
            magma_free_cpu( ipiv );
            magma_free_cpu( h_A );
            magma_free_cpu( h_R );

            magma_free( dA_magma );
            magma_free( dinfo_magma );
            magma_free( dipiv_magma );
            magma_free( dipiv_array );
            magma_free( dA_array );
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
