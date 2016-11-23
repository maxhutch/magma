/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zgetrf_mgpu.cpp, normal z -> s, Sun Nov 20 20:20:35 2016
       @author Mark Gates
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


// Initialize matrix to random.
// Having this in separate function ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix( magma_int_t m, magma_int_t n, float *h_A, magma_int_t lda )
{
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t n2 = lda*n;
    lapackf77_slarnv( &ione, ISEED, &n2, h_A );
}


// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
float get_residual(
    magma_int_t m, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    if ( m != n ) {
        printf( "\nERROR: residual check defined only for square matrices\n" );
        return -1;
    }
    
    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const magma_int_t ione = 1;
    
    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,2};
    magma_int_t info = 0;
    float *x, *b;
    
    // initialize RHS
    TESTING_CHECK( magma_smalloc_cpu( &x, n ));
    TESTING_CHECK( magma_smalloc_cpu( &b, n ));
    lapackf77_slarnv( &ione, ISEED, &n, b );
    blasf77_scopy( &n, b, &ione, x, &ione );
    
    // solve Ax = b
    lapackf77_sgetrs( "Notrans", &n, &ione, A, &lda, ipiv, x, &n, &info );
    if (info != 0) {
        printf("lapackf77_sgetrs returned error %lld: %s.\n",
               (long long) info, magma_strerror( info ));
    }
    
    // reset to original A
    init_matrix( m, n, A, lda );
    
    // compute r = Ax - b, saved in b
    blasf77_sgemv( "Notrans", &m, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    float norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_slange( "F", &m, &n, A, &lda, work );
    norm_r = lapackf77_slange( "F", &n, &ione, b, &n, work );
    norm_x = lapackf77_slange( "F", &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_sprint( 1, n, b, 1 );
    
    magma_free_cpu( x );
    magma_free_cpu( b );
    
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%lld\n", norm_r, norm_A, norm_x, (long long) n );
    return norm_r / (n * norm_A * norm_x);
}


// On input, LU and ipiv is LU factorization of A. On output, LU is overwritten.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 3 more matrices to store A, L, and U.
float get_LU_error(magma_int_t M, magma_int_t N,
                    float *LU, magma_int_t lda,
                    magma_int_t *ipiv)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    float alpha = MAGMA_S_ONE;
    float beta  = MAGMA_S_ZERO;
    float *A, *L, *U;
    float work[1], matnorm, residual;
    
    TESTING_CHECK( magma_smalloc_cpu( &A, lda*N    ));
    TESTING_CHECK( magma_smalloc_cpu( &L, M*min_mn ));
    TESTING_CHECK( magma_smalloc_cpu( &U, min_mn*N ));
    memset( L, 0, M*min_mn*sizeof(float) );
    memset( U, 0, min_mn*N*sizeof(float) );

    // set to original A
    init_matrix( M, N, A, lda );
    lapackf77_slaswp( &N, A, &lda, &ione, &min_mn, ipiv, &ione);
    
    // copy LU to L and U, and set diagonal to 1
    lapackf77_slacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_slacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );
    for (j=0; j < min_mn; j++)
        L[j+j*M] = MAGMA_S_MAKE( 1., 0. );
    
    matnorm = lapackf77_slange("f", &M, &N, A, &lda, work);

    blasf77_sgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_S_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_slange("f", &M, &N, LU, &lda, work);

    magma_free_cpu( A );
    magma_free_cpu( L );
    magma_free_cpu( U );

    return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sgetrf_mgpu
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t    gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    float           error;
    float *h_A;
    magmaFloat_ptr d_lA[ MagmaMaxGPUs ];
    magma_int_t *ipiv;
    magma_int_t M, N, n2, lda, ldda, n_local, ngpu;
    magma_int_t info, min_mn, nb, ldn_local;
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.ngpu = abs( opts.ngpu );  // always uses multi-GPU code
    
    float tol = opts.tolerance * lapackf77_slamch("E");

    magma_queue_t queues[MagmaMaxGPUs];
    for( int dev = 0; dev < opts.ngpu; ++dev ) {
        magma_queue_create( dev, &queues[dev] );
    }
    
    printf("%% ngpu %lld\n", (long long) opts.ngpu );
    if ( opts.check == 2 ) {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    }
    else {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |PA-LU|/(N*|A|)\n");
    }
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            nb     = magma_get_sgetrf_nb( M, N );
            gflops = FLOPS_SGETRF( M, N ) / 1e9;
            
            // ngpu must be at least the number of blocks
            ngpu = min( opts.ngpu, magma_ceildiv(N,nb) );
            if ( ngpu < opts.ngpu ) {
                printf( " * too many GPUs for the matrix size, using %lld GPUs\n", (long long) ngpu );
            }
            
            // Allocate host memory for the matrix
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, min_mn ));
            TESTING_CHECK( magma_smalloc_cpu( &h_A,  n2     ));
            
            // Allocate device memory
            for( int dev=0; dev < ngpu; dev++ ) {
                n_local = ((N/nb)/ngpu)*nb;
                if (dev < (N/nb) % ngpu)
                    n_local += nb;
                else if (dev == (N/nb) % ngpu)
                    n_local += N % nb;
                ldn_local = magma_roundup( n_local, opts.align );  // multiple of 32 by default  // TODO why?
                magma_setdevice( dev );
                TESTING_CHECK( magma_smalloc( &d_lA[dev], ldda*ldn_local ));
            }
    
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                init_matrix( M, N, h_A, lda );
                
                cpu_time = magma_wtime();
                lapackf77_sgetrf( &M, &N, h_A, &lda, ipiv, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_sgetrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            init_matrix( M, N, h_A, lda );
            magma_ssetmatrix_1D_col_bcyclic( ngpu, M, N, nb, h_A, lda, d_lA, ldda, queues );

            gpu_time = magma_wtime();
            magma_sgetrf_mgpu( ngpu, M, N, d_lA, ldda, ipiv, &info );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_sgetrf_mgpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
                       
            magma_sgetmatrix_1D_col_bcyclic( ngpu, M, N, nb, d_lA, ldda, h_A, lda, queues );
    
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld %5lld  %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) M, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5lld %5lld    ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) M, (long long) N, gpu_perf, gpu_time );
            }
            if ( opts.check == 2 ) {
                error = get_residual( M, N, h_A, lda, ipiv );
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else if ( opts.check ) {
                error = get_LU_error( M, N, h_A, lda, ipiv );
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf( "     ---\n" );
            }
            
            magma_free_cpu( ipiv );
            magma_free_cpu( h_A );
            for( int dev=0; dev < ngpu; dev++ ) {
                magma_setdevice( dev );
                magma_free( d_lA[dev] );
            }
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    for( int dev = 0; dev < opts.ngpu; ++dev ) {
        magma_queue_destroy( queues[dev] );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
