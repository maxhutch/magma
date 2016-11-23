/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zher2k_batched.cpp, normal z -> d, Sun Nov 20 20:20:38 2016
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
#endif
#include "../control/magma_threadsetting.h"  // internal header

#define REAL

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dsyr2k_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    double          error, magma_error, normalize, work[1];
    magma_int_t N, K;
    magma_int_t An, Ak, Bn, Bk;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    double *h_A, *h_B, *h_C, *h_Cmagma;
    double *d_A, *d_B, *d_C;
    double **d_A_array = NULL;
    double **d_B_array = NULL;
    double **d_C_array = NULL;

    double c_neg_one = MAGMA_D_NEG_ONE;
    double alpha = MAGMA_D_MAKE(  0.29, -0.86 );
    double beta  = MAGMA_D_MAKE( -0.48,  0.38 );
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;
    batchCount = opts.batchcount;
    
    double *Anorm, *Bnorm, *Cnorm;
    TESTING_CHECK( magma_dmalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Bnorm, batchCount ));
    TESTING_CHECK( magma_dmalloc_cpu( &Cnorm, batchCount ));
    
    TESTING_CHECK( magma_malloc((void**)&d_A_array, batchCount*sizeof(double*)) );
    TESTING_CHECK( magma_malloc((void**)&d_B_array, batchCount*sizeof(double*)) );
    TESTING_CHECK( magma_malloc((void**)&d_C_array, batchCount*sizeof(double*)) );
    
    // See testing_dgemm about tolerance.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;
    
    #ifdef COMPLEX
    if (opts.transA == MagmaTrans) {
        opts.transA = MagmaConjTrans;
        printf("%% WARNING: transA = MagmaTrans changed to MagmaConjTrans\n");
    }
    #endif
    
    printf("%% If running lapack (option --lapack), MAGMA error is computed\n"
           "%% relative to CPU BLAS result.\n\n"
           "%% uplo = %s, trans = %s\n",
           lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA));
    
    printf("%% BatchCount     N     K   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            K = opts.ksize[itest];
            gflops = (batchCount * FLOPS_DSYR2K( K, N )) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                lda = An = N;
                ldb = Bn = N;
                Ak = K;
                Bk = K;
            }
            else {
                lda = An = K;
                ldb = Bn = K;
                Ak = N;
                Bk = N;
            }
            
            ldc = N;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*Ak*batchCount;
            sizeB = ldb*Bk*batchCount;
            sizeC = ldc*N*batchCount;
            
            TESTING_CHECK( magma_dmalloc_cpu(&h_A, sizeA) );
            TESTING_CHECK( magma_dmalloc_cpu(&h_B, sizeB) );
            TESTING_CHECK( magma_dmalloc_cpu(&h_C, sizeC) );
            TESTING_CHECK( magma_dmalloc_cpu(&h_Cmagma, sizeC) );

            TESTING_CHECK( magma_dmalloc(&d_A, ldda*Ak*batchCount) );
            TESTING_CHECK( magma_dmalloc(&d_B, lddb*Bk*batchCount) );
            TESTING_CHECK( magma_dmalloc(&d_C, lddc*N*batchCount)  );

            /* Initialize the matrices */
            lapackf77_dlarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_dlarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_dlarnv( &ione, ISEED, &sizeC, h_C );
            
            // Compute norms for error
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_dlange( "F", &An, &Ak, &h_A[s*lda*Ak], &lda, work );
                Bnorm[s] = lapackf77_dlange( "F", &Bn, &Bk, &h_B[s*ldb*Bk], &ldb, work );
                Cnorm[s] = safe_lapackf77_dlansy( "F", lapack_uplo_const(opts.uplo), &N, &h_C[s*ldc*N], &ldc, work );
            }

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_dsetmatrix( An, Ak*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_dsetmatrix( Bn, Bk*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            magma_dsetmatrix( N, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );
            
            magma_dset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*Ak, batchCount, opts.queue );
            magma_dset_pointer( d_B_array, d_B, lddb, 0, 0, lddb*Bk, batchCount, opts.queue );
            magma_dset_pointer( d_C_array, d_C, lddc, 0, 0, lddc*N,  batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_dsyr2k_batched(opts.uplo, opts.transA, N, K, 
                             alpha, d_A_array, ldda, 
                                    d_B_array, lddb, 
                             beta,  d_C_array, lddc, batchCount, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_dgetmatrix( N, N*batchCount, d_C, lddc, h_Cmagma, ldc, opts.queue );
            
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
                    blasf77_dsyr2k( lapack_uplo_const(opts.uplo),
                                    lapack_trans_const(opts.transA),
                                    &N, &K,
                                    &alpha, h_A + i*lda*Ak, &lda,
                                            h_B + i*ldb*Bk, &ldb,
                                    &beta,  h_C + i*ldc*N, &ldc );
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
                // error = |dC - C| / (2*gamma_{k+2}|A||B| + gamma_2|Cin|)
                magma_error = 0;

                for (int s=0; s < batchCount; s++) {
                    normalize = 2*sqrt(double(K+2))*Anorm[s]*Bnorm[s] + 2*Cnorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    magma_int_t Csize = ldc * N;
                    blasf77_daxpy( &Csize, &c_neg_one, &h_C[s*ldc*N], &ione, &h_Cmagma[s*ldc*N], &ione );
                    error = safe_lapackf77_dlansy( "F", lapack_uplo_const(opts.uplo), &N, &h_Cmagma[s*ldc*N], &ldc, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                }
                
                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                       (long long) batchCount, (long long) N, (long long) K,
                       magma_perf,  1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                       (long long) batchCount, (long long) N, (long long) K,
                       magma_perf,  1000.*magma_time);
            }
            
            magma_free_cpu( h_A  );
            magma_free_cpu( h_B  );
            magma_free_cpu( h_C  );
            magma_free_cpu( h_Cmagma  );

            magma_free( d_A );
            magma_free( d_B );
            magma_free( d_C );

            fflush( stdout);
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    magma_free_cpu( Anorm );
    magma_free_cpu( Bnorm );
    magma_free_cpu( Cnorm );

    magma_free( d_A_array );
    magma_free( d_B_array );
    magma_free( d_C_array );
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
