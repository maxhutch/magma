/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_zhemm_batched.cpp, normal z -> s, Sun Nov 20 20:20:38 2016
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
#include "magma_operators.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ssymm_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    float          error, magma_error, normalize, work[1];
    magma_int_t M, N;
    magma_int_t An;
    magma_int_t sizeA, sizeB, sizeC;
    magma_int_t lda, ldb, ldc, ldda, lddb, lddc;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magma_int_t batchCount;

    float *h_A, *h_B, *h_C, *h_Cmagma;
    float *d_A, *d_B, *d_C;
    float c_neg_one = MAGMA_S_NEG_ONE;
    float alpha = MAGMA_S_MAKE(  0.29, -0.86 );
    float beta  = MAGMA_S_MAKE( -0.48,  0.38 );
    float **d_A_array = NULL;
    float **d_B_array = NULL;
    float **d_C_array = NULL;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check; // check (-c) implies lapack (-l)
    batchCount = opts.batchcount;
    
    cpu_perf = cpu_time = 0.0;
    TESTING_CHECK( magma_malloc((void**)&d_A_array, batchCount * sizeof(float*)) );
    TESTING_CHECK( magma_malloc((void**)&d_B_array, batchCount * sizeof(float*)) );
    TESTING_CHECK( magma_malloc((void**)&d_C_array, batchCount * sizeof(float*)) );
    
    float *Anorm, *Bnorm, *Cnorm;
    TESTING_CHECK( magma_smalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Bnorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Cnorm, batchCount ));
    
    // See testing_sgemm about tolerance.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;
    
    printf("%% If running lapack (option --lapack), MAGMA error is computed relative to CPU BLAS result.\n\n"
           "%% side = %s, uplo = %s\n",
           lapack_side_const(opts.side),
           lapack_uplo_const(opts.uplo));
    printf("%% BatchCount     M     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = FLOPS_SSYMM(opts.side, M, N) / 1e9 * batchCount;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                An = M;
            } else {
                lda = N;
                An = N;
            }
            ldb = ldc = M;
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            lddc = magma_roundup( ldc, opts.align );  // multiple of 32 by default

            sizeA = lda*An*batchCount;
            sizeB = ldb*N*batchCount;
            sizeC = ldc*N*batchCount;
            
            TESTING_CHECK( magma_smalloc_cpu(&h_A, sizeA) );
            TESTING_CHECK( magma_smalloc_cpu(&h_B, sizeB) );
            TESTING_CHECK( magma_smalloc_cpu(&h_C, sizeC) );
            TESTING_CHECK( magma_smalloc_cpu(&h_Cmagma, sizeC) );
            
            TESTING_CHECK( magma_smalloc(&d_A, ldda*An*batchCount) );
            TESTING_CHECK( magma_smalloc(&d_B, lddb*N*batchCount) );
            TESTING_CHECK( magma_smalloc(&d_C, lddc*N*batchCount) );
            
            /* Initialize the matrices */
            lapackf77_slarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_slarnv( &ione, ISEED, &sizeB, h_B );
            lapackf77_slarnv( &ione, ISEED, &sizeC, h_C );
            
            // Compute norms for error
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = safe_lapackf77_slansy( "F", lapack_uplo_const(opts.uplo), &An, &h_A[s*lda*An], &lda, work );
                Bnorm[s] = lapackf77_slange( "F", &M, &N, &h_B[s*ldb*N], &ldb, work );
                Cnorm[s] = lapackf77_slange( "F", &M, &N, &h_C[s*ldc*N], &ldc, work );
            }

            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_ssetmatrix( An, An*batchCount, h_A, lda, d_A, ldda, opts.queue );
            magma_ssetmatrix( M, N*batchCount, h_B, ldb, d_B, lddb, opts.queue );
            magma_ssetmatrix( M, N*batchCount, h_C, ldc, d_C, lddc, opts.queue );
            
            magma_sset_pointer( d_A_array, d_A, ldda, 0, 0, ldda*An, batchCount, opts.queue );
            magma_sset_pointer( d_B_array, d_B, lddb, 0, 0, lddb*N, batchCount, opts.queue );
            magma_sset_pointer( d_C_array, d_C, lddc, 0, 0, lddc*N, batchCount, opts.queue );

            magma_time = magma_sync_wtime( opts.queue );
            magmablas_ssymm_batched( 
                    opts.side, opts.uplo, M, N, 
                    alpha, d_A_array, ldda, 
                           d_B_array, lddb, 
                    beta,  d_C_array, lddc, 
                    batchCount, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            magma_sgetmatrix( M, N*batchCount, d_C, lddc, h_Cmagma, ldc, opts.queue );
            
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
                    blasf77_ssymm( lapack_side_const(opts.side),
                                   lapack_uplo_const(opts.uplo),
                                   &M, &N,
                                   &alpha, h_A + i*lda*An, &lda,
                                           h_B + i*ldb*N, &ldb,
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
                // compute error compared to lapack
                // error = |dC - C| / (gamma_{k+2}|A||B| + gamma_2|Cin|); k = Am
                magma_error = 0;
                
                for (int s=0; s < batchCount; s++) {
                    normalize = sqrt(float(An+2))*fabs(alpha)*Anorm[s]*Bnorm[s] + 2*fabs(beta)*Cnorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    magma_int_t Csize = ldc * N;
                    blasf77_saxpy( &Csize, &c_neg_one, &h_C[s*ldc*N], &ione, &h_Cmagma[s*ldc*N], &ione );
                    error = lapackf77_slange( "F", &M, &N, &h_Cmagma[s*ldc*N], &ldc, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                }

                bool okay = (magma_error < tol);
                status += ! okay;
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                   (long long)batchCount, (long long)M, (long long)N,
                   magma_perf,  1000.*magma_time,
                   cpu_perf,    1000.*cpu_time,
                   magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                   (long long)batchCount, (long long)M, (long long)N,
                   magma_perf,  1000.*magma_time );
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_C );
            magma_free_cpu( h_Cmagma );
            
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
