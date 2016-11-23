/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from testing/testing_ztrmm_batched.cpp, normal z -> c, Sun Nov 20 20:20:37 2016
       @author Chongxiao Cao
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
#include "../control/magma_threadsetting.h"
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ctrmm_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, cpu_perf, cpu_time;
    float          error, magma_error, normalize, work[1];
    magma_int_t M, N, NN;
    magma_int_t Ak, Akk;
    magma_int_t sizeA, sizeB;
    magma_int_t lda, ldb, ldda, lddb;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    
    magmaFloatComplex **dA_array, **dB_array;
    magmaFloatComplex *h_A, *h_B, *h_Bmagma;
    magmaFloatComplex_ptr d_A, d_B;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex alpha = MAGMA_C_MAKE(  0.29, -0.86 );
    int status = 0;
    
    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    magma_int_t batchCount = opts.batchcount; 

    float *Anorm, *Bnorm;
    TESTING_CHECK( magma_smalloc_cpu( &Anorm, batchCount ));
    TESTING_CHECK( magma_smalloc_cpu( &Bnorm, batchCount ));
    
    TESTING_CHECK( magma_malloc((void**)&dA_array, batchCount*sizeof(magmaFloatComplex*)) );
    TESTING_CHECK( magma_malloc((void**)&dB_array, batchCount*sizeof(magmaFloatComplex*)) );
    
    // See testing_cgemm about tolerance.
    float eps = lapackf77_slamch("E");
    float tol = 3*eps;
    
    printf("%% If running lapack (option --lapack), MAGMA error is computed\n"
           "%% relative to CPU BLAS result.\n\n");
    printf("%% side = %s, uplo = %s, transA = %s, diag = %s\n",
           lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
           lapack_trans_const(opts.transA), lapack_diag_const(opts.diag) );
    printf("%% BatchCount     M     N   MAGMA Gflop/s (ms)   CPU Gflop/s (ms)   MAGMA error\n");
    printf("%%=============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            gflops = batchCount * FLOPS_CTRMM(opts.side, M, N) / 1e9;

            if ( opts.side == MagmaLeft ) {
                lda = M;
                Ak = M;
            }
            else {
                lda = N;
                Ak = N;
            }
            
            ldb = M;
            Akk = Ak * batchCount;
            NN = N * batchCount; 
            
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddb = magma_roundup( ldb, opts.align );  // multiple of 32 by default
            
            sizeA = lda*Ak*batchCount;
            sizeB = ldb*N*batchCount;
            
            TESTING_CHECK( magma_cmalloc_cpu( &h_A,       lda*Ak*batchCount ) );
            TESTING_CHECK( magma_cmalloc_cpu( &h_B,       ldb*N*batchCount  ) );
            TESTING_CHECK( magma_cmalloc_cpu( &h_Bmagma, ldb*N*batchCount  ) );
            
            TESTING_CHECK( magma_cmalloc( &d_A, ldda*Ak*batchCount ) );
            TESTING_CHECK( magma_cmalloc( &d_B, lddb*N*batchCount  ) );
            
            /* Initialize the matrices */
            lapackf77_clarnv( &ione, ISEED, &sizeA, h_A );
            lapackf77_clarnv( &ione, ISEED, &sizeB, h_B );

            // Compute norms for error
            for (int s = 0; s < batchCount; ++s) {
                Anorm[s] = lapackf77_clantr( "F", lapack_uplo_const(opts.uplo),
                                                  lapack_diag_const(opts.diag),
                                             &Ak, &Ak, &h_A[s*lda*Ak], &lda, work );
                Bnorm[s] = lapackf77_clange( "F", &M, &N, &h_B[s*ldb*N], &ldb, work );
            }
            
            /* =====================================================================
               Performs operation using CUBLAS
               =================================================================== */
            magma_csetmatrix( Ak, Akk, h_A, lda, d_A, ldda, opts.queue );
            magma_csetmatrix( M,  NN,  h_B, ldb, d_B, lddb, opts.queue );
            
            magma_cset_pointer( dA_array, d_A, ldda, 0, 0, ldda*Ak, batchCount, opts.queue );
            magma_cset_pointer( dB_array, d_B, lddb, 0, 0, lddb*N,  batchCount, opts.queue );
            
            magma_time = magma_sync_wtime( opts.queue );
            magmablas_ctrmm_batched( 
                    opts.side, opts.uplo, opts.transA, opts.diag, 
                    M, N, 
                    alpha, dA_array, ldda, 
                           dB_array, lddb, 
                    batchCount, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_cgetmatrix( M, NN, d_B, lddb, h_Bmagma, ldb, opts.queue );
            
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
                for (int s=0; s < batchCount; s++){
                    blasf77_ctrmm( lapack_side_const(opts.side), lapack_uplo_const(opts.uplo),
                                   lapack_trans_const(opts.transA), lapack_diag_const(opts.diag),
                                   &M, &N,
                                   &alpha, h_A+s*lda*Ak, &lda,
                                           h_B+s*ldb*N,  &ldb );
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
                // error = |dB - B| / (gamma_{k}|A||Bin|); k = Ak; no beta
                magma_error = 0;
                
                for (int s  = 0; s < batchCount; s++) {
                    normalize = sqrt(float(Ak))*Anorm[s]*Bnorm[s];
                    if (normalize == 0)
                        normalize = 1;
                    magma_int_t Bsize = ldb*N;
                    blasf77_caxpy( &Bsize, &c_neg_one, &h_B[s*ldb*N], &ione, &h_Bmagma[s*ldb*N], &ione );
                    error = lapackf77_clange( "F", &M, &N, &h_Bmagma[s*ldb*N], &ldb, work )
                          / normalize;
                    magma_error = magma_max_nan( error, magma_error );
                }
                bool okay = (magma_error < tol);
                status += ! okay;
                
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long)batchCount,
                       (long long)M, (long long)N,
                       magma_perf, 1000.*magma_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, (okay ? "ok" : "failed"));
            }
            else {
                printf("  %10lld %5lld %5lld   %7.2f (%7.2f)     ---   (  ---  )     ---\n",
                       (long long)batchCount,
                       (long long)M, (long long)N,
                       magma_perf, 1000.*magma_time);
            }
            
            magma_free_cpu( h_A );
            magma_free_cpu( h_B );
            magma_free_cpu( h_Bmagma );
            
            magma_free( d_A );
            magma_free( d_B );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_free( dA_array );
    magma_free( dB_array );
    
    magma_free_cpu( Anorm );
    magma_free_cpu( Bnorm );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
