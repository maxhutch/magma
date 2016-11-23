/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Mark Gates
       @precisions mixed zc -> ds
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlag2c and clag2z
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double error, work[1];
    float serror, swork[1];
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaFloatComplex  s_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t m, n, lda, ldda, size, info;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magmaFloatComplex   *SA, *SR;
    magmaDoubleComplex   *A,  *R;
    magmaFloatComplex_ptr dSA;
    magmaDoubleComplex_ptr dA;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    printf("%% func     M     N     CPU GB/s (ms)       GPU GB/s (ms)     ||R||_F\n");
    printf("%%====================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m = opts.msize[itest];
            n = opts.nsize[itest];
            lda  = m;
            ldda = magma_roundup( m, opts.align );  // multiple of 32 by default
            // m*n double-complex loads and m*n single-complex stores (and vice-versa for clag2z)
            gbytes = (real_Double_t) m*n * (sizeof(magmaDoubleComplex) + sizeof(magmaFloatComplex)) / 1e9;
            size = ldda*n;  // ldda >= lda
            
            TESTING_CHECK( magma_cmalloc_cpu( &SA, size ));
            TESTING_CHECK( magma_zmalloc_cpu( &A, size ));
            TESTING_CHECK( magma_cmalloc_cpu( &SR, size ));
            TESTING_CHECK( magma_zmalloc_cpu( &R, size ));
            
            TESTING_CHECK( magma_cmalloc( &dSA, size ));
            TESTING_CHECK( magma_zmalloc( &dA, size ));
            
            lapackf77_zlarnv( &ione, ISEED, &size,  A );
            lapackf77_clarnv( &ione, ISEED, &size, SA );
            
            magma_zsetmatrix( m, n, A,  lda, dA,  ldda, opts.queue );
            magma_csetmatrix( m, n, SA, lda, dSA, ldda, opts.queue );
            
            /* =====================================================================
               Performs operation using LAPACK zlag2c
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_zlag2c( &m, &n, A, &lda, SA, &lda, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (info != 0) {
                printf("lapackf77_zlag2c returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* ====================================================================
               Performs operation using MAGMA zlag2c
               =================================================================== */
            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_zlag2c( m, n, dA, ldda, dSA, ldda, opts.queue, &info );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (info != 0) {
                printf("magmablas_zlag2c returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            magma_cgetmatrix( m, n, dSA, ldda, SR, lda, opts.queue );
            
            /* =====================================================================
               compute error |SA_magma - SA_lapack|
               should be zero if both are IEEE compliant
               =================================================================== */
            blasf77_caxpy( &size, &s_neg_one, SA, &ione, SR, &ione );
            serror = lapackf77_clange( "Fro", &m, &n, SR, &lda, swork );
            
            printf( "zlag2c %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                    (long long) m, (long long) n,
                    cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                    serror, (serror == 0 ? "ok" : "failed") );
            status += ! (serror == 0);
            
            /* =====================================================================
               Reset matrices
               =================================================================== */
            lapackf77_zlarnv( &ione, ISEED, &size,  A );
            lapackf77_clarnv( &ione, ISEED, &size, SA );
            
            magma_zsetmatrix( m, n, A,  lda, dA,  ldda, opts.queue );
            magma_csetmatrix( m, n, SA, lda, dSA, ldda, opts.queue );
            
            /* =====================================================================
               Performs operation using LAPACK clag2z
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_clag2z( &m, &n, SA, &lda, A, &lda, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (info != 0) {
                printf("lapackf77_clag2z returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* ====================================================================
               Performs operation using MAGMA clag2z
               =================================================================== */
            magma_csetmatrix( m, n, SA, lda, dSA, ldda, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_clag2z( m, n, dSA, ldda, dA, ldda, opts.queue, &info );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (info != 0) {
                printf("magmablas_clag2z returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            magma_zgetmatrix( m, n, dA, ldda, R, lda, opts.queue );
            
            /* =====================================================================
               compute error |A_magma - A_lapack|
               should be zero if both are IEEE compliant
               =================================================================== */
            blasf77_zaxpy( &size, &c_neg_one, A, &ione, R, &ione );
            error = lapackf77_zlange( "Fro", &m, &n, R, &lda, work );
            
            printf( "clag2z %5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                    (long long) m, (long long) n,
                    cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                    error, (error == 0 ? "ok" : "failed") );
            status += ! (error == 0);
            
            magma_free_cpu( SA );
            magma_free_cpu( A );
            magma_free_cpu( SR );
            magma_free_cpu( R );
            
            magma_free( dSA );
            magma_free( dA );
            printf( "\n" );
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
