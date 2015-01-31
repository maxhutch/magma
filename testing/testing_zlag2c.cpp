/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

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
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlag2c and clag2z
*/
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double error, work[1];
    float serror, swork[1];
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaFloatComplex  s_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t m, n, lda, ldda, size, info;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    magmaFloatComplex   *SA, *SR;
    magmaDoubleComplex   *A,  *R;
    magmaFloatComplex  *dSA;
    magmaDoubleComplex_ptr dA;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("func       M     N     CPU GB/s (ms)       GPU GB/s (ms)     ||R||_F\n");
    printf("=====================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            m = opts.msize[itest];
            n = opts.nsize[itest];
            lda  = m;
            ldda = ((m+31)/32)*32;
            // m*n double-complex loads and m*n single-complex stores (and vice-versa for clag2z)
            gbytes = (real_Double_t) m*n * (sizeof(magmaDoubleComplex) + sizeof(magmaFloatComplex)) / 1e9;
            size = ldda*n;  // ldda >= lda
            
            TESTING_MALLOC_CPU(  SA, magmaFloatComplex,  size );
            TESTING_MALLOC_CPU(   A, magmaDoubleComplex, size );
            TESTING_MALLOC_CPU(  SR, magmaFloatComplex,  size );
            TESTING_MALLOC_CPU(   R, magmaDoubleComplex, size );
            
            TESTING_MALLOC_DEV( dSA, magmaFloatComplex,  size );
            TESTING_MALLOC_DEV(  dA, magmaDoubleComplex, size );
            
            lapackf77_zlarnv( &ione, ISEED, &size,  A );
            lapackf77_clarnv( &ione, ISEED, &size, SA );
            
            magma_zsetmatrix( m, n, A,  lda, dA,  ldda );
            magma_csetmatrix( m, n, SA, lda, dSA, ldda );
            
            /* =====================================================================
               Performs operation using LAPACK zlag2c
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_zlag2c( &m, &n, A, &lda, SA, &lda, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (info != 0)
                printf("lapackf77_zlag2c returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* ====================================================================
               Performs operation using MAGMA zlag2c
               =================================================================== */            
            gpu_time = magma_sync_wtime(0);
            magmablas_zlag2c( m, n, dA, ldda, dSA, ldda, &info );
            gpu_time = magma_sync_wtime(0) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (info != 0)
                printf("magmablas_zlag2c returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            magma_cgetmatrix( m, n, dSA, ldda, SR, lda );
            
            /* =====================================================================
               compute error |SA_magma - SA_lapack|
               should be zero if both are IEEE compliant
               =================================================================== */
            blasf77_caxpy( &size, &s_neg_one, SA, &ione, SR, &ione );
            serror = lapackf77_clange( "Fro", &m, &n, SR, &lda, swork );
            
            printf( "zlag2c %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                    (int) m, (int) n,
                    cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                    serror, (serror == 0 ? "ok" : "failed") );
            status += ! (serror == 0);
            
            
            /* =====================================================================
               Reset matrices
               =================================================================== */
            lapackf77_zlarnv( &ione, ISEED, &size,  A );
            lapackf77_clarnv( &ione, ISEED, &size, SA );
            
            magma_zsetmatrix( m, n, A,  lda, dA,  ldda );
            magma_csetmatrix( m, n, SA, lda, dSA, ldda );
            
            /* =====================================================================
               Performs operation using LAPACK clag2z
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_clag2z( &m, &n, SA, &lda, A, &lda, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (info != 0)
                printf("lapackf77_clag2z returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* ====================================================================
               Performs operation using MAGMA clag2z
               =================================================================== */
            magma_csetmatrix( m, n, SA, lda, dSA, ldda );
            
            gpu_time = magma_sync_wtime(0);
            magmablas_clag2z( m, n, dSA, ldda, dA, ldda, &info );
            gpu_time = magma_sync_wtime(0) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (info != 0)
                printf("magmablas_clag2z returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            magma_zgetmatrix( m, n, dA, ldda, R, lda );
            
            /* =====================================================================
               compute error |A_magma - A_lapack|
               should be zero if both are IEEE compliant
               =================================================================== */
            blasf77_zaxpy( &size, &c_neg_one, A, &ione, R, &ione );
            error = lapackf77_zlange( "Fro", &m, &n, R, &lda, work );
            
            printf( "clag2z %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                    (int) m, (int) n,
                    cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                    error, (error == 0 ? "ok" : "failed") );
            status += ! (error == 0);
            
            TESTING_FREE_CPU(  SA );
            TESTING_FREE_CPU(   A );
            TESTING_FREE_CPU(  SR );
            TESTING_FREE_CPU(   R );
                                 
            TESTING_FREE_DEV( dSA );
            TESTING_FREE_DEV(  dA );
            printf( "\n" );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }
    
    TESTING_FINALIZE();
    return status;
}
