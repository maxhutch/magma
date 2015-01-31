/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
       @generated from testing_zlag2c.cpp mixed zc -> ds, Fri Jan 30 19:00:22 2015
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
   -- Testing dlag2s and slag2d
*/
int main( int argc, char** argv )
{
    TESTING_INIT();
    
    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double error, work[1];
    float serror, swork[1];
    double c_neg_one = MAGMA_D_NEG_ONE;
    float  s_neg_one = MAGMA_S_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t m, n, lda, ldda, size, info;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t status = 0;
    float   *SA, *SR;
    double   *A,  *R;
    float  *dSA;
    magmaDouble_ptr dA;
    
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
            // m*n double-real loads and m*n single-real stores (and vice-versa for slag2d)
            gbytes = (real_Double_t) m*n * (sizeof(double) + sizeof(float)) / 1e9;
            size = ldda*n;  // ldda >= lda
            
            TESTING_MALLOC_CPU(  SA, float,  size );
            TESTING_MALLOC_CPU(   A, double, size );
            TESTING_MALLOC_CPU(  SR, float,  size );
            TESTING_MALLOC_CPU(   R, double, size );
            
            TESTING_MALLOC_DEV( dSA, float,  size );
            TESTING_MALLOC_DEV(  dA, double, size );
            
            lapackf77_dlarnv( &ione, ISEED, &size,  A );
            lapackf77_slarnv( &ione, ISEED, &size, SA );
            
            magma_dsetmatrix( m, n, A,  lda, dA,  ldda );
            magma_ssetmatrix( m, n, SA, lda, dSA, ldda );
            
            /* =====================================================================
               Performs operation using LAPACK dlag2s
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_dlag2s( &m, &n, A, &lda, SA, &lda, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (info != 0)
                printf("lapackf77_dlag2s returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* ====================================================================
               Performs operation using MAGMA dlag2s
               =================================================================== */            
            gpu_time = magma_sync_wtime(0);
            magmablas_dlag2s( m, n, dA, ldda, dSA, ldda, &info );
            gpu_time = magma_sync_wtime(0) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (info != 0)
                printf("magmablas_dlag2s returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            magma_sgetmatrix( m, n, dSA, ldda, SR, lda );
            
            /* =====================================================================
               compute error |SA_magma - SA_lapack|
               should be zero if both are IEEE compliant
               =================================================================== */
            blasf77_saxpy( &size, &s_neg_one, SA, &ione, SR, &ione );
            serror = lapackf77_slange( "Fro", &m, &n, SR, &lda, swork );
            
            printf( "dlag2s %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                    (int) m, (int) n,
                    cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                    serror, (serror == 0 ? "ok" : "failed") );
            status += ! (serror == 0);
            
            
            /* =====================================================================
               Reset matrices
               =================================================================== */
            lapackf77_dlarnv( &ione, ISEED, &size,  A );
            lapackf77_slarnv( &ione, ISEED, &size, SA );
            
            magma_dsetmatrix( m, n, A,  lda, dA,  ldda );
            magma_ssetmatrix( m, n, SA, lda, dSA, ldda );
            
            /* =====================================================================
               Performs operation using LAPACK slag2d
               =================================================================== */
            cpu_time = magma_wtime();
            lapackf77_slag2d( &m, &n, SA, &lda, A, &lda, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (info != 0)
                printf("lapackf77_slag2d returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            /* ====================================================================
               Performs operation using MAGMA slag2d
               =================================================================== */
            magma_ssetmatrix( m, n, SA, lda, dSA, ldda );
            
            gpu_time = magma_sync_wtime(0);
            magmablas_slag2d( m, n, dSA, ldda, dA, ldda, &info );
            gpu_time = magma_sync_wtime(0) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (info != 0)
                printf("magmablas_slag2d returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            magma_dgetmatrix( m, n, dA, ldda, R, lda );
            
            /* =====================================================================
               compute error |A_magma - A_lapack|
               should be zero if both are IEEE compliant
               =================================================================== */
            blasf77_daxpy( &size, &c_neg_one, A, &ione, R, &ione );
            error = lapackf77_dlange( "Fro", &m, &n, R, &lda, work );
            
            printf( "slag2d %5d %5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
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
