/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:56 2013
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing clanhe
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaFloatComplex *h_A;
    float *h_work;
    magmaFloatComplex *d_A;
    float *d_work;
    magma_int_t N, n2, lda, ldda;
    magma_int_t idist    = 3;  // normal distribution (otherwise max norm is always ~ 1)
    magma_int_t ISEED[4] = {0,0,0,1};
    float      error, norm_magma, norm_lapack;
    magma_int_t status = 0;
    bool        not_supported;

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    const char* uplo[] = { MagmaLowerStr, MagmaUpperStr };
    const char* norm[] = { MagmaInfNormStr, MagmaOneNormStr, MagmaMaxNormStr };
    
    // inf-norm not supported on Tesla (CUDA arch 1.x)
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200 ) {
        printf("!!!! NOTE: %s and %s norm are not supported on CUDA architecture %d (less than 200).\n"
               "!!!! It should report \"parameter number 1 had an illegal value\" below.\n\n",
               MagmaInfNormStr, MagmaOneNormStr, arch );
        for( int inorm = 0; inorm < 2; ++inorm ) {
        for( int iuplo = 0; iuplo < 2; ++iuplo ) {
            printf( "Testing that magmablas_clanhe( %s, %s, ... ) returns -1 error...\n",
                    norm[inorm], uplo[iuplo] );
            norm_magma = magmablas_clanhe( *norm[inorm], *uplo[iuplo], 1, NULL, 1, NULL );
            if ( norm_magma != -1 ) {
                printf( "expected magmablas_clanhe to return -1 error, but got %f\n", norm_magma );
                status = 1;
            }
        }}
        printf( "...return values %s\n\n", (status == 0 ? "ok" : "failed") );
    }
    
    printf("    N   norm   uplo    CPU GByte/s (ms)    GPU GByte/s (ms)    error   \n");
    printf("=======================================================================\n");
    for( int i = 0; i < opts.ntest; ++i ) {
        for( int inorm = 0; inorm < 3; ++inorm ) {
        for( int iuplo = 0; iuplo < 2; ++iuplo ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            not_supported = ((arch < 200) &&
                ( lapackf77_lsame( norm[inorm], MagmaInfNormStr ) ||
                  lapackf77_lsame( norm[inorm], MagmaOneNormStr )    ));
            if ( not_supported ) {
                continue;
            }
            
            N   = opts.nsize[i];
            lda = N;
            n2  = lda*N;
            ldda = roundup( N, opts.pad );
            // read upper or lower triangle
            gbytes = 0.5*(N+1)*N*sizeof(magmaFloatComplex) / 1e9;
            
            TESTING_MALLOC_CPU( h_A,    magmaFloatComplex, n2 );
            TESTING_MALLOC_CPU( h_work, float, N );
            
            TESTING_MALLOC_DEV( d_A,    magmaFloatComplex, ldda*N );
            TESTING_MALLOC_DEV( d_work, float, N );
            
            /* Initialize the matrix */
            lapackf77_clarnv( &idist, ISEED, &n2, h_A );
            //magma_cmake_hermitian( N, h_A, lda );
            magma_csetmatrix( N, N, h_A, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            norm_magma = magmablas_clanhe( *norm[inorm], *uplo[iuplo], N, d_A, ldda, d_work );
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (norm_magma < 0)
                printf("magmablas_clanhe returned error %f: %s.\n",
                       norm_magma, magma_strerror( (int) norm_magma ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            cpu_time = magma_wtime();
            norm_lapack = lapackf77_clanhe( norm[inorm], uplo[iuplo], &N, h_A, &lda, h_work );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (norm_lapack < 0)
                printf("lapackf77_clanhe returned error %f: %s.\n",
                       norm_lapack, magma_strerror( (int) norm_lapack ));
            
            /* =====================================================================
               Check the result compared to LAPACK
               =================================================================== */
            if ( lapackf77_lsame( norm[inorm], MagmaMaxNormStr ))
                error = fabs( norm_magma - norm_lapack );
            else
                error = fabs( norm_magma - norm_lapack ) / norm_lapack;
            
            printf("%5d   %4s   %5s   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e  %s\n",
                   (int) N, norm[inorm], uplo[iuplo],
                   cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                   error, (error < tol ? "ok" : "failed") );
            status |= ! (error < tol);
            
            TESTING_FREE_CPU( h_A    );
            TESTING_FREE_CPU( h_work );
            
            TESTING_FREE_DEV( d_A    );
            TESTING_FREE_DEV( d_work );
        }}} // end iuplo, inorm, iter
        printf( "\n" );
    }

    TESTING_FINALIZE();
    return status;
}
