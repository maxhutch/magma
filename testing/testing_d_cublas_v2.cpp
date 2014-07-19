/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from testing_z_cublas_v2.cpp normal z -> d, Fri Jul 18 17:34:22 2014
       @author Mark Gates
       
       This demonstrates how to use cublas_v2 with magma.
       Simply include cublas_v2.h before magma.h,
       to override its internal include of cublas.h (v1).
*/

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>  // include before magma.h

#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

int main( int argc, char** argv )
{
    magma_init();
    cublasHandle_t handle;
    cudaSetDevice( 0 );
    cublasCreate( &handle );
    
    double *A, *B, *C;
    double *dA, *dB, *dC;
    double error, work[1];
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = { 1, 2, 3, 4 };
    magma_int_t n, lda, ldda, size, info;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("    N   |dC - C|/|C|\n");
    printf("====================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            // for this simple case, all matrices are N-by-N
            n = opts.nsize[itest];
            lda = n;
            ldda = ((n+31)/32)*32;
            
            magma_dmalloc_cpu( &A, lda*n );
            magma_dmalloc_cpu( &B, lda*n );
            magma_dmalloc_cpu( &C, lda*n );
            magma_dmalloc( &dA, ldda*n );
            magma_dmalloc( &dB, ldda*n );
            magma_dmalloc( &dC, ldda*n );
            
            // initialize matrices
            size = lda*n;
            lapackf77_dlarnv( &ione, ISEED, &size, A );
            lapackf77_dlarnv( &ione, ISEED, &size, B );
            lapackf77_dlarnv( &ione, ISEED, &size, C );
            // increase diagonal to be SPD
            for( int i=0; i < n; ++i ) {
                C[i+i*lda] = MAGMA_D_ADD( C[i+i*lda], MAGMA_D_MAKE( n*n, 0 ));
            }
            
            magma_dsetmatrix( n, n, A, lda, dA, ldda );
            magma_dsetmatrix( n, n, B, lda, dB, ldda );
            magma_dsetmatrix( n, n, C, lda, dC, ldda );
            
            // compute with cublas
            cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                         &c_neg_one, dA, ldda, dB, ldda, &c_one, dC, ldda );
            
            magma_dpotrf_gpu( MagmaLower, n, dC, ldda, &info );
            if (info != 0)
                printf("magma_dpotrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            // compute with LAPACK
            blasf77_dgemm( MagmaNoTransStr, MagmaNoTransStr, &n, &n, &n,
                           &c_neg_one, A, &lda, B, &lda, &c_one, C, &lda );
            
            lapackf77_dpotrf( MagmaLowerStr, &n, C, &lda, &info );
            if (info != 0)
                printf("lapackf77_dpotrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            // compute difference, |dC - C| / |C|
            magma_dgetmatrix( n, n, dC, ldda, A, lda );
            blasf77_daxpy( &size, &c_neg_one, C, &ione, A, &ione );
            error = lapackf77_dlange( "F", &n, &n, A, &lda, work )
                  / lapackf77_dlange( "F", &n, &n, C, &lda, work );
            printf( "%5d   %8.2e   %s\n",
                    (int) n, error, (error < tol ? "ok" : "failed"));
            status += ! (error < tol);
            
            magma_free( dA );
            magma_free( dB );
            magma_free( dC );
            magma_free_cpu( A );
            magma_free_cpu( B );
            magma_free_cpu( C );
        }
    }
    
    cublasDestroy( handle );
    magma_finalize();
    return status;
}
