/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from testing_z_cublas_v2.cpp normal z -> s, Fri May 30 10:41:17 2014
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
    
    float *A, *B, *C;
    float *dA, *dB, *dC;
    float error, work[1];
    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t ISEED[4] = { 1, 2, 3, 4 };
    magma_int_t n, lda, ldda, size, info;
    magma_int_t status = 0;
    
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    float tol = opts.tolerance * lapackf77_slamch("E");
    
    printf("    N   |dC - C|/|C|\n");
    printf("====================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            // for this simple case, all matrices are N-by-N
            n = opts.nsize[itest];
            lda = n;
            ldda = ((n+31)/32)*32;
            
            magma_smalloc_cpu( &A, lda*n );
            magma_smalloc_cpu( &B, lda*n );
            magma_smalloc_cpu( &C, lda*n );
            magma_smalloc( &dA, ldda*n );
            magma_smalloc( &dB, ldda*n );
            magma_smalloc( &dC, ldda*n );
            
            // initialize matrices
            size = lda*n;
            lapackf77_slarnv( &ione, ISEED, &size, A );
            lapackf77_slarnv( &ione, ISEED, &size, B );
            lapackf77_slarnv( &ione, ISEED, &size, C );
            // increase diagonal to be SPD
            for( int i=0; i < n; ++i ) {
                C[i+i*lda] = MAGMA_S_ADD( C[i+i*lda], MAGMA_S_MAKE( n*n, 0 ));
            }
            
            magma_ssetmatrix( n, n, A, lda, dA, ldda );
            magma_ssetmatrix( n, n, B, lda, dB, ldda );
            magma_ssetmatrix( n, n, C, lda, dC, ldda );
            
            // compute with cublas
            cublasSgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                         &c_neg_one, dA, ldda, dB, ldda, &c_one, dC, ldda );
            
            magma_spotrf_gpu( MagmaLower, n, dC, ldda, &info );
            if (info != 0)
                printf("magma_spotrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            // compute with LAPACK
            blasf77_sgemm( MagmaNoTransStr, MagmaNoTransStr, &n, &n, &n,
                           &c_neg_one, A, &lda, B, &lda, &c_one, C, &lda );
            
            lapackf77_spotrf( MagmaLowerStr, &n, C, &lda, &info );
            if (info != 0)
                printf("lapackf77_spotrf returned error %d: %s.\n",
                       (int) info, magma_strerror( info ));
            
            // compute difference, |dC - C| / |C|
            magma_sgetmatrix( n, n, dC, ldda, A, lda );
            blasf77_saxpy( &size, &c_neg_one, C, &ione, A, &ione );
            error = lapackf77_slange( "F", &n, &n, A, &lda, work )
                  / lapackf77_slange( "F", &n, &n, C, &lda, work );
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
