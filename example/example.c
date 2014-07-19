// This is a simple standalone example. See README.txt

#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"     // if you need CUBLAS, include before magma.h
#include "magma.h"
#include "magma_lapack.h"  // if you need BLAS & LAPACK

#include "zfill.h"         // code to fill matrix; replace with your application code


// ------------------------------------------------------------
// Solve A * X = B, where A and X are stored in CPU host memory.
// Internally, MAGMA transfers data to the GPU device
// and uses a hybrid CPU + GPU algorithm.
void cpu_interface( magma_int_t n, magma_int_t nrhs )
{
    magmaDoubleComplex *A=NULL, *X=NULL;
    magma_int_t *ipiv=NULL;
    magma_int_t lda  = n;
    magma_int_t ldx  = lda;
    magma_int_t info = 0;
    
    // magma malloc_cpu routines are type-safe and align to memory boundaries,
    // but you can use malloc or new if you prefer.
    magma_zmalloc_cpu( &A, lda*n );
    magma_zmalloc_cpu( &X, ldx*nrhs );
    magma_imalloc_cpu( &ipiv, n );
    if ( A == NULL || X == NULL || ipiv == NULL ) {
        fprintf( stderr, "malloc failed\n" );
        goto cleanup;
    }
    
    zfill_matrix( n, n, A, lda );
    zfill_rhs( n, nrhs, X, ldx );
    
    magma_zgesv( n, 1, A, lda, ipiv, X, lda, &info );
    if ( info != 0 ) {
        fprintf( stderr, "magma_zgesv failed with info=%d\n", info );
    }
    
    // TODO: use result in X
    
cleanup:
    magma_free_cpu( A );
    magma_free_cpu( X );
    magma_free_cpu( ipiv );
}


// ------------------------------------------------------------
// Solve dA * dX = dB, where dA and dX are stored in GPU device memory.
// Internally, MAGMA uses a hybrid CPU + GPU algorithm.
void gpu_interface( magma_int_t n, magma_int_t nrhs )
{
    magmaDoubleComplex *dA=NULL, *dX=NULL;
    magma_int_t *ipiv=NULL;
    magma_int_t ldda = ((n+31)/32)*32;  // round up to multiple of 32 for best GPU performance
    magma_int_t lddx = ldda;
    magma_int_t info = 0;
    
    // magma malloc (GPU) routines are type-safe,
    // but you can use cudaMalloc if you prefer.
    magma_zmalloc( &dA, ldda*n );
    magma_zmalloc( &dX, lddx*nrhs );
    magma_imalloc_cpu( &ipiv, n );  // ipiv always on CPU
    if ( dA == NULL || dX == NULL || ipiv == NULL ) {
        fprintf( stderr, "malloc failed\n" );
        goto cleanup;
    }
    
    zfill_matrix_gpu( n, n, dA, ldda );
    zfill_rhs_gpu( n, nrhs, dX, lddx );
    
    magma_zgesv_gpu( n, 1, dA, ldda, ipiv, dX, ldda, &info );
    if ( info != 0 ) {
        fprintf( stderr, "magma_zgesv_gpu failed with info=%d\n", info );
    }
    
    // TODO: use result in dX
    
cleanup:
    magma_free( dA );
    magma_free( dX );
    magma_free_cpu( ipiv );
}


// ------------------------------------------------------------
int main( int argc, char** argv )
{
    magma_init();
    
    magma_int_t n = 1000;
    magma_int_t nrhs = 1;
    
    printf( "using MAGMA CPU interface\n" );
    cpu_interface( n, nrhs );

    printf( "using MAGMA GPU interface\n" );
    gpu_interface( n, nrhs );
    
    magma_finalize();
    return 0;
}
