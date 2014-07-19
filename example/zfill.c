#include <stdio.h>
#include <stdlib.h>

#include "zfill.h"


// ------------------------------------------------------------
// Replace with your code to initialize the A matrix.
// This simply initializes it to random values.
// Note that A is stored column-wise, not row-wise.
//
// m   - number of rows,    m >= 0.
// n   - number of columns, n >= 0.
// A   - m-by-n array of size lda*n.
// lda - leading dimension of A, lda >= m.
//
// When lda > m, rows (m, ..., lda-1) below the bottom of the matrix are ignored.
// This is helpful for working with sub-matrices, and for aligning the top
// of columns to memory boundaries (or avoiding such alignment).
// Significantly better memory performance is achieved by having the outer loop
// over columns (j), and the inner loop over rows (i), than the reverse.
void zfill_matrix(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    
    magma_int_t i, j;
    for( j=0; j < n; ++j ) {
        for( i=0; i < m; ++i ) {
            A(i,j) = MAGMA_Z_MAKE( rand() / ((double) RAND_MAX),    // real part
                                   rand() / ((double) RAND_MAX) );  // imag part
        }
    }
}


// ------------------------------------------------------------
// Replace with your code to initialize the X rhs.
void zfill_rhs(
    magma_int_t m, magma_int_t nrhs, magmaDoubleComplex *X, magma_int_t ldx )
{
    zfill_matrix( m, nrhs, X, ldx );
}


// ------------------------------------------------------------
// Replace with your code to initialize the dA matrix on the GPU device.
// This simply leverages the CPU version above to initialize it to random values,
// and copies the matrix to the GPU.
void zfill_matrix_gpu(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldda )
{
    magmaDoubleComplex *A;
    int lda = ldda;
    magma_zmalloc_cpu( &A, m*lda );
    if ( A == NULL ) {
        fprintf( stderr, "malloc failed\n" );
        return;
    }
    zfill_matrix( m, n, A, lda );
    magma_zsetmatrix( m, n, A, lda, dA, ldda );
    magma_free_cpu( A );
}


// ------------------------------------------------------------
// Replace with your code to initialize the dX rhs on the GPU device.
void zfill_rhs_gpu(
    magma_int_t m, magma_int_t nrhs, magmaDoubleComplex *dX, magma_int_t lddx )
{
    zfill_matrix_gpu( m, nrhs, dX, lddx );
}
