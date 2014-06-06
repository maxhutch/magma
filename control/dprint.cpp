/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @author Mark Gates
       @generated from zprint.cpp normal z -> d, Fri May 30 10:41:11 2014

*/
#include "common_magma.h"

#define PRECISION_d

#define A(i,j) (A + i + j*lda)

// -------------------------
// Prints a matrix that is on the CPU host.
extern "C"
void magma_dprint( magma_int_t m, magma_int_t n, const double *A, magma_int_t lda )
{
    if ( magma_is_devptr( A ) == 1 ) {
        fprintf( stderr, "ERROR: dprint called with device pointer.\n" );
        exit(1);
    }
    
    double c_zero = MAGMA_D_ZERO;
    
    if ( m == 1 ) {
        printf( "[ " );
    }
    else {
        printf( "[\n" );
    }
    for( int i = 0; i < m; ++i ) {
        for( int j = 0; j < n; ++j ) {
            if ( MAGMA_D_EQUAL( *A(i,j), c_zero )) {
                printf( "   0.    " );
            }
            else {
#if defined(PRECISION_z) || defined(PRECISION_c)
                printf( " %8.4f+%8.4fi", MAGMA_D_REAL( *A(i,j) ), MAGMA_D_IMAG( *A(i,j) ));
#else
                printf( " %8.4f", MAGMA_D_REAL( *A(i,j) ));
#endif
            }
        }
        if ( m > 1 ) {
            printf( "\n" );
        }
        else {
            printf( " " );
        }
    }
    printf( "];\n" );
}

// -------------------------
// Prints a matrix that is on the GPU device.
// Internally allocates memory on host, copies it to the host, prints it,
// and de-allocates host memory.
extern "C"
void magma_dprint_gpu( magma_int_t m, magma_int_t n, const double *dA, magma_int_t ldda )
{
    if ( magma_is_devptr( dA ) == 0 ) {
        fprintf( stderr, "ERROR: dprint_gpu called with host pointer.\n" );
        exit(1);
    }
    
    magma_int_t lda = m;
    double* A;
    magma_dmalloc_cpu( &A, lda*n );
    magma_dgetmatrix( m, n, dA, ldda, A, lda );
    magma_dprint( m, n, A, lda );
    magma_free_cpu( A );
}
