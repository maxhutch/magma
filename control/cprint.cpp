/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Mark Gates
       @generated c Tue Dec 17 13:18:17 2013

*/
#include "common_magma.h"

#define PRECISION_c

#define A(i,j) (A + i + j*lda)

// -------------------------
// Prints a matrix that is on the CPU host.
extern "C"
void magma_cprint( magma_int_t m, magma_int_t n, const magmaFloatComplex *A, magma_int_t lda )
{
    if ( magma_is_devptr( A ) == 1 ) {
        fprintf( stderr, "ERROR: cprint called with device pointer.\n" );
        exit(1);
    }
    
    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    
    if ( m == 1 ) {
        printf( "[ " );
    }
    else {
        printf( "[\n" );
    }
    for( int i = 0; i < m; ++i ) {
        for( int j = 0; j < n; ++j ) {
            if ( MAGMA_C_EQUAL( *A(i,j), c_zero )) {
                printf( "   0.    " );
            }
            else {
#if defined(PRECISION_z) || defined(PRECISION_c)
                printf( " %8.4f+%8.4fi", MAGMA_C_REAL( *A(i,j) ), MAGMA_C_IMAG( *A(i,j) ));
#else
                printf( " %8.4f", MAGMA_C_REAL( *A(i,j) ));
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
void magma_cprint_gpu( magma_int_t m, magma_int_t n, const magmaFloatComplex *dA, magma_int_t ldda )
{
    if ( magma_is_devptr( dA ) == 0 ) {
        fprintf( stderr, "ERROR: cprint_gpu called with host pointer.\n" );
        exit(1);
    }
    
    magma_int_t lda = m;
    magmaFloatComplex* A;
    magma_cmalloc_cpu( &A, lda*n );
    magma_cgetmatrix( m, n, dA, ldda, A, lda );
    magma_cprint( m, n, A, lda );
    magma_free_cpu( A );
}
