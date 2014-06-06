/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal z -> s d c
       @author Mark Gates
*/
#include "common_magma.h"
#include <assert.h>

#define NB 64

/*
    Matrix is m x m, and is divided into block rows, each NB x m.
    Each block has NB threads.
    Each thread copies one row, iterating across all columns below diagonal.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
__global__ void
zsymmetrize_lower( int m, magmaDoubleComplex *dA, int ldda )
{
    // dA iterates across row i and dAT iterates down column i.
    int i = blockIdx.x*NB + threadIdx.x;
    magmaDoubleComplex *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        magmaDoubleComplex *dAend = dA + i*ldda;
        while( dA < dAend ) {
            *dAT = cuConj(*dA);  // upper := lower
            dA  += ldda;
            dAT += 1;
        }
    }
}


// only difference with _lower version is direction dA=dAT instead of dAT=dA.
__global__ void
zsymmetrize_upper( int m, magmaDoubleComplex *dA, int ldda )
{
    // dA iterates across row i and dAT iterates down column i.
    int i = blockIdx.x*NB + threadIdx.x;
    magmaDoubleComplex *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        magmaDoubleComplex *dAend = dA + i*ldda;
        while( dA < dAend ) {
            *dA = cuConj(*dAT);  // lower := upper
            dA  += ldda;
            dAT += 1;
        }
    }
}


extern "C" void
magmablas_zsymmetrize( char uplo, magma_int_t m, magmaDoubleComplex *dA, magma_int_t ldda )
{
/*
    Purpose
    =======
    
    ZSYMMETRIZE copies lower triangle to upper triangle, or vice-versa,
    to make dA a general representation of a symmetric matrix.
    
    Arguments
    =========
    
    UPLO    (input) CHARACTER*1
            Specifies the part of the matrix dA that is valid on input.
            = 'U':      Upper triangular part
            = 'L':      Lower triangular part
    
    M       (input) INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    dA      (input/output) COMPLEX DOUBLE PRECISION array, dimension (LDDA,N)
            The m by m matrix dA.
    
    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    =====================================================================   */

    //printf( "m %d, grid %d, threads %d\n", m, grid.x, threads.x );
    if ( m == 0 )
        return;
    
    assert( m >= 0 );
    assert( ldda >= m );
    
    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB );
    
    if ( (uplo == 'U') || (uplo == 'u') ) {
        zsymmetrize_upper<<< grid, threads, 0, magma_stream >>>( m, dA, ldda );
    }
    else if ( (uplo == 'L') || (uplo == 'l') ) {
        zsymmetrize_lower<<< grid, threads, 0, magma_stream >>>( m, dA, ldda );
    }
    else {
        printf( "uplo has illegal value\n" );
        exit(1);
    }
}
