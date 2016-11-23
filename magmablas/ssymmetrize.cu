/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from magmablas/zsymmetrize.cu, normal z -> s, Sun Nov 20 20:20:30 2016
       @author Mark Gates
*/
#include "magma_internal.h"

#define NB 64

/*
    Matrix is m x m, and is divided into block rows, each NB x m.
    Each block has NB threads.
    Each thread copies one row, iterating across all columns below diagonal.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
__global__ void
ssymmetrize_lower( int m, float *dA, int ldda )
{
    // dA iterates across row i and dAT iterates down column i.
    int i = blockIdx.x*NB + threadIdx.x;
    float *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        float *dAend = dA + i*ldda;  // end at diagonal dA(i,i)
        while( dA < dAend ) {
            *dAT = MAGMA_S_CONJ(*dA);  // upper := lower
            dA  += ldda;
            dAT += 1;
        }
        *dA = MAGMA_S_MAKE( MAGMA_S_REAL(*dA), 0 );  // make diagonal real
    }
}


// only difference with _lower version is direction dA=dAT instead of dAT=dA.
__global__ void
ssymmetrize_upper( int m, float *dA, int ldda )
{
    // dA iterates across row i and dAT iterates down column i.
    int i = blockIdx.x*NB + threadIdx.x;
    float *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        float *dAend = dA + i*ldda;  // end at diagonal dA(i,i)
        while( dA < dAend ) {
            *dA = MAGMA_S_CONJ(*dAT);  // lower := upper
            dA  += ldda;
            dAT += 1;
        }
        *dA = MAGMA_S_MAKE( MAGMA_S_REAL(*dA), 0 );  // make diagonal real
    }
}


/***************************************************************************//**
    Purpose
    -------
    
    SSYMMETRIZE copies lower triangle to upper triangle, or vice-versa,
    to make dA a general representation of a symmetric matrix.
    In Complex, it sets the diagonal to be Real.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA that is valid on input.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in,out]
    dA      REAL array, dimension (LDDA,N)
            The m by m matrix dA.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_symmetrize
*******************************************************************************/
extern "C" void
magmablas_ssymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( ldda < max(1,m) )
        info = -4;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 )
        return;
    
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    
    if ( uplo == MagmaUpper ) {
        ssymmetrize_upper<<< grid, threads, 0, queue->cuda_stream() >>>( m, dA, ldda );
    }
    else {
        ssymmetrize_lower<<< grid, threads, 0, queue->cuda_stream() >>>( m, dA, ldda );
    }
}
