/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zlacpy.cu normal z -> s, Fri May 30 10:40:40 2014
       @author Mark Gates
*/
#include "common_magma.h"
#include <assert.h>

#define NB 64

/* =====================================================================
    Matrix is m x n, and is divided into block rows, each NB x n.
    Each CUDA block has NB threads to handle one block row.
    Each thread copies one row, iterating across all columns.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
__global__ void
slacpy_kernel(
    int m, int n,
    const float *dA, int ldda,
    float       *dB, int lddb )
{
    // dA and dB iterate across row i
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < m ) {
        dA += i;
        dB += i;
        const float *dAend = dA + n*ldda;
        while( dA < dAend ) {
            *dB = *dA;
            dA += ldda;
            dB += lddb;
        }
    }
}


/* ===================================================================== */
/**
    Note
    --------
    - UPLO Parameter is disabled
    - Do we want to provide a generic function to the user with all the options?
    
    Purpose
    -------
    SLACPY copies all or part of a two-dimensional matrix dA to another
    matrix dB.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be copied to dB.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:  All of the matrix dA
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA      COMPLEX REAL array, dimension (LDDA,N)
            The m by n matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[out]
    dB      COMPLEX REAL array, dimension (LDDB,N)
            The m by n matrix dB.
            On exit, dB = dA in the locations specified by UPLO.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    

    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const float *dA, magma_int_t ldda,
    float       *dB, magma_int_t lddb )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || n == 0 )
        return;
    
    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB );
    
    if ( uplo == MagmaUpper ) {
        fprintf(stderr, "lacpy upper is not implemented\n");
    }
    else if ( uplo == MagmaLower ) {
        fprintf(stderr, "lacpy lower is not implemented\n");
    }
    else {
        slacpy_kernel<<< grid, threads, 0, magma_stream >>>(
            m, n, dA, ldda, dB, lddb );
    }
}
