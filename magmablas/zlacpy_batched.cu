/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
       @author Mark Gates
*/
#include "common_magma.h"

#define NB 64

/* =====================================================================
    Batches zlacpy of multiple arrays;
    y-dimension of grid is different arrays,
    x-dimension of grid is blocks for each array.
    Matrix is m x n, and is divided into block rows, each NB x n.
    Each CUDA block has NB threads to handle one block row.
    Each thread copies one row, iterating across all columns.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
__global__ void
zlacpy_batched_kernel(
    int m, int n,
    const magmaDoubleComplex * const *dAarray, int ldda,
    magmaDoubleComplex              **dBarray, int lddb )
{
    // dA and dB iterate across row i
    const magmaDoubleComplex *dA = dAarray[ blockIdx.y ];
    magmaDoubleComplex       *dB = dBarray[ blockIdx.y ];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < m ) {
        dA += i;
        dB += i;
        const magmaDoubleComplex *dAend = dA + n*ldda;
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
    ZLACPY copies all or part of a set of two-dimensional matrices dAarray[i]
    to another set of matrices dBarray[i], for i = 0, ..., batchCount-1.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of each matrix dAarray[i] to be copied to dBarray[i].
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:  All of each matrix dAarray[i]
    
    @param[in]
    m       INTEGER
            The number of rows of each matrix dAarray[i].  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of each matrix dAarray[i].  N >= 0.
    
    @param[in]
    dAarray array on GPU, dimension(batchCount), of pointers to arrays,
            with each array a COMPLEX_16 array, dimension (LDDA,N)
            The m by n matrices dAarray[i].
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of each array dAarray[i].  LDDA >= max(1,M).
    
    @param[out]
    dBarray array on GPU, dimension(batchCount), of pointers to arrays,
            with each array a COMPLEX_16 array, dimension (LDDB,N)
            The m by n matrices dBarray[i].
            On exit, matrix dBarray[i] = matrix dAarray[i] in the locations
            specified by UPLO.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of each array dBarray[i].  LDDB >= max(1,M).
    
    @param[in]
    batchCount INTEGER
            The number of matrices to add; length of dAarray and dBarray.
            batchCount >= 0.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlacpy_batched_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex * const *dAarray, magma_int_t ldda,
    magmaDoubleComplex              **dBarray, magma_int_t lddb,
    magma_int_t batchCount,
    magma_queue_t queue )
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
    else if ( batchCount < 0 )
        info = -8;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || n == 0 || batchCount == 0 )
        return;
    
    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB, batchCount );
    
    if ( uplo == MagmaUpper ) {
        fprintf(stderr, "lacpy upper is not implemented\n");
    }
    else if ( uplo == MagmaLower ) {
        fprintf(stderr, "lacpy lower is not implemented\n");
    }
    else {
        zlacpy_batched_kernel<<< grid, threads, 0, queue >>>(
            m, n, dAarray, ldda, dBarray, lddb );
    }
}


/**
    @see magmablas_zlacpy_batched_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    const magmaDoubleComplex * const *dAarray, magma_int_t ldda,
    magmaDoubleComplex              **dBarray, magma_int_t lddb,
    magma_int_t batchCount )
{
    magmablas_zlacpy_batched_q(
        uplo, m, n, dAarray, ldda, dBarray, lddb, batchCount, magma_stream );
}
