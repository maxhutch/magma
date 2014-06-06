/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:45 2013
       @author Mark Gates
*/
#include "common_magma.h"
#include <assert.h>

#define NB 64

/* =====================================================================
    Batches dlacpy of multiple arrays;
    y-dimension of grid is different arrays,
    x-dimension of grid is blocks for each array.
    Matrix is m x n, and is divided into block rows, each NB x n.
    Each CUDA block has NB threads to handle one block row.
    Each thread adds one row, iterating across all columns.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
    
    TODO. Block in both directions, for large matrices.
    E.g., each block does 64x64 tile, instead of 64xN tile.
*/
__global__ void
dgeadd_batched_kernel(
    int m, int n,
    double alpha,
    const double * const *dAarray, int ldda,
    double              **dBarray, int lddb )
{
    // dA and dB iterate across row i
    const double *dA = dAarray[ blockIdx.y ];
    double       *dB = dBarray[ blockIdx.y ];
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if ( i < m ) {
        dA += i;
        dB += i;
        const double *dAend = dA + n*ldda;
        while( dA < dAend ) {
            *dB = alpha*(*dA) + (*dB);
            dA += ldda;
            dB += lddb;
        }
    }
}


/* ===================================================================== */
extern "C" void
magmablas_dgeadd_batched(
    magma_int_t m, magma_int_t n,
    double alpha,
    const double * const *dAarray, magma_int_t ldda,
    double              **dBarray, magma_int_t lddb,
    magma_int_t batchCount )
{
/*
    Purpose
    =======
    ZGEADD adds two sets of matrices, dAarray[i] = alpha*dAarray[i] + dBarray[i],
    for i = 0, ..., batchCount-1.
    
    Arguments
    =========
    
    M       (input) INTEGER
            The number of rows of each matrix dAarray[i].  M >= 0.
    
    N       (input) INTEGER
            The number of columns of each matrix dAarray[i].  N >= 0.
    
    ALPHA   (input) COMPLEX DOUBLE PRECISION
            The scalar alpha.
            
    dAarray (input) array on GPU, dimension(batchCount), of pointers to arrays,
            with each array a COMPLEX DOUBLE PRECISION array, dimension (LDDA,N)
            The m by n matrices dAarray[i].
    
    LDDA    (input) INTEGER
            The leading dimension of each array dAarray[i].  LDDA >= max(1,M).
            
    dBarray (input/output) array on GPU, dimension(batchCount), of pointers to arrays,
            with each array a COMPLEX DOUBLE PRECISION array, dimension (LDDB,N)
            The m by n matrices dBarray[i].
    
    LDDB    (input) INTEGER
            The leading dimension of each array dBarray[i].  LDDB >= max(1,M).
    
    batchCount (input) INTEGER
            The number of matrices to add; length of dAarray and dBarray.
            batchCount >= 0.
    
    =====================================================================   */

    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
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
        
    dgeadd_batched_kernel<<< grid, threads, 0, magma_stream >>>(
        m, n, alpha, dAarray, ldda, dBarray, lddb );
}
