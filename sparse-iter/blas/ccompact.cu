/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zcompact.cu normal z -> c, Fri May 30 10:41:37 2014
       @author Stan Tomov
*/
#include "common_magma.h"
#include <assert.h>

#define NB 64

/* =====================================================================
    Matrix is m x n, and is divided into block rows, each NB x n.
    Each CUDA block has NB threads to handle one block row.
    Each thread handles one row, iterating across all columns.
*/
__global__ void
ccompact_kernel(
    int m, int n,
    magmaFloatComplex *dA, int ldda,
    float *dnorms, float tol,
    magma_index_t *active, magma_index_t *cBlock)
{
    // dA is processed across row i (by the current thread)
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int cBlockSize = 0;
    if ( i < m ) {
        dA += i;
        
        for(int j = 0; j<n; j++){
            if (dnorms[j] > tol && active[j]){
               dA[ldda*cBlockSize] = dA[ldda*j];
               cBlockSize++;
            }
            else if (i==0)
               active[j] = 0;
        }
    }

    if (i==0)
       *cBlock = cBlockSize;
}

__global__ void
ccompactactive_kernel(
    int m, int n,
    magmaFloatComplex *dA, int ldda,
    magma_index_t *active)
{
    // dA is processed across row i (by the current thread)
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int cBlockSize = 0;
    if ( i < m ) {
        dA += i;

        for(int j = 0; j<n; j++){
            if (active[j]){
               dA[ldda*cBlockSize] = dA[ldda*j];
               cBlockSize++;
            }
        }
    }
}


/* ===================================================================== */

extern "C" void
magma_ccompact(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *dA, magma_int_t ldda,
    float *dnorms, float tol, 
    magma_index_t *active, magma_index_t *cBlock)
{
/*
    Purpose
    =======
    ZCOMPACT takes a set of n vectors of size m (in dA) and their norms and
    compacts them into the cBlock size<=n vectors that have norms > tol.
    The active mask array has 1 or 0, showing if a vector remained or not
    in the compacted resulting set of vectors.
    
    Arguments
    =========    
    M       (input) INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    N       (input) INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    dA      (input/output) COMPLEX REAL array, dimension (LDDA,N)
            The m by n matrix dA.
    
    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    DNORMS  (input) REAL array, dimension N
            The norms of the N vectors in dA

    TOL     (input) DOUBLE PRECISON
            The tolerance value used in the criteria to compact or not.

    ACTIVE  (output) INTEGER array, dimension N
            A mask of 1s and 0s showing if a vector remains or has been removed

    CBLOCK  (output)
            The number of vectors that remain in dA (i.e., with norms > tol).
    =====================================================================   */

    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -4;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || n == 0 )
        return;
    
    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB );
    
    ccompact_kernel<<< grid, threads, 0, magma_stream >>>(
            m, n, dA, ldda, dnorms, tol, active, active+n );

    cublasGetMatrix( 1, 1, sizeof( magma_int_t ), active+n, 1, cBlock, 1 ) ;
}

/* ===================================================================== */

extern "C" void
magma_ccompactActive(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *dA, magma_int_t ldda,
    magma_index_t *active)
{
/*
    Purpose
    =======
    ZCOMPACTACTIVE takes a set of n vectors of size m (in dA) and an
    array of 1s and 0sindicating which vectors to compact (for 1s) and
    which to disregard (for 0s).

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix dA.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix dA.  N >= 0.

    dA      (input/output) COMPLEX REAL array, dimension (LDDA,N)
            The m by n matrix dA.

    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).

    ACTIVE  (input) INTEGER array, dimension N
            A mask of 1s and 0s showing if a vector remains or has been removed
    =====================================================================     */

    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -4;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    if ( m == 0 || n == 0 )
        return;

    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB );

    ccompactactive_kernel<<< grid, threads, 0, magma_stream >>>(
            m, n, dA, ldda, active);
}

/* ===================================================================== */
