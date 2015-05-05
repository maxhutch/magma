/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @precisions normal z -> s d c
       @author Stan Tomov
*/
#include "common_magmasparse.h"

#define NB 64

/* =====================================================================
    Matrix is m x n, and is divided into block rows, each NB x n.
    Each CUDA block has NB threads to handle one block row.
    Each thread handles one row, iterating across all columns.
*/
__global__ void
zcompact_kernel(
    int m, int n,
    magmaDoubleComplex *dA, 
    int ldda,
    double *dnorms, 
    double tol,
    magma_int_t *active, 
    magma_int_t *cBlock)
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
zcompactactive_kernel(
    int m, 
    int n,
    magmaDoubleComplex *dA, 
    int ldda,
    magma_int_t *active)
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
/**
    Purpose
    -------
    ZCOMPACT takes a set of n vectors of size m (in dA) and their norms and
    compacts them into the cBlock size<=n vectors that have norms > tol.
    The active mask array has 1 or 0, showing if a vector remained or not
    in the compacted resulting set of vectors.
    
    Arguments
    ---------
    @param[in]
    m           INTEGER
                The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n           INTEGER
                The number of columns of the matrix dA.  N >= 0.
    
    @param[in][in,out]
    dA          COMPLEX DOUBLE PRECISION array, dimension (LDDA,N)
                The m by n matrix dA.
    
    @param[in]
    ldda        INTEGER
                The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    dnorms      DOUBLE PRECISION array, dimension N
                The norms of the N vectors in dA

    @param[in]
    tol         DOUBLE PRECISON
                The tolerance value used in the criteria to compact or not.

    @param[in][out]
    active      INTEGER array, dimension N
                A mask of 1s and 0s showing if a vector remains or has been removed
            
    @param[in][out]
    cBlock      magmaInt_ptr
                The number of vectors that remain in dA (i.e., with norms > tol).
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zcompact(
    magma_int_t m, 
    magma_int_t n,
    magmaDoubleComplex_ptr dA, 
    magma_int_t ldda,
    magmaDouble_ptr dnorms, 
    double tol, 
    magmaInt_ptr active,
    magmaInt_ptr cBlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -4;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    if ( m == 0 || n == 0 )
        return info;
    
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    
    zcompact_kernel<<< grid, threads, 0, queue >>>(
            m, n, dA, ldda, dnorms, tol, active, active+n );

    magma_igetvector( 1, active+n, 1, cBlock, 1 );
    return info;
}


/* ===================================================================== */
/**
    Purpose
    -------
    ZCOMPACTACTIVE takes a set of n vectors of size m (in dA) and an
    array of 1s and 0sindicating which vectors to compact (for 1s) and
    which to disregard (for 0s).

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.

    @param[in][in,out]
    dA      COMPLEX DOUBLE PRECISION array, dimension (LDDA,N)
            The m by n matrix dA.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).

    @param[in]
    active  INTEGER array, dimension N
            A mask of 1s and 0s showing if a vector remains or has been removed
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zcompactActive(
    magma_int_t m, 
    magma_int_t n,
    magmaDoubleComplex_ptr dA, 
    magma_int_t ldda,
    magmaInt_ptr active,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < max(1,m))
        info = -4;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    if ( m == 0 || n == 0 )
        return info;

    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );

    zcompactactive_kernel<<< grid, threads, 0, queue >>>(
            m, n, dA, ldda, active);
    return info;
}

/* ===================================================================== */
