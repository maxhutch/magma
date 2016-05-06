/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zgecsrreimsplit.cu normal z -> d, Mon May  2 23:30:49 2016

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


// axpy kernel for matrices stored in the MAGMA format
__global__ void 
dgecsrreimsplit_kernel( 
    int num_rows, 
    int num_cols,  
    magma_index_t* rowidx,
    double * A, 
    double * ReA, 
    double * ImA )
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if( row<num_rows ){
        for( j=rowidx[row]; j<rowidx[row+1]; j++ ){
            ReA[ j ] = MAGMA_D_MAKE( MAGMA_D_REAL( A[ j ] ), 0.0 );
            ImA[ j ] = MAGMA_D_MAKE( MAGMA_D_IMAG( A[ j ] ), 0.0 );
        }
    }
}

/**
    Purpose
    -------
    
    This routine takes an input matrix A in CSR format and located on the GPU
    and splits it into two matrixes ReA and ImA containing the real and the 
    imaginary contributions of A.
    The output matrices are allocated within the routine.
    
    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A.
                
    @param[out]
    ReA         magma_d_matrix*
                output matrix contaning real contributions.
                
    @param[out]
    ImA         magma_d_matrix*
                output matrix contaning real contributions.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dblas
    ********************************************************************/

extern "C" 
magma_int_t
magma_dgecsrreimsplit(
    magma_d_matrix A,
    magma_d_matrix *ReA,
    magma_d_matrix *ImA,
    magma_queue_t queue )
{
    
    magma_dmtransfer( A, ReA, Magma_DEV, Magma_DEV, queue );
    magma_dmtransfer( A, ImA, Magma_DEV, Magma_DEV, queue );
    
    
    int m = A.num_rows;
    int n = A.num_cols;
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    dgecsrreimsplit_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                    ( m, n, A.row, A.dval, ReA->dval, ImA->dval );
                    
    return MAGMA_SUCCESS;
}
