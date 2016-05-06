/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


// axpy kernel for matrices stored in the MAGMA format
__global__ void 
zgedensereimsplit_kernel( 
    int num_rows, 
    int num_cols,  
    magma_index_t* rowidx,
    magmaDoubleComplex * A, 
    magmaDoubleComplex * ReA, 
    magmaDoubleComplex * ImA )
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if( row<num_rows ){
        for( j=0; j<num_cols; j++ ){
            ReA[ j ] = MAGMA_Z_MAKE( MAGMA_Z_REAL( A[ j ] ), 0.0 );
            ImA[ j ] = MAGMA_Z_MAKE( MAGMA_Z_IMAG( A[ j ] ), 0.0 );
        }
    }
}

/**
    Purpose
    -------
    
    This routine takes an input matrix A in DENSE format and located on the GPU
    and splits it into two matrixes ReA and ImA containing the real and the 
    imaginary contributions of A.
    The output matrices are allocated within the routine.
    
    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A.
                
    @param[out]
    ReA         magma_z_matrix*
                output matrix contaning real contributions.
                
    @param[out]
    ImA         magma_z_matrix*
                output matrix contaning complex contributions.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" 
magma_int_t
magma_zgedensereimsplit(
    magma_z_matrix A,
    magma_z_matrix *ReA,
    magma_z_matrix *ImA,
    magma_queue_t queue )
{
    
    magma_zmtransfer( A, ReA, Magma_DEV, Magma_DEV, queue );
    magma_zmtransfer( A, ImA, Magma_DEV, Magma_DEV, queue );
    
    
    int m = A.num_rows;
    int n = A.num_cols;
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    zgedensereimsplit_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                    ( m, n, A.row, A.dval, ReA->dval, ImA->dval );
                    
    return MAGMA_SUCCESS;
}
