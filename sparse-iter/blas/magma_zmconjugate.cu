/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


__global__ void 
magma_zmconjugate_kernel(  
    int num_rows,
    magma_index_t *rowptr, 
    magmaDoubleComplex *values )
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < num_rows ){
        for( int i = rowptr[row]; i < rowptr[row+1]; i++){
            values[i] = MAGMA_Z_CONJ( values[i] );
        }
    }
}



/**
    Purpose
    -------

    This function conjugates a matrix. For a real matrix, no value is changed.

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                input/output matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmconjugate(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    dim3 grid( magma_ceildiv( A->num_rows, BLOCK_SIZE ));
    magma_zmconjugate_kernel<<< grid, BLOCK_SIZE, 0, queue->cuda_stream() >>> 
                                    ( A->num_rows, A->drow, A->dval );
        
    return info;
}
