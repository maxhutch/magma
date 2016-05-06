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
zgeaxpy_kernel( 
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dx, 
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy)
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if( row<num_rows ){
        for( j=0; j<num_cols; j++ ){
            int idx = row + j*num_rows;
            dy[ idx ] = alpha * dx[ idx ] + beta * dy[ idx ];
        }
    }
}

/**
    Purpose
    -------
    
    This routine computes Y = alpha *  X + beta * Y on the GPU.
    The input format is a dense matrix (vector block) stored in 
    magma_z_matrix format.
    
    Arguments
    ---------

    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier.
                
    @param[in]
    X           magma_z_matrix
                input/output matrix Y.
                
    @param[in]
    beta        magmaDoubleComplex
                scalar multiplier.
                
    @param[in,out]
    Y           magma_z_matrix*
                input matrix X.
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" 
magma_int_t
magma_zgeaxpy(
    magmaDoubleComplex alpha,
    magma_z_matrix X,
    magmaDoubleComplex beta,
    magma_z_matrix *Y,
    magma_queue_t queue )
{
    int m = X.num_rows;
    int n = X.num_cols;
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    zgeaxpy_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                    ( m, n, alpha, X.dval, beta, Y->dval );
                    
    return MAGMA_SUCCESS;
}
