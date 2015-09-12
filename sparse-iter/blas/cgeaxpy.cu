/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from zgeaxpy.cu normal z -> c, Fri Sep 11 18:29:42 2015

*/
#include "common_magma.h"
#include "common_magmasparse.h"

#define BLOCK_SIZE 256


// axpy kernel for matrices stored in the MAGMA format
__global__ void 
cgeaxpy_kernel( 
    int num_rows, 
    int num_cols, 
    magmaFloatComplex alpha, 
    magmaFloatComplex * dx, 
    magmaFloatComplex beta, 
    magmaFloatComplex * dy)
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
    magma_c_matrix format.
    
    Arguments
    ---------

    @param[in]
    alpha       magmaFloatComplex
                scalar multiplier.
                
    @param[in]
    X           magma_c_matrix
                input/output matrix Y.
                
    @param[in]
    beta        magmaFloatComplex
                scalar multiplier.
                
    @param[in,out]
    Y           magma_c_matrix*
                input matrix X.
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" 
magma_int_t
magma_cgeaxpy(
    magmaFloatComplex alpha,
    magma_c_matrix X,
    magmaFloatComplex beta,
    magma_c_matrix *Y,
    magma_queue_t queue )
{
    int m = X.num_rows;
    int n = X.num_cols;
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    cgeaxpy_kernel<<< grid, threads, 0, queue >>>
                    ( m, n, alpha, X.dval, beta, Y->dval );
                    
    return MAGMA_SUCCESS;
}
