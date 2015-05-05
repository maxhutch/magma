/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from zlobpcg_maxpy.cu normal z -> d, Sun May  3 11:22:58 2015

*/

#include "common_magma.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  512



__global__ void
magma_dlobpcg_maxpy_kernel( 
    magma_int_t num_rows, 
    magma_int_t num_vecs, 
    double * X, 
    double * Y)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x; // global row index

    if( row<num_rows ){
        for( int i=0; i<num_vecs; i++ ){ 

            Y[ row + i*num_rows ] += X[ row + i*num_rows ];
        }
    }
}




/**
    Purpose
    -------
    
    This routine computes a axpy for a mxn matrix:
        
        Y = X + Y
        
    It replaces:
            magma_daxpy(m*n, c_one, Y, 1, X, 1);


        / x1[0] x2[0] x3[0] \
        | x1[1] x2[1] x3[1] |
    X = | x1[2] x2[2] x3[2] | = x1[0] x1[1] x1[2] x1[3] x1[4] x2[0] x2[1] .
        | x1[3] x2[3] x3[3] |
        \ x1[4] x2[4] x3[4] /
    
    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows

    @param[in]
    num_vecs    magma_int_t
                number of vectors

    @param[in]
    X           magmaDouble_ptr 
                input vector X

    @param[in/out]
    Y           magmaDouble_ptr 
                input/output vector Y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_dlobpcg_maxpy(
    magma_int_t num_rows,
    magma_int_t num_vecs, 
    magmaDouble_ptr X,
    magmaDouble_ptr Y,
    magma_queue_t queue )
{
    // every thread handles one row

    magma_int_t block_size = BLOCK_SIZE;
     magma_int_t threads = BLOCK_SIZE;
    dim3 block( block_size );
    dim3 grid( magma_ceildiv( num_rows, block_size ) );

    magma_dlobpcg_maxpy_kernel<<< grid, threads, 0, queue >>>
                                ( num_rows, num_vecs, X, Y );


    return MAGMA_SUCCESS;
}



