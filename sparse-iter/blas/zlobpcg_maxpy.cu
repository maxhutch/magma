/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s

*/

#include "common_magma.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  512



__global__ void 
magma_zlobpcg_maxpy_kernel( magma_int_t num_rows, 
                            magma_int_t num_vecs, 
                            magmaDoubleComplex *X, 
                            magmaDoubleComplex *Y){

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
            magma_zaxpy(m*n, c_one, Y, 1, X, 1);


        / x1[0] x2[0] x3[0] \
        | x1[1] x2[1] x3[1] |
    X = | x1[2] x2[2] x3[2] | = x1[0] x1[1] x1[2] x1[3] x1[4] x2[0] x2[1] .
        | x1[3] x2[3] x3[3] |
        \ x1[4] x2[4] x3[4] /
    
    Arguments
    ---------

    @param
    num_rows    magma_int_t
                number of rows

    @param
    num_vecs    magma_int_t
                number of vectors

    @param
    X           magmaDoubleComplex*
                input vector X

    @param
    Y           magmaDoubleComplex*
                input/output vector Y


    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zlobpcg_maxpy(    magma_int_t num_rows,
                        magma_int_t num_vecs, 
                        magmaDoubleComplex *X,
                        magmaDoubleComplex *Y){

    // every thread handles one row

    magma_int_t block_size = BLOCK_SIZE;
 
    dim3 block( block_size );
    dim3 grid( (num_rows+block_size-1)/block_size );

    magma_zlobpcg_maxpy_kernel<<< grid, block, 0, magma_stream >>>
                                ( num_rows, num_vecs, X, Y );


    return MAGMA_SUCCESS;
}



