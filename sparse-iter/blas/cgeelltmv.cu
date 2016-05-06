/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zgeelltmv.cu normal z -> c, Mon May  2 23:30:44 2016

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512


// ELL SpMV kernel
//Michael Garland
template<bool betazero>
__global__ void 
cgeelltmv_kernel( 
    int num_rows, 
    int num_cols,
    int num_cols_per_row,
    magmaFloatComplex alpha, 
    magmaFloatComplex * dval, 
    magma_index_t * dcolind,
    magmaFloatComplex * dx,
    magmaFloatComplex beta, 
    magmaFloatComplex * dy)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows ) {
        magmaFloatComplex dot = MAGMA_C_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_rows * n + row ];
            magmaFloatComplex val = dval [ num_rows * n + row ];
            //if ( val != MAGMA_C_ZERO )
                dot += val * dx[col ];
        }
        if (betazero) {
            dy[ row ] = dot * alpha;
        } else {
            dy[ row ] = dot * alpha + beta * dy [ row ];
        }
    }
}

// shifted ELL SpMV kernel
//Michael Garland
__global__ void 
cgeelltmv_kernel_shift( 
    int num_rows, 
    int num_cols,
    int num_cols_per_row,
    magmaFloatComplex alpha, 
    magmaFloatComplex lambda, 
    magmaFloatComplex * dval, 
    magma_index_t * dcolind,
    magmaFloatComplex * dx,
    magmaFloatComplex beta, 
    int offset,
    int blocksize,
    magma_index_t * addrows,
    magmaFloatComplex * dy)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows ) {
        magmaFloatComplex dot = MAGMA_C_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_rows * n + row ];
            magmaFloatComplex val = dval [ num_rows * n + row ];
            if ( val != 0)
                dot += val * dx[col ];
        }
        if ( row < blocksize )
            dy[ row ] = dot * alpha - lambda 
                    * dx[ offset+row ] + beta * dy [ row ];
        else
            dy[ row ] = dot * alpha - lambda 
                    * dx[ addrows[row-blocksize] ] + beta * dy [ row ];            
    }
}




/**
    Purpose
    -------
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is ELL.
    
    Arguments
    ---------
    
    @param[in]
    transA      magma_trans_t
                transposition parameter for A
                
    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 
                
    @param[in]
    nnz_per_row magma_int_t
                number of elements in the longest row 

    @param[in]
    alpha       magmaFloatComplex
                scalar multiplier

    @param[in]
    dval        magmaFloatComplex_ptr
                array containing values of A in ELL

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELL

    @param[in]
    dx          magmaFloatComplex_ptr
                input vector x

    @param[in]
    beta        magmaFloatComplex
                scalar multiplier

    @param[out]
    dy          magmaFloatComplex_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_c
    ********************************************************************/

extern "C" magma_int_t
magma_cgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    if (beta == MAGMA_C_ZERO) {
        cgeelltmv_kernel<true><<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );
    } else {
        cgeelltmv_kernel<false><<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );
    }


   return MAGMA_SUCCESS;
}


/**
    Purpose
    -------
    
    This routine computes y = alpha *( A - lambda I ) * x + beta * y on the GPU.
    Input format is ELL.
    
    Arguments
    ---------

    @param[in]
    transA      magma_trans_t
                transposition parameter for A    

    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 
                
    @param[in]
    nnz_per_row magma_int_t
                number of elements in the longest row 

    @param[in]
    alpha       magmaFloatComplex
                scalar multiplier

    @param[in]
    lambda      magmaFloatComplex
                scalar multiplier

    @param[in]
    dval        magmaFloatComplex_ptr
                array containing values of A in ELL

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELL

    @param[in]
    dx          magmaFloatComplex_ptr
                input vector x

    @param[in]
    beta        magmaFloatComplex
                scalar multiplier
                
    @param[in]
    offset      magma_int_t 
                in case not the main diagonal is scaled
                
    @param[in]
    blocksize   magma_int_t 
                in case of processing multiple vectors  
                
    @param[in]
    addrows     magmaIndex_ptr
                in case the matrixpowerskernel is used

    @param[out]
    dy          magmaFloatComplex_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_cgeelltmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaFloatComplex alpha,
    magmaFloatComplex lambda,
    magmaFloatComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magmaIndex_ptr addrows,
    magmaFloatComplex_ptr dy,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    magmaFloatComplex tmp_shift;
    //magma_csetvector(1,&lambda,1,&tmp_shift,1); 
    tmp_shift = lambda;
    cgeelltmv_kernel_shift<<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, tmp_shift, dval, dcolind, dx, 
                            beta, offset, blocksize, addrows, dy );


   return MAGMA_SUCCESS;
}
