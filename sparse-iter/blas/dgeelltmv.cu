/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zgeelltmv.cu normal z -> d, Mon May  2 23:30:44 2016

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512


// ELL SpMV kernel
//Michael Garland
template<bool betazero>
__global__ void 
dgeelltmv_kernel( 
    int num_rows, 
    int num_cols,
    int num_cols_per_row,
    double alpha, 
    double * dval, 
    magma_index_t * dcolind,
    double * dx,
    double beta, 
    double * dy)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows ) {
        double dot = MAGMA_D_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_rows * n + row ];
            double val = dval [ num_rows * n + row ];
            //if ( val != MAGMA_D_ZERO )
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
dgeelltmv_kernel_shift( 
    int num_rows, 
    int num_cols,
    int num_cols_per_row,
    double alpha, 
    double lambda, 
    double * dval, 
    magma_index_t * dcolind,
    double * dx,
    double beta, 
    int offset,
    int blocksize,
    magma_index_t * addrows,
    double * dy)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows ) {
        double dot = MAGMA_D_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_rows * n + row ];
            double val = dval [ num_rows * n + row ];
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
    alpha       double
                scalar multiplier

    @param[in]
    dval        magmaDouble_ptr
                array containing values of A in ELL

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELL

    @param[in]
    dx          magmaDouble_ptr
                input vector x

    @param[in]
    beta        double
                scalar multiplier

    @param[out]
    dy          magmaDouble_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_d
    ********************************************************************/

extern "C" magma_int_t
magma_dgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    if (beta == MAGMA_D_ZERO) {
        dgeelltmv_kernel<true><<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );
    } else {
        dgeelltmv_kernel<false><<< grid, threads, 0, queue->cuda_stream() >>>
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
    alpha       double
                scalar multiplier

    @param[in]
    lambda      double
                scalar multiplier

    @param[in]
    dval        magmaDouble_ptr
                array containing values of A in ELL

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELL

    @param[in]
    dx          magmaDouble_ptr
                input vector x

    @param[in]
    beta        double
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
    dy          magmaDouble_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dblas
    ********************************************************************/

extern "C" magma_int_t
magma_dgeelltmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    double alpha,
    double lambda,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magmaIndex_ptr addrows,
    magmaDouble_ptr dy,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    double tmp_shift;
    //magma_dsetvector(1,&lambda,1,&tmp_shift,1); 
    tmp_shift = lambda;
    dgeelltmv_kernel_shift<<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, tmp_shift, dval, dcolind, dx, 
                            beta, offset, blocksize, addrows, dy );


   return MAGMA_SUCCESS;
}
