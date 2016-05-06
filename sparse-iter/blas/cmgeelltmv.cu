/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmgeelltmv.cu normal z -> c, Mon May  2 23:30:48 2016

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

template<bool betazero>
__global__ void 
cmgeelltmv_kernel( 
        int num_rows, 
        int num_cols,
        int num_vecs,
        int num_cols_per_row,
        magmaFloatComplex alpha, 
        magmaFloatComplex * dval, 
        magma_index_t * dcolind,
        magmaFloatComplex * dx,
        magmaFloatComplex beta, 
        magmaFloatComplex * dy)
{
    extern __shared__ magmaFloatComplex dot[];
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row < num_rows ) {
        for( int i=0; i<num_vecs; i++ )
            dot[ threadIdx.x+ i*blockDim.x ] = MAGMA_C_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_rows * n + row ];
            magmaFloatComplex val = dval [ num_rows * n + row ];
                for( int i=0; i<num_vecs; i++ )
                    dot[ threadIdx.x + i*blockDim.x ] += 
                                        val * dx[col + i * num_cols ];
        }
        for( int i=0; i<num_vecs; i++ ) {
            if (betazero) {
                dy[ row + i*num_cols ] = dot[ threadIdx.x + i*blockDim.x ] *alpha;
            } else {
                dy[ row + i*num_cols ] = dot[ threadIdx.x + i*blockDim.x ] 
                                        * alpha + beta * dy [ row + i*num_cols ];
            }
        }
    }
}





/**
    Purpose
    -------
    
    This routine computes Y = alpha *  A *  X + beta * Y for X and Y sets of 
    num_vec vectors on the GPU. Input format is ELL. 
    
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
    num_vecs    mama_int_t
                number of vectors
                
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

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_cmgeelltmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
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
    unsigned int MEM_SIZE =  num_vecs* BLOCK_SIZE 
                * sizeof( magmaFloatComplex ); // num_vecs vectors 
    if (beta == MAGMA_C_ZERO) {
        cmgeelltmv_kernel<true><<< grid, threads, MEM_SIZE, queue->cuda_stream() >>>
            ( m, n, num_vecs, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );
    } else {
        cmgeelltmv_kernel<false><<< grid, threads, MEM_SIZE, queue->cuda_stream() >>>
            ( m, n, num_vecs, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );
    }


    return MAGMA_SUCCESS;
}
