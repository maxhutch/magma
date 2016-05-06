/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergeblockkrylov.cu normal z -> s, Mon May  2 23:30:50 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 16

#define PRECISION_s


// These routines merge multiple kernels from qmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_smergeblockkrylov_kernel(  
    int num_rows, 
    int num_cols, 
    float *alpha,
    float *p, 
    float *x )
{
    int num_vecs = num_cols;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int vec = blockIdx.y;
    
    if ( row<num_rows ) {
        
        float val = x[ row + vec * num_rows ];
        
        for( int j=0; j<num_vecs; j++ ){
            
            float lalpha = alpha[ j * num_vecs + vec ];
            float xval = p[ row + j * num_rows ];
            
            val += lalpha * xval;
            
        }
        x[ row + vec * num_rows ] = val;
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    v = y / rho
    y = y / rho
    w = wt / psi
    z = z / psi
    
    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       magmaFloat_ptr 
                matrix containing all SKP
                
    @param[in]
    p           magmaFloat_ptr 
                search directions
                
    @param[in,out]
    x           magmaFloat_ptr 
                approximation vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_smergeblockkrylov(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloat_ptr alpha, 
    magmaFloat_ptr p,
    magmaFloat_ptr x,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE, num_cols );
    
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_smergeblockkrylov_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>
                ( num_rows, num_cols, alpha, p, x );

   return MAGMA_SUCCESS;
}

