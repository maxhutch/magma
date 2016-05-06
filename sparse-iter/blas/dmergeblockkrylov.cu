/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergeblockkrylov.cu normal z -> d, Mon May  2 23:30:50 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 16

#define PRECISION_d


// These routines merge multiple kernels from qmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_dmergeblockkrylov_kernel(  
    int num_rows, 
    int num_cols, 
    double *alpha,
    double *p, 
    double *x )
{
    int num_vecs = num_cols;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int vec = blockIdx.y;
    
    if ( row<num_rows ) {
        
        double val = x[ row + vec * num_rows ];
        
        for( int j=0; j<num_vecs; j++ ){
            
            double lalpha = alpha[ j * num_vecs + vec ];
            double xval = p[ row + j * num_rows ];
            
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
    alpha       magmaDouble_ptr 
                matrix containing all SKP
                
    @param[in]
    p           magmaDouble_ptr 
                search directions
                
    @param[in,out]
    x           magmaDouble_ptr 
                approximation vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dmergeblockkrylov(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDouble_ptr alpha, 
    magmaDouble_ptr p,
    magmaDouble_ptr x,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE, num_cols );
    
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dmergeblockkrylov_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>
                ( num_rows, num_cols, alpha, p, x );

   return MAGMA_SUCCESS;
}

