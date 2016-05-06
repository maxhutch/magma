/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 16

#define PRECISION_z


// These routines merge multiple kernels from qmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_zmergeblockkrylov_kernel(  
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex *alpha,
    magmaDoubleComplex *p, 
    magmaDoubleComplex *x )
{
    int num_vecs = num_cols;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int vec = blockIdx.y;
    
    if ( row<num_rows ) {
        
        magmaDoubleComplex val = x[ row + vec * num_rows ];
        
        for( int j=0; j<num_vecs; j++ ){
            
            magmaDoubleComplex lalpha = alpha[ j * num_vecs + vec ];
            magmaDoubleComplex xval = p[ row + j * num_rows ];
            
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
    alpha       magmaDoubleComplex_ptr 
                matrix containing all SKP
                
    @param[in]
    p           magmaDoubleComplex_ptr 
                search directions
                
    @param[in,out]
    x           magmaDoubleComplex_ptr 
                approximation vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zmergeblockkrylov(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex_ptr alpha, 
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr x,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE, num_cols );
    
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zmergeblockkrylov_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>
                ( num_rows, num_cols, alpha, p, x );

   return MAGMA_SUCCESS;
}

