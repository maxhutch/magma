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

#define BLOCK_SIZE 512

#define PRECISION_z


// These routines merge multiple kernels from zidr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_zidr_smoothing_1_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex *drs,
    magmaDoubleComplex *dr,
    magmaDoubleComplex *dt )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            dt[ i+j*num_rows ] =  drs[ i+j*num_rows ] - dr[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    dt = drs - dr

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n

    @param[in]
    drs         magmaDoubleComplex_ptr 
                vector

    @param[in]
    dr          magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    dt          magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zidr_smoothing_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex_ptr drs,
    magmaDoubleComplex_ptr dr, 
    magmaDoubleComplex_ptr dt, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zidr_smoothing_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, drs, dr, dt );

   return MAGMA_SUCCESS;
}



__global__ void
magma_zidr_smoothing_2_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex omega,
    magmaDoubleComplex *dx,
    magmaDoubleComplex *dxs )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            dxs[ i+j*num_rows ] = dxs[ i+j*num_rows ] + omega * dxs[ i+j*num_rows ]
                    - omega * dx[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    dxs = dxs - gamma*(dxs-dx)

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    omega       magmaDoubleComplex
                scalar
                
    @param[in]
    dx          magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    dxs         magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zidr_smoothing_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dxs, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zidr_smoothing_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, omega, dx, dxs);

   return MAGMA_SUCCESS;
}
