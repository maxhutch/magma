/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergeidr.cu normal z -> c, Mon May  2 23:30:46 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_c


// These routines merge multiple kernels from cidr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_cidr_smoothing_1_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex *drs,
    magmaFloatComplex *dr,
    magmaFloatComplex *dt )
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
    drs         magmaFloatComplex_ptr 
                vector

    @param[in]
    dr          magmaFloatComplex_ptr 
                vector

    @param[in,out]
    dt          magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cidr_smoothing_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex_ptr drs,
    magmaFloatComplex_ptr dr, 
    magmaFloatComplex_ptr dt, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cidr_smoothing_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, drs, dr, dt );

   return MAGMA_SUCCESS;
}



__global__ void
magma_cidr_smoothing_2_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex omega,
    magmaFloatComplex *dx,
    magmaFloatComplex *dxs )
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
    omega       magmaFloatComplex
                scalar
                
    @param[in]
    dx          magmaFloatComplex_ptr 
                vector

    @param[in,out]
    dxs         magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cidr_smoothing_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex omega,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dxs, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cidr_smoothing_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, omega, dx, dxs);

   return MAGMA_SUCCESS;
}
