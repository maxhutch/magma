/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergebicgstab.cu normal z -> s, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_s


// These routines merge multiple kernels from bicgstab into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_sbicgstab_1_kernel(  
    int num_rows, 
    int num_cols, 
    float beta,
    float omega,
    float *r, 
    float *v,
    float *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            p[ i+j*num_rows ] = r[ i+j*num_rows ] + 
                beta * ( p[ i+j*num_rows ] - omega * v[ i+j*num_rows ] );
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    p = r + beta * ( p - omega * v )
    
    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    beta        float
                scalar
                
    @param[in]
    omega       float
                scalar
                
    @param[in]
    r           magmaFloat_ptr 
                vector
                
    @param[in]
    v           magmaFloat_ptr 
                vector
                
    @param[in,out]
    p           magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_sbicgstab_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float beta,
    float omega,
    magmaFloat_ptr r, 
    magmaFloat_ptr v,
    magmaFloat_ptr p,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sbicgstab_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, omega,
                     r, v, p );

   return MAGMA_SUCCESS;
}





__global__ void
magma_sbicgstab_2_kernel(  
    int num_rows,
    int num_cols,
    float alpha,
    magmaFloat_ptr r,
    magmaFloat_ptr v,
    magmaFloat_ptr s )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            s[ i+j*num_rows ] = r[ i+j*num_rows ] - alpha * v[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    s = r - alpha v

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       float
                scalar
                
    @param[in]
    r           magmaFloat_ptr 
                vector
                
    @param[in]
    v           magmaFloat_ptr 
                vector

    @param[in,out]
    s           magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_sbicgstab_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float alpha,
    magmaFloat_ptr r,
    magmaFloat_ptr v,
    magmaFloat_ptr s, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sbicgstab_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, r, v, s );

   return MAGMA_SUCCESS;
}





__global__ void
magma_sbicgstab_3_kernel(  
    int num_rows,
    int num_cols,
    float alpha,
    float omega,
    float *p,
    float *s,
    float *t,
    float *x,
    float *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            float tmp = s[ i+j*num_rows ];
            x[ i+j*num_rows ] = x[ i+j*num_rows ] 
                        + alpha * p[ i+j*num_rows ] + omega * tmp;
            r[ i+j*num_rows ] = tmp - omega * t[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + alpha * p + omega * s
    r = s - omega * t

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       float
                scalar
                
    @param[in]
    omega       float
                scalar
                
    @param[in]
    p           magmaFloat_ptr 
                vector
                    
    @param[in]
    s           magmaFloat_ptr 
                vector
                    
    @param[in]
    t           magmaFloat_ptr 
                vector

    @param[in,out]
    x           magmaFloat_ptr 
                vector
                
    @param[in,out]
    r           magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_sbicgstab_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float alpha,
    float omega,
    magmaFloat_ptr p,
    magmaFloat_ptr s,
    magmaFloat_ptr t,
    magmaFloat_ptr x,
    magmaFloat_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sbicgstab_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, omega, p, s, t, x, r );

   return MAGMA_SUCCESS;
}




__global__ void
magma_sbicgstab_4_kernel(  
    int num_rows,
    int num_cols,
    float alpha,
    float omega,
    float *y,
    float *z,
    float *s,
    float *t,
    float *x,
    float *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            x[ i+j*num_rows ] = x[ i+j*num_rows ] 
                        + alpha * y[ i+j*num_rows ] + omega * z[ i+j*num_rows ];
            r[ i+j*num_rows ] = s[ i+j*num_rows ] - omega * t[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + alpha * y + omega * z
    r = s - omega * t

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       float
                scalar
                
    @param[in]
    omega       float
                scalar
                
    @param[in]
    y           magmaFloat_ptr 
                vector
                
    @param[in]
    z           magmaFloat_ptr 
                vector
                    
    @param[in]
    s           magmaFloat_ptr 
                vector
                    
    @param[in]
    t           magmaFloat_ptr 
                vector

    @param[in,out]
    x           magmaFloat_ptr 
                vector
                
    @param[in,out]
    r           magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_sbicgstab_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float alpha,
    float omega,
    magmaFloat_ptr y,
    magmaFloat_ptr z,
    magmaFloat_ptr s,
    magmaFloat_ptr t,
    magmaFloat_ptr x,
    magmaFloat_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sbicgstab_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, omega, y, z, s, t, x, r );

   return MAGMA_SUCCESS;
}

