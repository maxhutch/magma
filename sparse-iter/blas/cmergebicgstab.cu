/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergebicgstab.cu normal z -> c, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_c


// These routines merge multiple kernels from bicgstab into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_cbicgstab_1_kernel(  
    int num_rows, 
    int num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex omega,
    magmaFloatComplex *r, 
    magmaFloatComplex *v,
    magmaFloatComplex *p )
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
    beta        magmaFloatComplex
                scalar
                
    @param[in]
    omega       magmaFloatComplex
                scalar
                
    @param[in]
    r           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    v           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    p           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cbicgstab_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex omega,
    magmaFloatComplex_ptr r, 
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr p,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cbicgstab_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, omega,
                     r, v, p );

   return MAGMA_SUCCESS;
}





__global__ void
magma_cbicgstab_2_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr s )
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
    alpha       magmaFloatComplex
                scalar
                
    @param[in]
    r           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    v           magmaFloatComplex_ptr 
                vector

    @param[in,out]
    s           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cbicgstab_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr s, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cbicgstab_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, r, v, s );

   return MAGMA_SUCCESS;
}





__global__ void
magma_cbicgstab_3_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex alpha,
    magmaFloatComplex omega,
    magmaFloatComplex *p,
    magmaFloatComplex *s,
    magmaFloatComplex *t,
    magmaFloatComplex *x,
    magmaFloatComplex *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaFloatComplex tmp = s[ i+j*num_rows ];
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
    alpha       magmaFloatComplex
                scalar
                
    @param[in]
    omega       magmaFloatComplex
                scalar
                
    @param[in]
    p           magmaFloatComplex_ptr 
                vector
                    
    @param[in]
    s           magmaFloatComplex_ptr 
                vector
                    
    @param[in]
    t           magmaFloatComplex_ptr 
                vector

    @param[in,out]
    x           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    r           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cbicgstab_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex omega,
    magmaFloatComplex_ptr p,
    magmaFloatComplex_ptr s,
    magmaFloatComplex_ptr t,
    magmaFloatComplex_ptr x,
    magmaFloatComplex_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cbicgstab_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, omega, p, s, t, x, r );

   return MAGMA_SUCCESS;
}




__global__ void
magma_cbicgstab_4_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex alpha,
    magmaFloatComplex omega,
    magmaFloatComplex *y,
    magmaFloatComplex *z,
    magmaFloatComplex *s,
    magmaFloatComplex *t,
    magmaFloatComplex *x,
    magmaFloatComplex *r )
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
    alpha       magmaFloatComplex
                scalar
                
    @param[in]
    omega       magmaFloatComplex
                scalar
                
    @param[in]
    y           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    z           magmaFloatComplex_ptr 
                vector
                    
    @param[in]
    s           magmaFloatComplex_ptr 
                vector
                    
    @param[in]
    t           magmaFloatComplex_ptr 
                vector

    @param[in,out]
    x           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    r           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_cbicgstab_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex omega,
    magmaFloatComplex_ptr y,
    magmaFloatComplex_ptr z,
    magmaFloatComplex_ptr s,
    magmaFloatComplex_ptr t,
    magmaFloatComplex_ptr x,
    magmaFloatComplex_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_cbicgstab_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, omega, y, z, s, t, x, r );

   return MAGMA_SUCCESS;
}

