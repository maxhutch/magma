/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergebicgstab.cu normal z -> d, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_d


// These routines merge multiple kernels from bicgstab into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_dbicgstab_1_kernel(  
    int num_rows, 
    int num_cols, 
    double beta,
    double omega,
    double *r, 
    double *v,
    double *p )
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
    beta        double
                scalar
                
    @param[in]
    omega       double
                scalar
                
    @param[in]
    r           magmaDouble_ptr 
                vector
                
    @param[in]
    v           magmaDouble_ptr 
                vector
                
    @param[in,out]
    p           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dbicgstab_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    double omega,
    magmaDouble_ptr r, 
    magmaDouble_ptr v,
    magmaDouble_ptr p,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dbicgstab_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, omega,
                     r, v, p );

   return MAGMA_SUCCESS;
}





__global__ void
magma_dbicgstab_2_kernel(  
    int num_rows,
    int num_cols,
    double alpha,
    magmaDouble_ptr r,
    magmaDouble_ptr v,
    magmaDouble_ptr s )
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
    alpha       double
                scalar
                
    @param[in]
    r           magmaDouble_ptr 
                vector
                
    @param[in]
    v           magmaDouble_ptr 
                vector

    @param[in,out]
    s           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dbicgstab_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    magmaDouble_ptr r,
    magmaDouble_ptr v,
    magmaDouble_ptr s, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dbicgstab_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, r, v, s );

   return MAGMA_SUCCESS;
}





__global__ void
magma_dbicgstab_3_kernel(  
    int num_rows,
    int num_cols,
    double alpha,
    double omega,
    double *p,
    double *s,
    double *t,
    double *x,
    double *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            double tmp = s[ i+j*num_rows ];
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
    alpha       double
                scalar
                
    @param[in]
    omega       double
                scalar
                
    @param[in]
    p           magmaDouble_ptr 
                vector
                    
    @param[in]
    s           magmaDouble_ptr 
                vector
                    
    @param[in]
    t           magmaDouble_ptr 
                vector

    @param[in,out]
    x           magmaDouble_ptr 
                vector
                
    @param[in,out]
    r           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dbicgstab_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    double omega,
    magmaDouble_ptr p,
    magmaDouble_ptr s,
    magmaDouble_ptr t,
    magmaDouble_ptr x,
    magmaDouble_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dbicgstab_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, omega, p, s, t, x, r );

   return MAGMA_SUCCESS;
}




__global__ void
magma_dbicgstab_4_kernel(  
    int num_rows,
    int num_cols,
    double alpha,
    double omega,
    double *y,
    double *z,
    double *s,
    double *t,
    double *x,
    double *r )
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
    alpha       double
                scalar
                
    @param[in]
    omega       double
                scalar
                
    @param[in]
    y           magmaDouble_ptr 
                vector
                
    @param[in]
    z           magmaDouble_ptr 
                vector
                    
    @param[in]
    s           magmaDouble_ptr 
                vector
                    
    @param[in]
    t           magmaDouble_ptr 
                vector

    @param[in,out]
    x           magmaDouble_ptr 
                vector
                
    @param[in,out]
    r           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dbicgstab_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    double omega,
    magmaDouble_ptr y,
    magmaDouble_ptr z,
    magmaDouble_ptr s,
    magmaDouble_ptr t,
    magmaDouble_ptr x,
    magmaDouble_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dbicgstab_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, omega, y, z, s, t, x, r );

   return MAGMA_SUCCESS;
}

