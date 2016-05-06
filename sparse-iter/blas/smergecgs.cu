/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergecgs.cu normal z -> s, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_s


// These routines merge multiple kernels from scgs into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_scgs_1_kernel(  
    int num_rows,
    int num_cols,
    float beta,
    float *r,
    float *q,
    float *u,
    float *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            float tmp;
            tmp =  r[ i+j*num_rows ] + beta * q[ i+j*num_rows ];
            p[ i+j*num_rows ] = tmp + beta * q[ i+j*num_rows ] 
                                + beta * beta * p[ i+j*num_rows ];
            u[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    u = r + beta q
    p = u + beta*(q + beta*p)

    Arguments
    ---------

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
    r           magmaFloat_ptr 
                vector

    @param[in]
    q           magmaFloat_ptr 
                vector

    @param[in,out]
    u           magmaFloat_ptr 
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
magma_scgs_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float beta,
    magmaFloat_ptr r,
    magmaFloat_ptr q, 
    magmaFloat_ptr u,
    magmaFloat_ptr p,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_scgs_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, r, q, u, p );

   return MAGMA_SUCCESS;
}





__global__ void
magma_scgs_2_kernel(  
    int num_rows,
    int num_cols,
    float *r,
    float *u,
    float *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            float tmp;
            tmp = r[ i+j*num_rows ];
            u[ i+j*num_rows ] = tmp;
            p[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    u = r
    p = r

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    r           magmaFloat_ptr 
                vector

    @param[in,out]
    u           magmaFloat_ptr 
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
magma_scgs_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloat_ptr r,
    magmaFloat_ptr u,
    magmaFloat_ptr p, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_scgs_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, r, u, p);

   return MAGMA_SUCCESS;
}





__global__ void
magma_scgs_3_kernel(  
    int num_rows,
    int num_cols,
    float alpha,
    float *v_hat,
    float *u,
    float *q,
    float *t )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            float uloc,  tmp;
            uloc = u[ i+j*num_rows ];
            tmp = uloc - alpha * v_hat[ i+j*num_rows ];
            t[ i+j*num_rows ] = tmp + uloc;
            q[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    q = u - alpha v_hat
    t = u + q

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
    v_hat       magmaFloat_ptr 
                vector
    
    @param[in]
    u           magmaFloat_ptr 
                vector

    @param[in,out]
    q           magmaFloat_ptr 
                vector
                
    @param[in,out]
    t           magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_scgs_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float alpha,
    magmaFloat_ptr v_hat,
    magmaFloat_ptr u, 
    magmaFloat_ptr q,
    magmaFloat_ptr t, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_scgs_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, v_hat, u, q, t );

   return MAGMA_SUCCESS;
}


__global__ void
magma_scgs_4_kernel(  
    int num_rows,
    int num_cols,
    float alpha,
    float *u_hat,
    float *t,
    float *x,
    float *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            x[ i+j*num_rows ] = x[ i+j*num_rows ] 
                                + alpha * u_hat[ i+j*num_rows ];
            r[ i+j*num_rows ] = r[ i+j*num_rows ] 
                                - alpha * t[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + alpha u_hat
    r = r -alpha*A u_hat = r -alpha*t

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
    u_hat       magmaFloat_ptr 
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
magma_scgs_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float alpha,
    magmaFloat_ptr u_hat,
    magmaFloat_ptr t,
    magmaFloat_ptr x, 
    magmaFloat_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_scgs_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, u_hat, t, x, r );

   return MAGMA_SUCCESS;
}



