/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergecgs.cu normal z -> d, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_d


// These routines merge multiple kernels from dcgs into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_dcgs_1_kernel(  
    int num_rows,
    int num_cols,
    double beta,
    double *r,
    double *q,
    double *u,
    double *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            double tmp;
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
    beta        double
                scalar

    @param[in]
    r           magmaDouble_ptr 
                vector

    @param[in]
    q           magmaDouble_ptr 
                vector

    @param[in,out]
    u           magmaDouble_ptr 
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
magma_dcgs_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    magmaDouble_ptr r,
    magmaDouble_ptr q, 
    magmaDouble_ptr u,
    magmaDouble_ptr p,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dcgs_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, r, q, u, p );

   return MAGMA_SUCCESS;
}





__global__ void
magma_dcgs_2_kernel(  
    int num_rows,
    int num_cols,
    double *r,
    double *u,
    double *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            double tmp;
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
    r           magmaDouble_ptr 
                vector

    @param[in,out]
    u           magmaDouble_ptr 
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
magma_dcgs_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDouble_ptr r,
    magmaDouble_ptr u,
    magmaDouble_ptr p, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dcgs_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, r, u, p);

   return MAGMA_SUCCESS;
}





__global__ void
magma_dcgs_3_kernel(  
    int num_rows,
    int num_cols,
    double alpha,
    double *v_hat,
    double *u,
    double *q,
    double *t )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            double uloc,  tmp;
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
    alpha       double
                scalar
                
    @param[in]
    v_hat       magmaDouble_ptr 
                vector
    
    @param[in]
    u           magmaDouble_ptr 
                vector

    @param[in,out]
    q           magmaDouble_ptr 
                vector
                
    @param[in,out]
    t           magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dcgs_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    magmaDouble_ptr v_hat,
    magmaDouble_ptr u, 
    magmaDouble_ptr q,
    magmaDouble_ptr t, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dcgs_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, v_hat, u, q, t );

   return MAGMA_SUCCESS;
}


__global__ void
magma_dcgs_4_kernel(  
    int num_rows,
    int num_cols,
    double alpha,
    double *u_hat,
    double *t,
    double *x,
    double *r )
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
    alpha       double
                scalar
                
    @param[in]
    u_hat       magmaDouble_ptr 
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
magma_dcgs_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    magmaDouble_ptr u_hat,
    magmaDouble_ptr t,
    magmaDouble_ptr x, 
    magmaDouble_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dcgs_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, u_hat, t, x, r );

   return MAGMA_SUCCESS;
}



