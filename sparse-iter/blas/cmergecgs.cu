/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergecgs.cu normal z -> c, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_c


// These routines merge multiple kernels from ccgs into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_ccgs_1_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex beta,
    magmaFloatComplex *r,
    magmaFloatComplex *q,
    magmaFloatComplex *u,
    magmaFloatComplex *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaFloatComplex tmp;
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
    beta        magmaFloatComplex
                scalar

    @param[in]
    r           magmaFloatComplex_ptr 
                vector

    @param[in]
    q           magmaFloatComplex_ptr 
                vector

    @param[in,out]
    u           magmaFloatComplex_ptr 
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
magma_ccgs_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr q, 
    magmaFloatComplex_ptr u,
    magmaFloatComplex_ptr p,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ccgs_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, r, q, u, p );

   return MAGMA_SUCCESS;
}





__global__ void
magma_ccgs_2_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex *r,
    magmaFloatComplex *u,
    magmaFloatComplex *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaFloatComplex tmp;
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
    r           magmaFloatComplex_ptr 
                vector

    @param[in,out]
    u           magmaFloatComplex_ptr 
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
magma_ccgs_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr u,
    magmaFloatComplex_ptr p, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ccgs_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, r, u, p);

   return MAGMA_SUCCESS;
}





__global__ void
magma_ccgs_3_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex alpha,
    magmaFloatComplex *v_hat,
    magmaFloatComplex *u,
    magmaFloatComplex *q,
    magmaFloatComplex *t )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaFloatComplex uloc,  tmp;
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
    alpha       magmaFloatComplex
                scalar
                
    @param[in]
    v_hat       magmaFloatComplex_ptr 
                vector
    
    @param[in]
    u           magmaFloatComplex_ptr 
                vector

    @param[in,out]
    q           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    t           magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ccgs_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr v_hat,
    magmaFloatComplex_ptr u, 
    magmaFloatComplex_ptr q,
    magmaFloatComplex_ptr t, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ccgs_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, v_hat, u, q, t );

   return MAGMA_SUCCESS;
}


__global__ void
magma_ccgs_4_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex alpha,
    magmaFloatComplex *u_hat,
    magmaFloatComplex *t,
    magmaFloatComplex *x,
    magmaFloatComplex *r )
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
    alpha       magmaFloatComplex
                scalar
                
    @param[in]
    u_hat       magmaFloatComplex_ptr 
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
magma_ccgs_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr u_hat,
    magmaFloatComplex_ptr t,
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ccgs_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, u_hat, t, x, r );

   return MAGMA_SUCCESS;
}



