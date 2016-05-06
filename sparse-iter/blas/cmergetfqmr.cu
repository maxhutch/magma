/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergetfqmr.cu normal z -> c, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_c


// These routines merge multiple kernels from tfqmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_ctfqmr_1_kernel(  
    int num_rows, 
    int num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex sigma,
    magmaFloatComplex *v, 
    magmaFloatComplex *Au,
    magmaFloatComplex *u_m,
    magmaFloatComplex *pu_m,
    magmaFloatComplex *u_mp1,
    magmaFloatComplex *w, 
    magmaFloatComplex *d,
    magmaFloatComplex *Ad )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            u_mp1[ i+j*num_rows ] = u_m[ i+j*num_rows ] - alpha * v[ i+j*num_rows ];
            w[ i+j*num_rows ] = w[ i+j*num_rows ] - alpha * Au[ i+j*num_rows ];
            d[ i+j*num_rows ] = pu_m[ i+j*num_rows ] + sigma * d[ i+j*num_rows ];
            Ad[ i+j*num_rows ] = Au[ i+j*num_rows ] + sigma * Ad[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    u_mp1 = u_mp1 - alpha*v;
    w = w - alpha*Au;
    d = pu_m + sigma*d;
    Ad = Au + sigma*Ad;
    
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
    sigma       magmaFloatComplex
                scalar
                
    @param[in]
    v           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    Au          magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    u_m         magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    pu_m         magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    u_mp1       magmaFloatComplex_ptr 
                vector

    @param[in,out]
    w           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    d           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    Ad          magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ctfqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex sigma,
    magmaFloatComplex_ptr v, 
    magmaFloatComplex_ptr Au,
    magmaFloatComplex_ptr u_m,
    magmaFloatComplex_ptr pu_m,
    magmaFloatComplex_ptr u_mp1,
    magmaFloatComplex_ptr w, 
    magmaFloatComplex_ptr d,
    magmaFloatComplex_ptr Ad,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ctfqmr_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, sigma,
                     v, Au, u_m, pu_m, u_mp1, w, d, Ad );

   return MAGMA_SUCCESS;
}





__global__ void
magma_ctfqmr_2_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex eta,
    magmaFloatComplex_ptr d,
    magmaFloatComplex_ptr Ad,
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + eta * d[ i+j*num_rows ];
            r[ i+j*num_rows ] = r[ i+j*num_rows ] - eta * Ad[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + eta * d
    r = r - eta * Ad

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    eta         magmaFloatComplex
                scalar
                
    @param[in]
    d           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    Ad          magmaFloatComplex_ptr 
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
magma_ctfqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex eta,
    magmaFloatComplex_ptr d,
    magmaFloatComplex_ptr Ad,
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ctfqmr_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, d, Ad, x, r );

   return MAGMA_SUCCESS;
}





__global__ void
magma_ctfqmr_3_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex beta,
    magmaFloatComplex *w,
    magmaFloatComplex *u_m,
    magmaFloatComplex *u_mp1 )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            u_mp1[ i+j*num_rows ] = w[ i+j*num_rows ] + beta * u_m[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    u_mp1 = w + beta*u_mp1

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
    w           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    u_m         magmaFloatComplex_ptr 
                vector

    @param[in,out]
    u_mp1       magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ctfqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr w,
    magmaFloatComplex_ptr u_m,
    magmaFloatComplex_ptr u_mp1, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ctfqmr_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, w, u_m, u_mp1 );

   return MAGMA_SUCCESS;
}




__global__ void
magma_ctfqmr_4_kernel(  
    int num_rows,
    int num_cols,
    magmaFloatComplex beta,
    magmaFloatComplex *Au_new,
    magmaFloatComplex *v,
    magmaFloatComplex *Au )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaFloatComplex tmp = Au_new[ i+j*num_rows ];
                v[ i+j*num_rows ] = tmp + beta * Au[ i+j*num_rows ] 
                                    + beta * beta * v[ i+j*num_rows ];
                Au[ i+j*num_rows ] = tmp; 
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    v = Au_new + beta*(Au+beta*v);
    Au = Au_new

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
    Au_new      magmaFloatComplex_ptr 
                vector

    @param[in,out]
    v           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    Au          magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ctfqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaFloatComplex beta,
    magmaFloatComplex_ptr Au_new,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr Au, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ctfqmr_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, Au_new, v, Au );

   return MAGMA_SUCCESS;
}


__global__ void
magma_ctfqmr_5_kernel(  
    int num_rows,                   
    int num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex sigma,
    magmaFloatComplex *v, 
    magmaFloatComplex *Au,
    magmaFloatComplex *u_mp1,
    magmaFloatComplex *w, 
    magmaFloatComplex *d,
    magmaFloatComplex *Ad )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            w[ i+j*num_rows ] = w[ i+j*num_rows ] - alpha * Au[ i+j*num_rows ];
            d[ i+j*num_rows ] = u_mp1[ i+j*num_rows ] + sigma * d[ i+j*num_rows ];
            Ad[ i+j*num_rows ] = Au[ i+j*num_rows ] + sigma * Ad[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    w = w - alpha*Au;
    d = pu_m + sigma*d;
    Ad = Au + sigma*Ad;
    
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
    sigma       magmaFloatComplex
                scalar
                
    @param[in]
    v           magmaFloatComplex_ptr 
                vector
                
    @param[in]
    Au          magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    u_mp1       magmaFloatComplex_ptr 
                vector

    @param[in,out]
    w           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    d           magmaFloatComplex_ptr 
                vector
                
    @param[in,out]
    Ad          magmaFloatComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ctfqmr_5(  
    magma_int_t num_rows,               
    magma_int_t num_cols, 
    magmaFloatComplex alpha,
    magmaFloatComplex sigma,
    magmaFloatComplex_ptr v, 
    magmaFloatComplex_ptr Au,
    magmaFloatComplex_ptr u_mp1,
    magmaFloatComplex_ptr w, 
    magmaFloatComplex_ptr d,
    magmaFloatComplex_ptr Ad,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ctfqmr_5_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, sigma,
                     v, Au, u_mp1, w, d, Ad );

   return MAGMA_SUCCESS;
}

