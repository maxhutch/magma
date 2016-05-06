/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergetfqmr.cu normal z -> s, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_s


// These routines merge multiple kernels from tfqmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_stfqmr_1_kernel(  
    int num_rows, 
    int num_cols, 
    float alpha,
    float sigma,
    float *v, 
    float *Au,
    float *u_m,
    float *pu_m,
    float *u_mp1,
    float *w, 
    float *d,
    float *Ad )
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
    alpha       float
                scalar
                
    @param[in]
    sigma       float
                scalar
                
    @param[in]
    v           magmaFloat_ptr 
                vector
                
    @param[in]
    Au          magmaFloat_ptr 
                vector
                
    @param[in,out]
    u_m         magmaFloat_ptr 
                vector
                
    @param[in,out]
    pu_m         magmaFloat_ptr 
                vector
                
    @param[in,out]
    u_mp1       magmaFloat_ptr 
                vector

    @param[in,out]
    w           magmaFloat_ptr 
                vector
                
    @param[in,out]
    d           magmaFloat_ptr 
                vector
                
    @param[in,out]
    Ad          magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_stfqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float alpha,
    float sigma,
    magmaFloat_ptr v, 
    magmaFloat_ptr Au,
    magmaFloat_ptr u_m,
    magmaFloat_ptr pu_m,
    magmaFloat_ptr u_mp1,
    magmaFloat_ptr w, 
    magmaFloat_ptr d,
    magmaFloat_ptr Ad,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_stfqmr_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, sigma,
                     v, Au, u_m, pu_m, u_mp1, w, d, Ad );

   return MAGMA_SUCCESS;
}





__global__ void
magma_stfqmr_2_kernel(  
    int num_rows,
    int num_cols,
    float eta,
    magmaFloat_ptr d,
    magmaFloat_ptr Ad,
    magmaFloat_ptr x, 
    magmaFloat_ptr r )
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
    eta         float
                scalar
                
    @param[in]
    d           magmaFloat_ptr 
                vector
                
    @param[in]
    Ad          magmaFloat_ptr 
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
magma_stfqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float eta,
    magmaFloat_ptr d,
    magmaFloat_ptr Ad,
    magmaFloat_ptr x, 
    magmaFloat_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_stfqmr_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, d, Ad, x, r );

   return MAGMA_SUCCESS;
}





__global__ void
magma_stfqmr_3_kernel(  
    int num_rows,
    int num_cols,
    float beta,
    float *w,
    float *u_m,
    float *u_mp1 )
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
    beta        float
                scalar
                
    @param[in]
    w           magmaFloat_ptr 
                vector
                
    @param[in]
    u_m         magmaFloat_ptr 
                vector

    @param[in,out]
    u_mp1       magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_stfqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float beta,
    magmaFloat_ptr w,
    magmaFloat_ptr u_m,
    magmaFloat_ptr u_mp1, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_stfqmr_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, w, u_m, u_mp1 );

   return MAGMA_SUCCESS;
}




__global__ void
magma_stfqmr_4_kernel(  
    int num_rows,
    int num_cols,
    float beta,
    float *Au_new,
    float *v,
    float *Au )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            float tmp = Au_new[ i+j*num_rows ];
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
    beta        float
                scalar
                
    @param[in]
    Au_new      magmaFloat_ptr 
                vector

    @param[in,out]
    v           magmaFloat_ptr 
                vector
                
    @param[in,out]
    Au          magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_stfqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float beta,
    magmaFloat_ptr Au_new,
    magmaFloat_ptr v,
    magmaFloat_ptr Au, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_stfqmr_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, Au_new, v, Au );

   return MAGMA_SUCCESS;
}


__global__ void
magma_stfqmr_5_kernel(  
    int num_rows,                   
    int num_cols, 
    float alpha,
    float sigma,
    float *v, 
    float *Au,
    float *u_mp1,
    float *w, 
    float *d,
    float *Ad )
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
    alpha       float
                scalar
                
    @param[in]
    sigma       float
                scalar
                
    @param[in]
    v           magmaFloat_ptr 
                vector
                
    @param[in]
    Au          magmaFloat_ptr 
                vector
                
    @param[in,out]
    u_mp1       magmaFloat_ptr 
                vector

    @param[in,out]
    w           magmaFloat_ptr 
                vector
                
    @param[in,out]
    d           magmaFloat_ptr 
                vector
                
    @param[in,out]
    Ad          magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_stfqmr_5(  
    magma_int_t num_rows,               
    magma_int_t num_cols, 
    float alpha,
    float sigma,
    magmaFloat_ptr v, 
    magmaFloat_ptr Au,
    magmaFloat_ptr u_mp1,
    magmaFloat_ptr w, 
    magmaFloat_ptr d,
    magmaFloat_ptr Ad,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_stfqmr_5_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, sigma,
                     v, Au, u_mp1, w, d, Ad );

   return MAGMA_SUCCESS;
}

