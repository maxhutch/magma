/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergetfqmr.cu normal z -> d, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_d


// These routines merge multiple kernels from tfqmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_dtfqmr_1_kernel(  
    int num_rows, 
    int num_cols, 
    double alpha,
    double sigma,
    double *v, 
    double *Au,
    double *u_m,
    double *pu_m,
    double *u_mp1,
    double *w, 
    double *d,
    double *Ad )
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
    alpha       double
                scalar
                
    @param[in]
    sigma       double
                scalar
                
    @param[in]
    v           magmaDouble_ptr 
                vector
                
    @param[in]
    Au          magmaDouble_ptr 
                vector
                
    @param[in,out]
    u_m         magmaDouble_ptr 
                vector
                
    @param[in,out]
    pu_m         magmaDouble_ptr 
                vector
                
    @param[in,out]
    u_mp1       magmaDouble_ptr 
                vector

    @param[in,out]
    w           magmaDouble_ptr 
                vector
                
    @param[in,out]
    d           magmaDouble_ptr 
                vector
                
    @param[in,out]
    Ad          magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dtfqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double alpha,
    double sigma,
    magmaDouble_ptr v, 
    magmaDouble_ptr Au,
    magmaDouble_ptr u_m,
    magmaDouble_ptr pu_m,
    magmaDouble_ptr u_mp1,
    magmaDouble_ptr w, 
    magmaDouble_ptr d,
    magmaDouble_ptr Ad,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dtfqmr_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, sigma,
                     v, Au, u_m, pu_m, u_mp1, w, d, Ad );

   return MAGMA_SUCCESS;
}





__global__ void
magma_dtfqmr_2_kernel(  
    int num_rows,
    int num_cols,
    double eta,
    magmaDouble_ptr d,
    magmaDouble_ptr Ad,
    magmaDouble_ptr x, 
    magmaDouble_ptr r )
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
    eta         double
                scalar
                
    @param[in]
    d           magmaDouble_ptr 
                vector
                
    @param[in]
    Ad          magmaDouble_ptr 
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
magma_dtfqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double eta,
    magmaDouble_ptr d,
    magmaDouble_ptr Ad,
    magmaDouble_ptr x, 
    magmaDouble_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dtfqmr_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, d, Ad, x, r );

   return MAGMA_SUCCESS;
}





__global__ void
magma_dtfqmr_3_kernel(  
    int num_rows,
    int num_cols,
    double beta,
    double *w,
    double *u_m,
    double *u_mp1 )
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
    beta        double
                scalar
                
    @param[in]
    w           magmaDouble_ptr 
                vector
                
    @param[in]
    u_m         magmaDouble_ptr 
                vector

    @param[in,out]
    u_mp1       magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dtfqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    magmaDouble_ptr w,
    magmaDouble_ptr u_m,
    magmaDouble_ptr u_mp1, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dtfqmr_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, w, u_m, u_mp1 );

   return MAGMA_SUCCESS;
}




__global__ void
magma_dtfqmr_4_kernel(  
    int num_rows,
    int num_cols,
    double beta,
    double *Au_new,
    double *v,
    double *Au )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            double tmp = Au_new[ i+j*num_rows ];
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
    beta        double
                scalar
                
    @param[in]
    Au_new      magmaDouble_ptr 
                vector

    @param[in,out]
    v           magmaDouble_ptr 
                vector
                
    @param[in,out]
    Au          magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dtfqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    double beta,
    magmaDouble_ptr Au_new,
    magmaDouble_ptr v,
    magmaDouble_ptr Au, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dtfqmr_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, Au_new, v, Au );

   return MAGMA_SUCCESS;
}


__global__ void
magma_dtfqmr_5_kernel(  
    int num_rows,                   
    int num_cols, 
    double alpha,
    double sigma,
    double *v, 
    double *Au,
    double *u_mp1,
    double *w, 
    double *d,
    double *Ad )
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
    alpha       double
                scalar
                
    @param[in]
    sigma       double
                scalar
                
    @param[in]
    v           magmaDouble_ptr 
                vector
                
    @param[in]
    Au          magmaDouble_ptr 
                vector
                
    @param[in,out]
    u_mp1       magmaDouble_ptr 
                vector

    @param[in,out]
    w           magmaDouble_ptr 
                vector
                
    @param[in,out]
    d           magmaDouble_ptr 
                vector
                
    @param[in,out]
    Ad          magmaDouble_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_dtfqmr_5(  
    magma_int_t num_rows,               
    magma_int_t num_cols, 
    double alpha,
    double sigma,
    magmaDouble_ptr v, 
    magmaDouble_ptr Au,
    magmaDouble_ptr u_mp1,
    magmaDouble_ptr w, 
    magmaDouble_ptr d,
    magmaDouble_ptr Ad,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_dtfqmr_5_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, sigma,
                     v, Au, u_mp1, w, d, Ad );

   return MAGMA_SUCCESS;
}

