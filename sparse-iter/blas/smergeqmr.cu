/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergeqmr.cu normal z -> s, Mon May  2 23:30:47 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_s


// These routines merge multiple kernels from qmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_sqmr_1_kernel(  
    int num_rows, 
    int num_cols, 
    float rho,
    float psi,
    float *y, 
    float *z,
    float *v,
    float *w )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            float ytmp = y[ i+j*num_rows ] / rho;
            y[ i+j*num_rows ] = ytmp;
            v[ i+j*num_rows ] = ytmp;
            
            float ztmp = z[ i+j*num_rows ] / psi;
            z[ i+j*num_rows ] = ztmp;
            w[ i+j*num_rows ] = ztmp;
            
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    v = y / rho
    y = y / rho
    w = wt / psi
    z = z / psi
    
    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    rho         float
                scalar
                
    @param[in]
    psi         float
                scalar
                
    @param[in,out]
    y           magmaFloat_ptr 
                vector
                
    @param[in,out]
    z           magmaFloat_ptr 
                vector
                
    @param[in,out]
    v           magmaFloat_ptr 
                vector

    @param[in,out]
    w           magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_sqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float rho,
    float psi,
    magmaFloat_ptr y, 
    magmaFloat_ptr z,
    magmaFloat_ptr v,
    magmaFloat_ptr w,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sqmr_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, rho, psi,
                     y, z, v, w );

   return MAGMA_SUCCESS;
}





__global__ void
magma_sqmr_2_kernel(  
    int num_rows,
    int num_cols,
    float pde,
    float rde,
    magmaFloat_ptr y,
    magmaFloat_ptr z,
    magmaFloat_ptr p, 
    magmaFloat_ptr q )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            p[ i+j*num_rows ] = y[ i+j*num_rows ] - pde * p[ i+j*num_rows ];
            q[ i+j*num_rows ] = z[ i+j*num_rows ] - rde * q[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    p = y - pde * p
    q = z - rde * q

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    pde         float
                scalar

    @param[in]
    qde         float
                scalar
                
    @param[in]
    y           magmaFloat_ptr 
                vector
                
    @param[in]
    z           magmaFloat_ptr 
                vector

    @param[in,out]
    p           magmaFloat_ptr 
                vector
                
    @param[in,out]
    q           magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_sqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float pde,
    float rde,
    magmaFloat_ptr y,
    magmaFloat_ptr z,
    magmaFloat_ptr p, 
    magmaFloat_ptr q, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sqmr_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, pde, rde, y, z, p, q );

   return MAGMA_SUCCESS;
}





__global__ void
magma_sqmr_3_kernel(  
    int num_rows,
    int num_cols,
    float beta,
    float *pt,
    float *v,
    float *y )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            float tmp = pt[ i+j*num_rows ] - beta * v[ i+j*num_rows ];
            v[ i+j*num_rows ] = tmp;
            y[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    v = pt - beta * v
    y = v

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
    pt          magmaFloat_ptr 
                vector

    @param[in,out]
    v           magmaFloat_ptr 
                vector
                
    @param[in,out]
    y           magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_sqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float beta,
    magmaFloat_ptr pt,
    magmaFloat_ptr v,
    magmaFloat_ptr y,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sqmr_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, pt, v, y );

   return MAGMA_SUCCESS;
}




__global__ void
magma_sqmr_4_kernel(  
    int num_rows,
    int num_cols,
    float eta,
    float *p,
    float *pt,
    float *d,
    float *s,
    float *x,
    float *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            float tmpd = eta * p[ i+j*num_rows ];
            d[ i+j*num_rows ] = tmpd;
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + tmpd;
            float tmps = eta * pt[ i+j*num_rows ];
            s[ i+j*num_rows ] = tmps;
            r[ i+j*num_rows ] = r[ i+j*num_rows ] - tmps;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    d = eta * p;
    s = eta * pt;
    x = x + d;
    r = r - s;

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
    p           magmaFloat_ptr 
                vector
                
    @param[in]
    pt          magmaFloat_ptr 
                vector

    @param[in,out]
    d           magmaFloat_ptr 
                vector
                
    @param[in,out]
    s           magmaFloat_ptr 
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
magma_sqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float eta,
    magmaFloat_ptr p,
    magmaFloat_ptr pt,
    magmaFloat_ptr d, 
    magmaFloat_ptr s, 
    magmaFloat_ptr x, 
    magmaFloat_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sqmr_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, p, pt, d, s, x, r );

   return MAGMA_SUCCESS;
}


__global__ void
magma_sqmr_5_kernel(  
    int num_rows,
    int num_cols,
    float eta,
    float pds,
    float *p,
    float *pt,
    float *d,
    float *s,
    float *x,
    float *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            float tmpd = eta * p[ i+j*num_rows ] + pds * d[ i+j*num_rows ];
            d[ i+j*num_rows ] = tmpd;
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + tmpd;
            float tmps = eta * pt[ i+j*num_rows ] + pds * s[ i+j*num_rows ];
            s[ i+j*num_rows ] = tmps;
            r[ i+j*num_rows ] = r[ i+j*num_rows ] - tmps;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    d = eta * p + pds * d;
    s = eta * pt + pds * s;
    x = x + d;
    r = r - s;

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
    pds         float
                scalar
                
    @param[in]
    p           magmaFloat_ptr 
                vector
                
    @param[in]
    pt          magmaFloat_ptr 
                vector

    @param[in,out]
    d           magmaFloat_ptr 
                vector
                
    @param[in,out]
    s           magmaFloat_ptr 
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
magma_sqmr_5(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float eta,
    float pds,
    magmaFloat_ptr p,
    magmaFloat_ptr pt,
    magmaFloat_ptr d, 
    magmaFloat_ptr s, 
    magmaFloat_ptr x, 
    magmaFloat_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sqmr_5_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, pds, p, pt, d, s, x, r );

   return MAGMA_SUCCESS;
}


__global__ void
magma_sqmr_6_kernel(  
    int num_rows, 
    int num_cols, 
    float beta,
    float rho,
    float psi,
    float *y, 
    float *z,
    float *v,
    float *w,
    float *wt )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            
            float wttmp = wt[ i+j*num_rows ]
                                - MAGMA_S_CONJ( beta ) * w[ i+j*num_rows ];
                                
            wt[ i+j*num_rows ] = wttmp;
            
            float ztmp = wttmp / psi;
            z[ i+j*num_rows ] = ztmp;
            w[ i+j*num_rows ] = ztmp;
            
            float ytmp = y[ i+j*num_rows ] / rho;
            y[ i+j*num_rows ] = ytmp;
            v[ i+j*num_rows ] = ytmp;
            
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:
    
    wt = wt - conj(beta) * w
    v = y / rho
    y = y / rho
    w = wt / psi
    z = wt / psi
    
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
    rho         float
                scalar
                
    @param[in]
    psi         float
                scalar
                
    @param[in,out]
    y           magmaFloat_ptr 
                vector
                
    @param[in,out]
    z           magmaFloat_ptr 
                vector
                
    @param[in,out]
    v           magmaFloat_ptr 
                vector

    @param[in,out]
    w           magmaFloat_ptr 
                vector
                    
    @param[in,out]
    wt          magmaFloat_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_sqmr_6(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    float beta,
    float rho,
    float psi,
    magmaFloat_ptr y, 
    magmaFloat_ptr z,
    magmaFloat_ptr v,
    magmaFloat_ptr w,
    magmaFloat_ptr wt,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_sqmr_6_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, rho, psi,
                     y, z, v, w, wt );

   return MAGMA_SUCCESS;
}
