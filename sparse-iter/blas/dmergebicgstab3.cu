/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/blas/zmergebicgstab3.cu normal z -> d, Mon May  2 23:30:46 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512

#define PRECISION_d


// These routines merge multiple kernels from dmergebicgstab into one
// The difference to dmergedbicgstab2 is that the SpMV is not merged into the
// kernes. This results in higher flexibility at the price of lower performance.

/* -------------------------------------------------------------------------- */

__global__ void
magma_dbicgmerge1_kernel(  
    int n, 
    double * skp,
    double * v, 
    double * r, 
    double * p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double beta=skp[1];
    double omega=skp[2];
    if ( i<n ) {
        p[i] =  r[i] + beta * ( p[i] - omega * v[i] );
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    p = beta*p
    p = p-omega*beta*v
    p = p+r
    
    -> p = r + beta * ( p - omega * v ) 

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in]
    skp         magmaDouble_ptr
                set of scalar parameters

    @param[in]
    v           magmaDouble_ptr
                input vector v

    @param[in]
    r           magmaDouble_ptr
                input vector r

    @param[in,out]
    p           magmaDouble_ptr 
                input/output vector p

    @param[in]
    queue       magma_queue_t
                queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_dbicgmerge1(  
    magma_int_t n, 
    magmaDouble_ptr skp,
    magmaDouble_ptr v, 
    magmaDouble_ptr r, 
    magmaDouble_ptr p,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( n, BLOCK_SIZE ) );
    magma_dbicgmerge1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( n, skp, v, r, p );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void
magma_dbicgmerge2_kernel(  
    int n, 
    double * skp, 
    double * r,
    double * v, 
    double * s )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double alpha=skp[0];
    if ( i < n ) {
        s[i] =  r[i] - alpha * v[i];
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    s=r
    s=s-alpha*v
        
    -> s = r - alpha * v

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in]
    skp         magmaDouble_ptr 
                set of scalar parameters

    @param[in]
    r           magmaDouble_ptr 
                input vector r

    @param[in]
    v           magmaDouble_ptr 
                input vector v

    @param[out]
    s           magmaDouble_ptr 
                output vector s

    @param[in]
    queue       magma_queue_t
                queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_dbicgmerge2(  
    magma_int_t n, 
    magmaDouble_ptr skp, 
    magmaDouble_ptr r,
    magmaDouble_ptr v, 
    magmaDouble_ptr s,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( n, BLOCK_SIZE ) );

    magma_dbicgmerge2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( n, skp, r, v, s );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void
magma_dbicgmerge3_kernel(  
    int n, 
    double * skp, 
    double * p,
    double * se,
    double * t,
    double * x, 
    double * r
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double alpha=skp[0];
    double omega=skp[2];
    if ( i<n ) {
        double s;
        s = se[i];
        x[i] = x[i] + alpha * p[i] + omega * s;
        r[i] = s - omega * t[i];
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x=x+alpha*p
    x=x+omega*s
    r=s
    r=r-omega*t
        
    -> x = x + alpha * p + omega * s
    -> r = s - omega * t

    Arguments
    ---------

    @param[in]
    n           int
                dimension n

    @param[in]
    skp         magmaDouble_ptr 
                set of scalar parameters

    @param[in]
    p           magmaDouble_ptr 
                input p

    @param[in]
    s           magmaDouble_ptr 
                input s

    @param[in]
    t           magmaDouble_ptr 
                input t

    @param[in,out]
    x           magmaDouble_ptr 
                input/output x

    @param[in,out]
    r           magmaDouble_ptr 
                input/output r

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_dbicgmerge3(  
    magma_int_t n, 
    magmaDouble_ptr skp,
    magmaDouble_ptr p,
    magmaDouble_ptr s,
    magmaDouble_ptr t,
    magmaDouble_ptr x, 
    magmaDouble_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( n, BLOCK_SIZE ) );
    magma_dbicgmerge3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( n, skp, p, s, t, x, r );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void
magma_dbicgmerge4_kernel_1(  
    double * skp )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i==0 ) {
        double tmp = skp[0];
        skp[0] = skp[4]/tmp;
    }
}

__global__ void
magma_dbicgmerge4_kernel_2(  
    double * skp )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i==0 ) {
        skp[2] = skp[6]/skp[7];
        skp[3] = skp[4];
    }
}

__global__ void
magma_dbicgmerge4_kernel_3(  
    double * skp )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i==0 ) {
        double tmp1 = skp[4]/skp[3];
        double tmp2 = skp[0] / skp[2];
        skp[1] =  tmp1*tmp2;
        //skp[1] =  skp[4]/skp[3] * skp[0] / skp[2];
    }
}

/**
    Purpose
    -------

    Performs some parameter operations for the BiCGSTAB with scalars on GPU.

    Arguments
    ---------

    @param[in]
    type        int
                kernel type

    @param[in,out]
    skp         magmaDouble_ptr 
                vector with parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_dbicgmerge4(  
    magma_int_t type, 
    magmaDouble_ptr skp,
    magma_queue_t queue )
{
    dim3 Bs( 1 );
    dim3 Gs( 1 );
    if ( type == 1 )
        magma_dbicgmerge4_kernel_1<<< Gs, Bs, 0, queue->cuda_stream() >>>( skp );
    else if ( type == 2 )
        magma_dbicgmerge4_kernel_2<<< Gs, Bs, 0, queue->cuda_stream() >>>( skp );
    else if ( type == 3 )
        magma_dbicgmerge4_kernel_3<<< Gs, Bs, 0, queue->cuda_stream() >>>( skp );
    else
        printf("error: no kernel called\n");

   return MAGMA_SUCCESS;
}
