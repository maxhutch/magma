/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from zmergebicgstab.cu normal z -> c, Sun May  3 11:22:58 2015
       @author Hartwig Anzt

*/
#include "common_magma.h"

#define BLOCK_SIZE 512

#define PRECISION_c


// These routines merge multiple kernels from cmergebicgstab into one
// The difference to cmergedbicgstab2 is that the SpMV is not merged into the
// kernes. This results in higher flexibility at the price of lower performance.

/* -------------------------------------------------------------------------- */

__global__ void
magma_cbicgmerge1_kernel(  
    int n, 
    magmaFloatComplex * skp,
    magmaFloatComplex * v, 
    magmaFloatComplex * r, 
    magmaFloatComplex * p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaFloatComplex beta=skp[1];
    magmaFloatComplex omega=skp[2];
    if( i<n ){
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
    skp         magmaFloatComplex_ptr 
                set of scalar parameters

    @param[in]
    v           magmaFloatComplex_ptr 
                input v

    @param[in]
    r           magmaFloatComplex_ptr 
                input r

    @param[in/out]
    p           magmaFloatComplex_ptr 
                input/output p

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" int
magma_cbicgmerge1(  
    int n, 
    magmaFloatComplex_ptr skp,
    magmaFloatComplex_ptr v, 
    magmaFloatComplex_ptr r, 
    magmaFloatComplex_ptr p ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( n, BLOCK_SIZE ) );
    magma_cbicgmerge1_kernel<<<Gs, Bs, 0>>>( n, skp, v, r, p );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void
magma_cbicgmerge2_kernel(  
    int n, 
    magmaFloatComplex * skp, 
    magmaFloatComplex * r,
    magmaFloatComplex * v, 
    magmaFloatComplex * s )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaFloatComplex alpha=skp[0];
    if( i<n ){
        s[i] =  r[i] - alpha * v[i] ;
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
    skp         magmaFloatComplex_ptr 
                set of scalar parameters

    @param[in]
    r           magmaFloatComplex_ptr 
                input r

    @param[in]
    v           magmaFloatComplex_ptr 
                input v

    @param[s]
    s           magmaFloatComplex_ptr 
                output s

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" int
magma_cbicgmerge2(  
    int n, 
    magmaFloatComplex_ptr skp, 
    magmaFloatComplex_ptr r,
    magmaFloatComplex_ptr v, 
    magmaFloatComplex_ptr s )
{

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( n, BLOCK_SIZE ) );

    magma_cbicgmerge2_kernel<<<Gs, Bs, 0>>>( n, skp, r, v, s );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void
magma_cbicgmerge3_kernel(  
    int n, 
    magmaFloatComplex * skp, 
    magmaFloatComplex * p,
    magmaFloatComplex * se,
    magmaFloatComplex * t,
    magmaFloatComplex * x, 
    magmaFloatComplex * r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaFloatComplex alpha=skp[0];
    magmaFloatComplex omega=skp[2];
    if( i<n ){
        magmaFloatComplex s;
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
    skp         magmaFloatComplex_ptr 
                set of scalar parameters

    @param[in]
    p           magmaFloatComplex_ptr 
                input p

    @param[in]
    s           magmaFloatComplex_ptr 
                input s

    @param[in]
    t           magmaFloatComplex_ptr 
                input t

    @param[in/out]
    x           magmaFloatComplex_ptr 
                input/output x

    @param[in/out]
    r           magmaFloatComplex_ptr 
                input/output r

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" int
magma_cbicgmerge3(  
    int n, 
    magmaFloatComplex_ptr skp,
    magmaFloatComplex_ptr p,
    magmaFloatComplex_ptr s,
    magmaFloatComplex_ptr t,
    magmaFloatComplex_ptr x, 
    magmaFloatComplex_ptr r )
{

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( n, BLOCK_SIZE ) );
    magma_cbicgmerge3_kernel<<<Gs, Bs, 0>>>( n, skp, p, s, t, x, r );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void
magma_cbicgmerge4_kernel_1(  
    magmaFloatComplex * skp )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaFloatComplex tmp = skp[0];
        skp[0] = skp[4]/tmp;
    }
}

__global__ void
magma_cbicgmerge4_kernel_2(  
    magmaFloatComplex * skp )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        skp[2] = skp[6]/skp[7];
        skp[3] = skp[4];
    }
}

__global__ void
magma_cbicgmerge4_kernel_3(  
    magmaFloatComplex * skp )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaFloatComplex tmp1 = skp[4]/skp[3];
        magmaFloatComplex tmp2 = skp[0] / skp[2];
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

    @param[in/out]
    skp         magmaFloatComplex_ptr 
                vector with parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgegpuk
    ********************************************************************/

extern "C" int
magma_cbicgmerge4(  
    int type, 
    magmaFloatComplex_ptr skp )
{

    dim3 Bs( 1 );
    dim3 Gs( 1 );
    if( type == 1 )
        magma_cbicgmerge4_kernel_1<<<Gs, Bs, 0>>>( skp );
    else if( type == 2 )
        magma_cbicgmerge4_kernel_2<<<Gs, Bs, 0>>>( skp );
    else if( type == 3 )
        magma_cbicgmerge4_kernel_3<<<Gs, Bs, 0>>>( skp );
    else
        printf("error: no kernel called\n");

   return MAGMA_SUCCESS;
}

