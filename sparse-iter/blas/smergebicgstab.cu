/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zmergebicgstab.cu normal z -> s, Fri Jul 18 17:34:28 2014
       @author Hartwig Anzt

*/
#include "common_magma.h"

#define BLOCK_SIZE 512

#define PRECISION_s


// These routines merge multiple kernels from smergebicgstab into one
// The difference to smergedbicgstab2 is that the SpMV is not merged into the
// kernes. This results in higher flexibility at the price of lower performance.

/* -------------------------------------------------------------------------- */

__global__ void 
magma_sbicgmerge1_kernel(  
                    int n, 
                    float *skp,
                    float *v, 
                    float *r, 
                    float *p ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float beta=skp[1];
    float omega=skp[2];
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

    @param
    n           int
                dimension n

    @param
    skp         float*
                set of scalar parameters

    @param
    v           float*
                input v

    @param
    r           float*
                input r

    @param
    p           float*
                input/output p


    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" int
magma_sbicgmerge1(  int n, 
                    float *skp,
                    float *v, 
                    float *r, 
                    float *p ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE-1)/BLOCK_SIZE );
    magma_sbicgmerge1_kernel<<<Gs, Bs, 0>>>( n, skp, v, r, p );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void 
magma_sbicgmerge2_kernel(  
                    int n, 
                    float *skp, 
                    float *r,
                    float *v, 
                    float *s ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha=skp[0];
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

    @param
    n           int
                dimension n

    @param
    skp         float*
                set of scalar parameters

    @param
    r           float*
                input r

    @param
    v           float*
                input v

    @param
    s           float*
                input/output s


    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" int
magma_sbicgmerge2(  int n, 
                    float *skp, 
                    float *r,
                    float *v, 
                    float *s ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE-1)/BLOCK_SIZE );

    magma_sbicgmerge2_kernel<<<Gs, Bs, 0>>>( n, skp, r, v, s );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void 
magma_sbicgmerge3_kernel(  
                    int n, 
                    float *skp, 
                    float *p,
                    float *se,
                    float *t,
                    float *x, 
                    float *r ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float alpha=skp[0];
    float omega=skp[2];
    if( i<n ){
        float s;
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

    @param
    n           int
                dimension n

    @param
    skp         float*
                set of scalar parameters

    @param
    p           float*
                input p

    @param
    s           float*
                input s

    @param
    t           float*
                input t

    @param
    x           float*
                input/output x

    @param
    r           float*
                input/output r


    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" int
magma_sbicgmerge3(  int n, 
                    float *skp,
                    float *p,
                    float *s,
                    float *t,
                    float *x, 
                    float *r ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE-1)/BLOCK_SIZE );
    magma_sbicgmerge3_kernel<<<Gs, Bs, 0>>>( n, skp, p, s, t, x, r );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void 
magma_sbicgmerge4_kernel_1(  
                    float *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        float tmp = skp[0];
        skp[0] = skp[4]/tmp;
    }
}

__global__ void 
magma_sbicgmerge4_kernel_2(  
                    float *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        skp[2] = skp[6]/skp[7];
        skp[3] = skp[4];
    }
}

__global__ void 
magma_sbicgmerge4_kernel_3(  
                    float *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        float tmp1 = skp[4]/skp[3];
        float tmp2 = skp[0] / skp[2];
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

    @param
    type        int
                kernel type

    @param
    skp         float*
                vector with parameters


    @ingroup magmasparse_sgegpuk
    ********************************************************************/

extern "C" int
magma_sbicgmerge4(  int type, 
                    float *skp ){

    dim3 Bs( 2 );
    dim3 Gs( 1 );
    if( type == 1 )
        magma_sbicgmerge4_kernel_1<<<Gs, Bs, 0>>>( skp );
    else if( type == 2 )
        magma_sbicgmerge4_kernel_2<<<Gs, Bs, 0>>>( skp );
    else if( type == 3 )
        magma_sbicgmerge4_kernel_3<<<Gs, Bs, 0>>>( skp );
    else
        printf("error: no kernel called\n");

   return MAGMA_SUCCESS;
}

