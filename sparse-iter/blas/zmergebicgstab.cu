/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "common_magma.h"

#define BLOCK_SIZE 512

#define PRECISION_z


// These routines merge multiple kernels from zmergebicgstab into one
// The difference to zmergedbicgstab2 is that the SpMV is not merged into the
// kernes. This results in higher flexibility at the price of lower performance.

/* -------------------------------------------------------------------------- */

__global__ void 
magma_zbicgmerge1_kernel(  
                    int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *p ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaDoubleComplex beta=skp[1];
    magmaDoubleComplex omega=skp[2];
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
    skp         magmaDoubleComplex*
                set of scalar parameters

    @param
    v           magmaDoubleComplex*
                input v

    @param
    r           magmaDoubleComplex*
                input r

    @param
    p           magmaDoubleComplex*
                input/output p


    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" int
magma_zbicgmerge1(  int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *r, 
                    magmaDoubleComplex *p ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE-1)/BLOCK_SIZE );
    magma_zbicgmerge1_kernel<<<Gs, Bs, 0>>>( n, skp, v, r, p );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void 
magma_zbicgmerge2_kernel(  
                    int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *s ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaDoubleComplex alpha=skp[0];
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
    skp         magmaDoubleComplex*
                set of scalar parameters

    @param
    r           magmaDoubleComplex*
                input r

    @param
    v           magmaDoubleComplex*
                input v

    @param
    s           magmaDoubleComplex*
                input/output s


    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" int
magma_zbicgmerge2(  int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *v, 
                    magmaDoubleComplex *s ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE-1)/BLOCK_SIZE );

    magma_zbicgmerge2_kernel<<<Gs, Bs, 0>>>( n, skp, r, v, s );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void 
magma_zbicgmerge3_kernel(  
                    int n, 
                    magmaDoubleComplex *skp, 
                    magmaDoubleComplex *p,
                    magmaDoubleComplex *se,
                    magmaDoubleComplex *t,
                    magmaDoubleComplex *x, 
                    magmaDoubleComplex *r ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    magmaDoubleComplex alpha=skp[0];
    magmaDoubleComplex omega=skp[2];
    if( i<n ){
        magmaDoubleComplex s;
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
    skp         magmaDoubleComplex*
                set of scalar parameters

    @param
    p           magmaDoubleComplex*
                input p

    @param
    s           magmaDoubleComplex*
                input s

    @param
    t           magmaDoubleComplex*
                input t

    @param
    x           magmaDoubleComplex*
                input/output x

    @param
    r           magmaDoubleComplex*
                input/output r


    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" int
magma_zbicgmerge3(  int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *p,
                    magmaDoubleComplex *s,
                    magmaDoubleComplex *t,
                    magmaDoubleComplex *x, 
                    magmaDoubleComplex *r ){

    
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( (n+BLOCK_SIZE-1)/BLOCK_SIZE );
    magma_zbicgmerge3_kernel<<<Gs, Bs, 0>>>( n, skp, p, s, t, x, r );

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

__global__ void 
magma_zbicgmerge4_kernel_1(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp = skp[0];
        skp[0] = skp[4]/tmp;
    }
}

__global__ void 
magma_zbicgmerge4_kernel_2(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        skp[2] = skp[6]/skp[7];
        skp[3] = skp[4];
    }
}

__global__ void 
magma_zbicgmerge4_kernel_3(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp1 = skp[4]/skp[3];
        magmaDoubleComplex tmp2 = skp[0] / skp[2];
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
    skp         magmaDoubleComplex*
                vector with parameters


    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" int
magma_zbicgmerge4(  int type, 
                    magmaDoubleComplex *skp ){

    dim3 Bs( 1 );
    dim3 Gs( 1 );
    if( type == 1 )
        magma_zbicgmerge4_kernel_1<<<Gs, Bs, 0>>>( skp );
    else if( type == 2 )
        magma_zbicgmerge4_kernel_2<<<Gs, Bs, 0>>>( skp );
    else if( type == 3 )
        magma_zbicgmerge4_kernel_3<<<Gs, Bs, 0>>>( skp );
    else
        printf("error: no kernel called\n");

   return MAGMA_SUCCESS;
}

