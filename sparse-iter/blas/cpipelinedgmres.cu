/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from zpipelinedgmres.cu normal z -> c, Sun May  3 11:22:58 2015
       @author Hartwig Anzt

*/
#include "common_magma.h"

#define COMPLEX

#define BLOCK_SIZE 512


template< int n >
__device__ void sum_reduce( /*int n,*/ int i, float* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  
        __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  
        __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  
        __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  
        __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  
        __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  
        __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  
        __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  
        __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  
        __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  
        __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  
        __syncthreads(); }
}

__global__ void
magma_cpipelined_correction( 
    int n,  
    int k,
    magmaFloatComplex * skp, 
    magmaFloatComplex * r,
    magmaFloatComplex * v )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float zz= 0.0, tmp= 0.0;

    extern __shared__ magmaFloatComplex temp[];    
    
    temp[ i ] = ( i < k ) ? skp[ i ] * skp[ i ] : MAGMA_C_MAKE( 0.0, 0.0);
    __syncthreads();
     if (i < 64) { temp[ i ] += temp[ i + 64 ]; } __syncthreads(); 
     if( i < 32 ){
        temp[ i ] += temp[ i + 32 ];__syncthreads();    
        temp[ i ] += temp[ i + 16 ];__syncthreads(); 
        temp[ i ] += temp[ i +  8 ];__syncthreads(); 
        temp[ i ] += temp[ i +  4 ];__syncthreads(); 
        temp[ i ] += temp[ i +  2 ];__syncthreads(); 
        temp[ i ] += temp[ i +  1 ];__syncthreads();      
    }
    if( i == 0 ){
        tmp = MAGMA_C_REAL( temp[ i ] );
        zz = MAGMA_C_REAL( skp[(k)] );
        skp[k] = MAGMA_C_MAKE( sqrt(zz-tmp),0.0 );
    }
}

__global__ void
magma_cpipelined_copyscale( 
    int n,  
    int k,
    magmaFloatComplex * skp, 
    magmaFloatComplex * r,
    magmaFloatComplex * v )
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    magmaFloatComplex rr=skp[k];

    if( i<n ){
        v[i] =  r[i] * 1.0 / rr;

    }
}

//----------------------------------------------------------------------------//

__global__ void
magma_cpipelinedscnrm2_kernel( 
    int m, 
    magmaFloatComplex * da, 
    int ldda, 
    magmaFloatComplex * dxnorm )
{
    const int i = threadIdx.x;
    magmaFloatComplex_ptr dx = da + blockIdx.x * ldda;

    __shared__ float sum[ 512 ];
    float re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = i; j < m; j += 512 ) {
        #ifdef REAL
            re = dx[j];
            lsum += re*re;
        #else
            re = MAGMA_C_REAL( dx[j] );
            float im = MAGMA_C_IMAG( dx[j] );
            lsum += re*re + im*im;
        #endif
    }
    sum[i] = lsum;
    sum_reduce< 512 >( i, sum );

    if (i==0)
        dxnorm[blockIdx.x] = MAGMA_C_MAKE( sqrt(sum[0]), 0.0 );
}

//----------------------------------------------------------------------------//

__global__ void
magma_cpipelinesscale( 
    int n, 
    magmaFloatComplex * r, 
    magmaFloatComplex * drnorm )
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i<n ){
        r[i] =  r[i] * 1.0 / drnorm[0];
    }
}

/**
    Purpose
    -------

    Computes the correction term of the pipelined GMRES according to P. Ghysels 
    and scales and copies the new search direction
    
    Returns the vector v = r/ ( skp[k] - (sum_i=1^k skp[i]^2) ) .

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i

    @param[in]
    k           int
                # skp entries v_i^T * r ( without r )

    @param[in]
    r           magmaFloatComplex_ptr 
                vector of length n

    @param[in]
    v           magmaFloatComplex_ptr 
                vector of length n
                
    @param[in]  
    skp         magmaFloatComplex_ptr 
                array of parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_ccopyscale(
    int n, 
    int k,
    magmaFloatComplex_ptr r, 
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr skp,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( k, BLOCK_SIZE ) );
    unsigned int Ms =   Bs.x * sizeof( magmaFloatComplex ); 

    dim3 Gs2( magma_ceildiv( n, BLOCK_SIZE ) );


    magma_cpipelined_correction<<<Gs, Bs, Ms, queue >>>
                                            ( n, k, skp, r, v );
    magma_cpipelined_copyscale<<<Gs2, Bs, 0, queue >>>
                                            ( n, k, skp, r, v );

    return MAGMA_SUCCESS;
}


extern "C" magma_int_t
magma_scnrm2scale(
    int m, 
    magmaFloatComplex_ptr r, 
    int lddr, 
    magmaFloatComplex_ptr drnorm,
    magma_queue_t queue )
{
    dim3  blocks( 1 );
    dim3 threads( 512 );
    magma_cpipelinedscnrm2_kernel<<< blocks, threads, 0, queue >>>
                                ( m, r, lddr, drnorm );

    dim3 Bs( BLOCK_SIZE );
    dim3 Gs2( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_cpipelinesscale<<<Gs2, Bs, 0, queue >>>( m, r, drnorm );

    return MAGMA_SUCCESS;
}

