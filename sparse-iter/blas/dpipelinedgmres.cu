/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from zpipelinedgmres.cu normal z -> d, Sun May  3 11:22:58 2015
       @author Hartwig Anzt

*/
#include "common_magma.h"

#define REAL

#define BLOCK_SIZE 512


template< int n >
__device__ void sum_reduce( /*int n,*/ int i, double* x )
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
magma_dpipelined_correction( 
    int n,  
    int k,
    double * skp, 
    double * r,
    double * v )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double zz= 0.0, tmp= 0.0;

    extern __shared__ double temp[];    
    
    temp[ i ] = ( i < k ) ? skp[ i ] * skp[ i ] : MAGMA_D_MAKE( 0.0, 0.0);
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
        tmp = MAGMA_D_REAL( temp[ i ] );
        zz = MAGMA_D_REAL( skp[(k)] );
        skp[k] = MAGMA_D_MAKE( sqrt(zz-tmp),0.0 );
    }
}

__global__ void
magma_dpipelined_copyscale( 
    int n,  
    int k,
    double * skp, 
    double * r,
    double * v )
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double rr=skp[k];

    if( i<n ){
        v[i] =  r[i] * 1.0 / rr;

    }
}

//----------------------------------------------------------------------------//

__global__ void
magma_dpipelineddnrm2_kernel( 
    int m, 
    double * da, 
    int ldda, 
    double * dxnorm )
{
    const int i = threadIdx.x;
    magmaDouble_ptr dx = da + blockIdx.x * ldda;

    __shared__ double sum[ 512 ];
    double re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = i; j < m; j += 512 ) {
        #ifdef REAL
            re = dx[j];
            lsum += re*re;
        #else
            re = MAGMA_D_REAL( dx[j] );
            double im = MAGMA_D_IMAG( dx[j] );
            lsum += re*re + im*im;
        #endif
    }
    sum[i] = lsum;
    sum_reduce< 512 >( i, sum );

    if (i==0)
        dxnorm[blockIdx.x] = MAGMA_D_MAKE( sqrt(sum[0]), 0.0 );
}

//----------------------------------------------------------------------------//

__global__ void
magma_dpipelinedscale( 
    int n, 
    double * r, 
    double * drnorm )
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
    r           magmaDouble_ptr 
                vector of length n

    @param[in]
    v           magmaDouble_ptr 
                vector of length n
                
    @param[in]  
    skp         magmaDouble_ptr 
                array of parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dcopyscale(
    int n, 
    int k,
    magmaDouble_ptr r, 
    magmaDouble_ptr v,
    magmaDouble_ptr skp,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( k, BLOCK_SIZE ) );
    unsigned int Ms =   Bs.x * sizeof( double ); 

    dim3 Gs2( magma_ceildiv( n, BLOCK_SIZE ) );


    magma_dpipelined_correction<<<Gs, Bs, Ms, queue >>>
                                            ( n, k, skp, r, v );
    magma_dpipelined_copyscale<<<Gs2, Bs, 0, queue >>>
                                            ( n, k, skp, r, v );

    return MAGMA_SUCCESS;
}


extern "C" magma_int_t
magma_dnrm2scale(
    int m, 
    magmaDouble_ptr r, 
    int lddr, 
    magmaDouble_ptr drnorm,
    magma_queue_t queue )
{
    dim3  blocks( 1 );
    dim3 threads( 512 );
    magma_dpipelineddnrm2_kernel<<< blocks, threads, 0, queue >>>
                                ( m, r, lddr, drnorm );

    dim3 Bs( BLOCK_SIZE );
    dim3 Gs2( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_dpipelinedscale<<<Gs2, Bs, 0, queue >>>( m, r, drnorm );

    return MAGMA_SUCCESS;
}

