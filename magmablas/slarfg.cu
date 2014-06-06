/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:45 2013
       
       @author Mark Gates
*/
#include "common_magma.h"

#define PRECISION_s

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512


// ----------------------------------------
// Does sum reduction of array x, leaving total in x[0].
// Contents of x are destroyed in the process.
// With k threads, can reduce array up to 2*k in size.
// Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
// Having n as template parameter allows compiler to evaluate some conditions at compile time.
template< int n >
__device__
void sum_reduce( /*int n,*/ int i, float* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce


// ----------------------------------------
// CUDA kernel for magma_slarfg.
// Uses one block of BLOCK_SIZE (currently 512) threads.
// Each thread sums dx[ i + k*BLOCK_SIZE ]^2 for k = 0, 1, ...,
// then does parallel sum reduction to get norm-squared.
// 
//
// Currently setup to use BLOCK_SIZE threads, no matter how small dx is.
// This was slightly faster (5%) than passing n to sum_reduce.
// To use number of threads = min( BLOCK_SIZE, max( 1, n-1 )), pass n as
// argument to sum_reduce, rather than as template parameter.
__global__
void magma_slarfg_kernel( int n, float* dx0, float* dx, int incx, float* dtau )
{
    const int i = threadIdx.x;
    __shared__ float sum[ BLOCK_SIZE ];
    __shared__ float scale;
    
    // get norm of x
    // dx has length n-1
    sum[i] = 0;
    for( int j = i; j < n-1; j += BLOCK_SIZE ) {
        sum[i] += dx[j*incx] * dx[j*incx];
    }
    sum_reduce< BLOCK_SIZE >( i, sum );
    //sum_reduce( blockDim.x, i, sum );
    
    if ( i == 0 ) {
        if ( sum[0] == 0 ) {
            *dtau = 0;
            scale = 0;
        }
        else {
            float alpha = *dx0;
            float beta  = sqrt( alpha*alpha + sum[0] );
            beta  = -copysign( beta, alpha );
            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau = (beta - alpha) / beta;
            *dx0  = beta;
            scale = 1 / (alpha - beta);
        }
    }
    
    // scale x
    __syncthreads();
    if ( scale != 0 ) {
        for( int j = i; j < n-1; j += BLOCK_SIZE ) {
            dx[j*incx] *= scale;
        }
    }
}


// ----------------------------------------
// Generates Householder elementary reflector H = I - tau v v^T to reduce
//   H [ dx0 ] = [ beta ]
//     [ dx  ]   [ 0    ]
// with beta = ±norm( [dx0, dx] ).
// Stores v over dx; first element of v is 1 and is not stored.
// Stores beta over dx0.
// Stores tau.
extern "C"
void magma_slarfg( magma_int_t n, float* dx0, float* dx, magma_int_t incx, float* dtau )
{
    dim3 blocks( 1 );
    dim3 threads( BLOCK_SIZE );
    //dim3 threads( min( BLOCK_SIZE, max( n-1, 1 )));
    magma_slarfg_kernel<<< blocks, threads >>>( n, dx0, dx, incx, dtau );
}
