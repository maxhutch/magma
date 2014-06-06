/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:45 2013

*/
#include "common_magma.h"

// 512 is maximum number of threads for CUDA capability 1.x
//#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 512
//#else
//   #define BLOCK_SIZE 768
//#endif


__global__ void magma_strmv_tkernel(float *T, int ldt, float *v, 
                                    float *y);

// ----------------------------------------
// Does sum reduction of array x, leaving total in x[0].
// Contents of x are destroyed in the process.
// With k threads, can reduce array up to 2*k in size.
// Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
// Having n as template parameter allows compiler to evaluate some conditions at compile time.
template< int n >
__device__ void zsum_reduce( /*int n,*/ int i, float* x )
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

//==============================================================================

__global__ void 
magma_sgemv_kernel1(int m, const float * __restrict__ V, int ldv, 
                    const float * __restrict__ c, 
                    float *dwork)
{
        const int i = threadIdx.x;
        const float *dV = V + (blockIdx.x) * ldv;

        __shared__ float sum[ BLOCK_SIZE ];
        float lsum;

        /*  lsum := v' * C  */
        lsum = MAGMA_S_ZERO;
        for( int j = i; j < m; j += BLOCK_SIZE )
           lsum += MAGMA_S_MUL( MAGMA_S_CNJG( dV[j] ), c[j] );
        
        sum[i] = lsum;
        zsum_reduce< BLOCK_SIZE >( i, sum );

        __syncthreads();
        if (i==0)
           dwork [blockIdx.x] = sum[0];
}

//==============================================================================

__global__ void
magma_sgemv_kernel3(int m, const float * __restrict__ V, int ldv, float *c,
                    float *dwork, float *tau)
{
        const int i = threadIdx.x;
        const float *dV = V + (blockIdx.x) * ldv;

        __shared__ float sum[ BLOCK_SIZE ];
        float lsum;

        if (i==0)
           c[0] = MAGMA_S_ONE;           

        /*  lsum := v' * C  */
        lsum = MAGMA_S_ZERO;
        for( int j = i; j < m; j += BLOCK_SIZE )
           lsum += MAGMA_S_MUL( MAGMA_S_CNJG( dV[j] ), c[j] );

        sum[i] = lsum;
        zsum_reduce< BLOCK_SIZE >( i, sum );

        __syncthreads();
        if (i==0)
           dwork [blockIdx.x] = -tau[0]*sum[0];
}

//==============================================================================

__global__ void
magma_sgemv_kernel2(int m, int n, const float * __restrict__ V, int ldv, 
                    const float * __restrict__ x, float *c)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    float lsum;

    V += j;

    lsum = MAGMA_S_ZERO;
    if (j < m){
       for(int k=0; k<n; k++)
          lsum += MAGMA_S_MUL( V[k*ldv], x[k]);
       
       c[j] -= lsum;
    }
}

//==============================================================================

/*
    Apply a real block reflector H to a real vector C from the left
    (i.e., C = H C). H is represented in the form
          H = I - V T V'
    where T is the real k-by-k upper triangular matrix in the 
    representation of the block reflector, and V is a real block of
    k elementary reflectors. 
*/
extern "C" void
magma_slarfbx_gpu(magma_int_t m, magma_int_t k, float *V, magma_int_t ldv,
                  float *T, magma_int_t ldt, float *c,
                  float *dwork)
{
    /* dwork = V' c                   */
    magma_sgemv_kernel1<<< k, BLOCK_SIZE, 0, magma_stream >>>(m, V, ldv, c, dwork); 

    /* dwork = T' dwork               */
    magma_strmv_tkernel<<< k, k, 0, magma_stream >>>( T, ldt, dwork, dwork+k);
 
    /* c = c - V dwork                */
    dim3  blocks3( (m + BLOCK_SIZE-1) / BLOCK_SIZE );
    dim3 threads3( BLOCK_SIZE );     
    magma_sgemv_kernel2<<< blocks3, threads3, 0, magma_stream >>>( m, k, V, ldv, dwork+k, c);
}

//==============================================================================
