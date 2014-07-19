/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zlarfx.cu normal z -> s, Fri Jul 18 17:34:12 2014

*/
#include "common_magma.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16


//==============================================================================

__global__
void magma_slarfx_kernel( int m, float *v, float *tau,
                         float *c, int ldc, float *xnorm,
                         float *T, int it )
{
    if ( !MAGMA_S_EQUAL(*tau, MAGMA_S_ZERO) ) {
        const int i = threadIdx.x;
        //float *dc = c + (blockIdx.x-it-1) * ldc;
        float *dc = c + (blockIdx.x) * ldc;

        __shared__ float sum[ BLOCK_SIZE ];
        float lsum;

        /* NOTE HERE C is the C at position C(i, 0) 
         * if blockIdx.x<it it performs the V(i:n,i)' * V(i:n,1:i-1)' used for computing T
         * if blockIdx.x>it it perform  w := v' * C  */
        lsum = MAGMA_S_ZERO;
        for( int j = i; j < m; j += BLOCK_SIZE ){
            if (j==0){
               lsum += MAGMA_S_MUL( MAGMA_S_ONE, dc[j] );
               v[j] = MAGMA_S_ONE;
            }
            else
               lsum += MAGMA_S_MUL( MAGMA_S_CNJG( v[j] ), dc[j] );
        }
        sum[i] = lsum;
        magma_sum_reduce< BLOCK_SIZE >( i, sum );

        /*  C := C - v * w  */
        __syncthreads();
        float z__1 = - MAGMA_S_CNJG(*tau) * sum[0];
        if (blockIdx.x>it){
           for( int j = m-i-1; j>=0 ; j -= BLOCK_SIZE )
                 dc[j] += z__1 * v[j];
           __syncthreads();

           /* Adjust the rest of the column norms */
           if (i==0){
             float temp = MAGMA_S_ABS( dc[0] ) / xnorm[blockIdx.x-it-1];
             temp = (temp + 1.) * (1. - temp);
             xnorm[blockIdx.x-it-1] = xnorm[blockIdx.x-it-1] * sqrt(temp); 
           }
        }
        else
        {
           if (blockIdx.x==it)
              *(T+it) = *tau;
           else
              *(T+blockIdx.x) = MAGMA_S_CNJG(z__1);
        }
    }
    else // Make last column of T zero
    {
       if (blockIdx.x<=it)
          *(T+blockIdx.x) = MAGMA_S_ZERO;
    }

}

//==============================================================================

__global__
void magma_strmv_kernel(const float *T, int ldt, float *t)
{
   const int i = threadIdx.x;
   T += i;

   __shared__ float tlocal[ BLOCK_SIZE ];
   float res = MAGMA_S_MAKE(0., 0.);

   tlocal[i] = t[i];
   __syncthreads();

   #pragma unroll
   for(int j=0; j<blockDim.x; j++)
      res +=  T[j*ldt]*tlocal[j];

   t[i] = res;
}

__global__
void magma_strmv_kernel2(const float *T, int ldt, float *t, 
                         float *y, float *tau)
{
   const int i = threadIdx.x;
   T += blockIdx.x;

   __shared__ float sum[ 128 ];

   sum[i] = T[i*ldt]*t[i];
   magma_sum_reduce_n(blockDim.x, i, sum);

   __syncthreads();

   if (i==0){
      y[blockIdx.x] = sum[0];
      if (blockIdx.x==0)
         y[gridDim.x] = tau[0];
   }
}

//==============================================================================

__global__
void magma_strmv_tkernel(float *T, int ldt, float *t, float *y)
{
   const int i = threadIdx.x;
   T += blockIdx.x*ldt;

   __shared__ float sum[ 128 ];

   sum[i] = MAGMA_S_CNJG(T[i])*t[i];
   magma_sum_reduce_n(blockDim.x, i, sum);

   __syncthreads();

   if (i==0)
      y[blockIdx.x] = sum[0];
}

//==============================================================================

/*
    Apply a real elementary reflector H to a real M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v'
    where tau is a real scalar and v is a real vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H' (the conjugate transpose of H), supply conjg(tau) 
    instead tau.

    The norms of v(:, 1:n) are given as input in xnorm(1:n). On exit, the norms
    are adjusted to hold the norms of v(2:m,2:n). This is a difference with the 
    LAPACK's slarf routine. 
 */
extern "C" void
magma_slarfx_gpu(magma_int_t m, magma_int_t n, float *v, float *tau,
                float *c, magma_int_t ldc, float *xnorm, 
                float *T, magma_int_t i, float *work )
{
    magma_int_t N = n + i + 1;

    if (i==0)
        magma_slarfx_kernel<<< N, BLOCK_SIZE, 0, magma_stream >>>( m, v, tau, c, ldc, xnorm, T+i*N, i);
    else
        magma_slarfx_kernel<<< N, BLOCK_SIZE, 0, magma_stream >>>( m, v, tau, c, ldc, xnorm, work, i);

    if (i > 0){
        //magma_strmv_kernel<<< 1, i, 0, magma_stream >>>( T, N, T+i*N);
        magma_strmv_kernel2<<< i, i, 0, magma_stream  >>>( T, N, work, T+i*N, tau);
    }
}

//==============================================================================
