/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlarfx.cu normal z -> s, Fri Jan 30 19:00:08 2015

*/
#include "common_magma.h"
#include "commonblas_s.h"
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
        const int tx = threadIdx.x;
        //float *dc = c + (blockIdx.x-it-1) * ldc;
        float *dc = c + (blockIdx.x) * ldc;

        __shared__ float sum[ BLOCK_SIZE ];
        float lsum;

        /* NOTE HERE C is the C at position C(i, 0) 
         * if blockIdx.x<it it performs the V(i:n,i)' * V(i:n,1:i-1)' used for computing T
         * if blockIdx.x>it it perform  w := v**H * C  */
        lsum = MAGMA_S_ZERO;
        for( int j = tx; j < m; j += BLOCK_SIZE ){
            if (j==0){
               lsum += MAGMA_S_MUL( MAGMA_S_ONE, dc[j] );
               v[j] = MAGMA_S_ONE;
            }
            else
               lsum += MAGMA_S_MUL( MAGMA_S_CNJG( v[j] ), dc[j] );
        }
        sum[tx] = lsum;
        magma_sum_reduce< BLOCK_SIZE >( tx, sum );

        /*  C := C - v * w  */
        __syncthreads();
        float z__1 = - MAGMA_S_CNJG(*tau) * sum[0];
        if (blockIdx.x>it){
           for( int j = m-tx-1; j>=0 ; j -= BLOCK_SIZE )
                 dc[j] += z__1 * v[j];
           __syncthreads();

           /* Adjust the rest of the column norms */
           /*
           if (tx==0){
             float temp = MAGMA_S_ABS( dc[0] ) / xnorm[blockIdx.x-it-1];
             temp = (temp + 1.) * (1. - temp);
             xnorm[blockIdx.x-it-1] = xnorm[blockIdx.x-it-1] * sqrt(temp); 
           }
           */
        }
        else
        {
           if (blockIdx.x==it)
              *(T+it) = *tau;
           else
              *(T+blockIdx.x) = MAGMA_S_CNJG(z__1);
        }
    }
    else if (blockIdx.x<=it)// in case tau is zero put the corresponding column of T to zero
    {
        *(T+blockIdx.x) = MAGMA_S_ZERO;
    }

}

//==============================================================================
extern "C"
__global__
void magma_strmv_kernel(const float *T, int ldt, float *t)
{
   const int tx = threadIdx.x;
   T += tx;

   __shared__ float tlocal[ BLOCK_SIZE ];
   float res = MAGMA_S_MAKE(0., 0.);

   tlocal[tx] = t[tx];
   __syncthreads();

   #pragma unroll
   for(int j=0; j<blockDim.x; j++)
      res +=  T[j*ldt]*tlocal[j];

   t[tx] = res;
}

extern "C"
__global__
void magma_strmv_kernel2(const float *T, int ldt, float *t, 
                         float *y, float *tau)
{
   const int tx = threadIdx.x;
   T += blockIdx.x;

   __shared__ float sum[ 128 ];

   sum[tx] = T[tx*ldt]*t[tx];
   magma_sum_reduce_n(blockDim.x, tx, sum);

   __syncthreads();

   if (tx==0){
      y[blockIdx.x] = sum[0];
      if (blockIdx.x==0)
         y[gridDim.x] = tau[0];
   }
}

//==============================================================================
extern "C"
__global__
void magma_strmv_tkernel(float *T, int ldt, float *t, float *y)
{
   const int tx = threadIdx.x;
   T += blockIdx.x*ldt;

   __shared__ float sum[ 128 ];

   sum[tx] = MAGMA_S_CNJG(T[tx])*t[tx];
   magma_sum_reduce_n(blockDim.x, tx, sum);

   __syncthreads();

   if (tx==0)
      y[blockIdx.x] = sum[0];
}

//==============================================================================

/*
    Apply a real elementary reflector H to a real M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v**H
    where tau is a real scalar and v is a real vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H**H (the conjugate transpose of H), supply conjg(tau) 
    instead tau.

    The norms of v(:, 1:n) are given as input in xnorm(1:n). On exit, the norms
    are adjusted to hold the norms of v(2:m,2:n). This is a difference with the 
    LAPACK's slarf routine. 
 */
extern "C" void
magma_slarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr v,
    magmaFloat_ptr tau,
    magmaFloat_ptr C, magma_int_t ldc,
    magmaFloat_ptr        xnorm, 
    magmaFloat_ptr dT, magma_int_t iter,
    magmaFloat_ptr work )
{
    magma_int_t N = n + iter + 1;

    if (iter==0)
        magma_slarfx_kernel<<< N, BLOCK_SIZE, 0, magma_stream >>>( m, v, tau, C, ldc, xnorm, dT+iter*N, iter);
    else
        magma_slarfx_kernel<<< N, BLOCK_SIZE, 0, magma_stream >>>( m, v, tau, C, ldc, xnorm, work, iter);

    if (iter > 0){
        //magma_strmv_kernel<<< 1, iter, 0, magma_stream >>>( dT, N, dT+iter*N);
        magma_strmv_kernel2<<< iter, iter, 0, magma_stream  >>>( dT, N, work, dT+iter*N, tau);
    }
}

//==============================================================================
