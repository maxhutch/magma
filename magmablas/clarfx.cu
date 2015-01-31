/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlarfx.cu normal z -> c, Fri Jan 30 19:00:09 2015

*/
#include "common_magma.h"
#include "commonblas_c.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16


//==============================================================================

__global__
void magma_clarfx_kernel( int m, magmaFloatComplex *v, magmaFloatComplex *tau,
                         magmaFloatComplex *c, int ldc, float *xnorm,
                         magmaFloatComplex *T, int it )
{
    if ( !MAGMA_C_EQUAL(*tau, MAGMA_C_ZERO) ) {
        const int tx = threadIdx.x;
        //magmaFloatComplex *dc = c + (blockIdx.x-it-1) * ldc;
        magmaFloatComplex *dc = c + (blockIdx.x) * ldc;

        __shared__ magmaFloatComplex sum[ BLOCK_SIZE ];
        magmaFloatComplex lsum;

        /* NOTE HERE C is the C at position C(i, 0) 
         * if blockIdx.x<it it performs the V(i:n,i)' * V(i:n,1:i-1)' used for computing T
         * if blockIdx.x>it it perform  w := v**H * C  */
        lsum = MAGMA_C_ZERO;
        for( int j = tx; j < m; j += BLOCK_SIZE ){
            if (j==0){
               lsum += MAGMA_C_MUL( MAGMA_C_ONE, dc[j] );
               v[j] = MAGMA_C_ONE;
            }
            else
               lsum += MAGMA_C_MUL( MAGMA_C_CNJG( v[j] ), dc[j] );
        }
        sum[tx] = lsum;
        magma_sum_reduce< BLOCK_SIZE >( tx, sum );

        /*  C := C - v * w  */
        __syncthreads();
        magmaFloatComplex z__1 = - MAGMA_C_CNJG(*tau) * sum[0];
        if (blockIdx.x>it){
           for( int j = m-tx-1; j>=0 ; j -= BLOCK_SIZE )
                 dc[j] += z__1 * v[j];
           __syncthreads();

           /* Adjust the rest of the column norms */
           /*
           if (tx==0){
             float temp = MAGMA_C_ABS( dc[0] ) / xnorm[blockIdx.x-it-1];
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
              *(T+blockIdx.x) = MAGMA_C_CNJG(z__1);
        }
    }
    else if (blockIdx.x<=it)// in case tau is zero put the corresponding column of T to zero
    {
        *(T+blockIdx.x) = MAGMA_C_ZERO;
    }

}

//==============================================================================
extern "C"
__global__
void magma_ctrmv_kernel(const magmaFloatComplex *T, int ldt, magmaFloatComplex *t)
{
   const int tx = threadIdx.x;
   T += tx;

   __shared__ magmaFloatComplex tlocal[ BLOCK_SIZE ];
   magmaFloatComplex res = MAGMA_C_MAKE(0., 0.);

   tlocal[tx] = t[tx];
   __syncthreads();

   #pragma unroll
   for(int j=0; j<blockDim.x; j++)
      res +=  T[j*ldt]*tlocal[j];

   t[tx] = res;
}

extern "C"
__global__
void magma_ctrmv_kernel2(const magmaFloatComplex *T, int ldt, magmaFloatComplex *t, 
                         magmaFloatComplex *y, magmaFloatComplex *tau)
{
   const int tx = threadIdx.x;
   T += blockIdx.x;

   __shared__ magmaFloatComplex sum[ 128 ];

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
void magma_ctrmv_tkernel(magmaFloatComplex *T, int ldt, magmaFloatComplex *t, magmaFloatComplex *y)
{
   const int tx = threadIdx.x;
   T += blockIdx.x*ldt;

   __shared__ magmaFloatComplex sum[ 128 ];

   sum[tx] = MAGMA_C_CNJG(T[tx])*t[tx];
   magma_sum_reduce_n(blockDim.x, tx, sum);

   __syncthreads();

   if (tx==0)
      y[blockIdx.x] = sum[0];
}

//==============================================================================

/*
    Apply a complex elementary reflector H to a complex M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v**H
    where tau is a complex scalar and v is a complex vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H**H (the conjugate transpose of H), supply conjg(tau) 
    instead tau.

    The norms of v(:, 1:n) are given as input in xnorm(1:n). On exit, the norms
    are adjusted to hold the norms of v(2:m,2:n). This is a difference with the 
    LAPACK's clarf routine. 
 */
extern "C" void
magma_clarfx_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr v,
    magmaFloatComplex_ptr tau,
    magmaFloatComplex_ptr C, magma_int_t ldc,
    magmaFloat_ptr        xnorm, 
    magmaFloatComplex_ptr dT, magma_int_t iter,
    magmaFloatComplex_ptr work )
{
    magma_int_t N = n + iter + 1;

    if (iter==0)
        magma_clarfx_kernel<<< N, BLOCK_SIZE, 0, magma_stream >>>( m, v, tau, C, ldc, xnorm, dT+iter*N, iter);
    else
        magma_clarfx_kernel<<< N, BLOCK_SIZE, 0, magma_stream >>>( m, v, tau, C, ldc, xnorm, work, iter);

    if (iter > 0){
        //magma_ctrmv_kernel<<< 1, iter, 0, magma_stream >>>( dT, N, dT+iter*N);
        magma_ctrmv_kernel2<<< iter, iter, 0, magma_stream  >>>( dT, N, work, dT+iter*N, tau);
    }
}

//==============================================================================
