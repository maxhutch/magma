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

#define PRECISION_s

__global__ void magma_sgemv_kernel3(int m, const float * __restrict__ V, int ldv, 
                                    float *c, float *dwork,
                                    float *tau);
__global__ void magma_strmv_kernel(const float *T, int ldt, float *v);
__global__ void magma_strmv_kernel2(const float *T, int ldt, 
                                    float *v, float *y, float *tau);

//==============================================================================

__global__
void magma_slarfgx_gpu_kernel( int n, float* dx0, float* dx, 
                               float *dtau, float *dxnorm,
                               float *dA, int it)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    __shared__ float scale;
    __shared__ float xnorm;    
  
    float dxi;

    if ( j < n-1)
        dxi = dx[j];
  
    if ( i == 0 ) {
        xnorm = *dxnorm;
        if ( xnorm == 0 ) {
            *dtau = MAGMA_S_ZERO;
        }
        else {

#if (defined(PRECISION_s) || defined(PRECISION_d))
            float alpha = *dx0;

            // no need to compute the norm as it is passed as input
            float beta  = xnorm; // sqrt( alpha*alpha + xnorm*xnorm );
            beta  = -copysign( beta, alpha );
 
            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
               *dtau = (beta - alpha) / beta;
               //*dx0  = 1.;
               *dA   = beta;  
            }

            scale = 1. / (alpha - beta);
#else
            float alpha = *dx0;
            float alphar =  MAGMA_S_REAL(alpha), alphai = MAGMA_S_IMAG(alpha);

            // no need to compute the norm as it is passed as input
            float beta  = xnorm; // sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
               *dtau = MAGMA_S_MAKE((beta - alphar)/beta, -alphai/beta);
               //*dx0  = MAGMA_S_MAKE(  1., 0.);
               *dA   = MAGMA_S_MAKE(beta, 0.);
            }            

            alpha = MAGMA_S_MAKE( MAGMA_S_REAL(alpha) - beta, MAGMA_S_IMAG(alpha));
            scale = MAGMA_S_DIV( MAGMA_S_ONE, alpha);
#endif
        }
    }

    // scale x
    __syncthreads();
    if ( xnorm != 0 && j < n-1)
        dx[j] = MAGMA_S_MUL(dxi, scale);

    if (j<it){
        *( dA-it+j) = *(dx0-it+j);
        *(dx0-it+j) = MAGMA_S_MAKE(0., 0.);
    } 
}

//==============================================================================

/*
   Generates Householder elementary reflector H = I - tau v v^T to reduce
     H [ dx0 ] = [ beta ]
       [ dx  ]   [ 0    ]
   with beta = ±norm( [dx0, dx] ) = ±dxnorm[0].
   Stores v over dx; first element of v is 1 and is not stored.
   Stores beta over dx0.
   Stores tau.  

   The difference with LAPACK's slarfg is that the norm of dx, and hance beta,
   are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_slarfgx_gpu(magma_int_t n, float *dx0, float *dx, 
                  float *dtau, float *dxnorm, 
                  float *dA, magma_int_t it)
{
    dim3 blocks((n+BLOCK_SIZE-1) / BLOCK_SIZE);
    dim3 threads( BLOCK_SIZE );
 
    magma_slarfgx_gpu_kernel<<< blocks, threads, 0, magma_stream >>>( n, dx0, dx, dtau, dxnorm, dA, it);
}

//==============================================================================

/*
   Generates Householder elementary reflector H = I - tau v v^T to reduce
     H [ dx0 ] = [ beta ]
       [ dx  ]   [ 0    ]
   with beta = ±norm( [dx0, dx] ) = ±dxnorm[0].
   Stores v over dx; first element of v is 1 and is not stored.
   Stores beta over dx0.
   Stores tau.

   The difference with LAPACK's slarfg is that the norm of dx, and hance beta,
   are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_slarfgtx_gpu(magma_int_t n, float *dx0, float *dx,
                   float *dtau, float *dxnorm,
                   float *dA, magma_int_t i, 
                   float *V, magma_int_t ldv, float *T, magma_int_t ldt, 
                   float *work)
{
   /*  Generate the elementary reflector H(i)  */
   magma_slarfgx_gpu(n, dx0, dx, dtau, dxnorm, dA, i);

   if (i==0){
      float tt = MAGMA_S_ONE;
      magmablas_slacpy(MagmaUpperLower, 1, 1, dtau, 1, T+i+i*ldt, 1);
      magma_ssetmatrix(1,1, &tt,1, dx0,1);
   }
   else
   {
      /* Compute the i-th column of T */      
      magma_sgemv_kernel3<<< i, BLOCK_SIZE, 0, magma_stream >>>(n, V, ldv, dx0, work, dtau);
      magma_strmv_kernel2<<< i, i, 0, magma_stream          >>>( T, ldt, work, T+i*ldt, dtau);
   }
}

//==============================================================================

