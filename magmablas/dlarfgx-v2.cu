/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:45 2013

*/
#include "common_magma.h"

// 512 is maximum number of threads for CUDA capability 1.x
//#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 512
//#else
//   #define BLOCK_SIZE 768
//#endif

#define PRECISION_d

__global__ void magma_dgemv_kernel3(int m, const double * __restrict__ V, int ldv, 
                                    double *c, double *dwork,
                                    double *tau);
__global__ void magma_dtrmv_kernel(const double *T, int ldt, double *v);
__global__ void magma_dtrmv_kernel2(const double *T, int ldt, 
                                    double *v, double *y, double *tau);

//==============================================================================

__global__
void magma_dlarfgx_gpu_kernel( int n, double* dx0, double* dx, 
                               double *dtau, double *dxnorm,
                               double *dA, int it)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    __shared__ double scale;
    __shared__ double xnorm;    
  
    double dxi;

    if ( j < n-1)
        dxi = dx[j];
  
    if ( i == 0 ) {
        xnorm = *dxnorm;
        if ( xnorm == 0 ) {
            *dtau = MAGMA_D_ZERO;
        }
        else {

#if (defined(PRECISION_s) || defined(PRECISION_d))
            double alpha = *dx0;

            // no need to compute the norm as it is passed as input
            double beta  = xnorm; // sqrt( alpha*alpha + xnorm*xnorm );
            beta  = -copysign( beta, alpha );
 
            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
               *dtau = (beta - alpha) / beta;
               //*dx0  = 1.;
               *dA   = beta;  
            }

            scale = 1. / (alpha - beta);
#else
            double alpha = *dx0;
            double alphar =  MAGMA_D_REAL(alpha), alphai = MAGMA_D_IMAG(alpha);

            // no need to compute the norm as it is passed as input
            double beta  = xnorm; // sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
               *dtau = MAGMA_D_MAKE((beta - alphar)/beta, -alphai/beta);
               //*dx0  = MAGMA_D_MAKE(  1., 0.);
               *dA   = MAGMA_D_MAKE(beta, 0.);
            }            

            alpha = MAGMA_D_MAKE( MAGMA_D_REAL(alpha) - beta, MAGMA_D_IMAG(alpha));
            scale = MAGMA_D_DIV( MAGMA_D_ONE, alpha);
#endif
        }
    }

    // scale x
    __syncthreads();
    if ( xnorm != 0 && j < n-1)
        dx[j] = MAGMA_D_MUL(dxi, scale);

    if (j<it){
        *( dA-it+j) = *(dx0-it+j);
        *(dx0-it+j) = MAGMA_D_MAKE(0., 0.);
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

   The difference with LAPACK's dlarfg is that the norm of dx, and hance beta,
   are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_dlarfgx_gpu(magma_int_t n, double *dx0, double *dx, 
                  double *dtau, double *dxnorm, 
                  double *dA, magma_int_t it)
{
    dim3 blocks((n+BLOCK_SIZE-1) / BLOCK_SIZE);
    dim3 threads( BLOCK_SIZE );
 
    magma_dlarfgx_gpu_kernel<<< blocks, threads, 0, magma_stream >>>( n, dx0, dx, dtau, dxnorm, dA, it);
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

   The difference with LAPACK's dlarfg is that the norm of dx, and hance beta,
   are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_dlarfgtx_gpu(magma_int_t n, double *dx0, double *dx,
                   double *dtau, double *dxnorm,
                   double *dA, magma_int_t i, 
                   double *V, magma_int_t ldv, double *T, magma_int_t ldt, 
                   double *work)
{
   /*  Generate the elementary reflector H(i)  */
   magma_dlarfgx_gpu(n, dx0, dx, dtau, dxnorm, dA, i);

   if (i==0){
      double tt = MAGMA_D_ONE;
      magmablas_dlacpy(MagmaUpperLower, 1, 1, dtau, 1, T+i+i*ldt, 1);
      magma_dsetmatrix(1,1, &tt,1, dx0,1);
   }
   else
   {
      /* Compute the i-th column of T */      
      magma_dgemv_kernel3<<< i, BLOCK_SIZE, 0, magma_stream >>>(n, V, ldv, dx0, work, dtau);
      magma_dtrmv_kernel2<<< i, i, 0, magma_stream          >>>( T, ldt, work, T+i*ldt, dtau);
   }
}

//==============================================================================

