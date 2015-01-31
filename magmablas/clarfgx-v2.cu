/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlarfgx-v2.cu normal z -> c, Fri Jan 30 19:00:09 2015

*/
#include "common_magma.h"
#include "commonblas_c.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define PRECISION_c


//==============================================================================

__global__
void magma_clarfgx_gpu_kernel( int n, magmaFloatComplex* dx0, magmaFloatComplex* dx,
                               magmaFloatComplex *dtau, float *dxnorm,
                               magmaFloatComplex *dA, int it)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    __shared__ magmaFloatComplex scale;
    __shared__ float xnorm;
  
    magmaFloatComplex dxi;

    if ( j < n-1 )
        dxi = dx[j];
  
    if ( i == 0 ) {
        xnorm = *dxnorm;
#if (defined(PRECISION_s) || defined(PRECISION_d))
        float alpha = *dx0;
        float alphai = MAGMA_C_ZERO;
        if ( (xnorm == 0 && alphai == MAGMA_C_ZERO ) || n == 1 )
#else
        magmaFloatComplex alpha = *dx0;
        float alphar =  MAGMA_C_REAL(alpha), alphai = MAGMA_C_IMAG(alpha);
        if ( (xnorm == 0 && alphai == MAGMA_C_ZERO ) || n == 0 )
#endif
        {
            *dtau = MAGMA_C_ZERO;
            *dA   = *dx0;
        }
        else {

#if (defined(PRECISION_s) || defined(PRECISION_d))
            // no need to compute the norm as it is passed as input
            float beta  = xnorm; // sqrt( alpha*alpha + xnorm*xnorm );
            beta  = -copysign( beta, alpha );
 
            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
                *dtau = (beta - alpha) / beta;
                //*dx0  = 1.; //cannot be done here because raise condition all threadblock need to read it for alpha
                *dA   = beta;
            }

            scale = 1. / (alpha - beta);
#else
            // no need to compute the norm as it is passed as input
            float beta  = xnorm; // sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
                *dtau = MAGMA_C_MAKE((beta - alphar)/beta, -alphai/beta);
                //*dx0  = MAGMA_C_MAKE(  1., 0.); //cannot be done here because raise condition all threadblock need to read it for alpha
                *dA   = MAGMA_C_MAKE(beta, 0.);
            }

            alpha = MAGMA_C_MAKE( MAGMA_C_REAL(alpha) - beta, MAGMA_C_IMAG(alpha));
            scale = MAGMA_C_DIV( MAGMA_C_ONE, alpha);
#endif
        }
    }

    // scale x
    __syncthreads();
    if ( xnorm != 0 && j < n-1)
        dx[j] = MAGMA_C_MUL(dxi, scale);

    if (j<it){
        *( dA-it+j) = *(dx0-it+j);
        *(dx0-it+j) = MAGMA_C_MAKE(0., 0.);
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
    
    The difference with LAPACK's clarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_clarfgx_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter)
{
    dim3 blocks((n+BLOCK_SIZE-1) / BLOCK_SIZE);
    dim3 threads( BLOCK_SIZE );
 
    magma_clarfgx_gpu_kernel<<< blocks, threads, 0, magma_stream >>>( n, dx0, dx, dtau, dxnorm, dA, iter);
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
    
    The difference with LAPACK's clarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_clarfgtx_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dA, magma_int_t iter,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr T,  magma_int_t ldt,
    magmaFloatComplex_ptr dwork)
{
    /*  Generate the elementary reflector H(iter)  */
    magma_clarfgx_gpu(n, dx0, dx, dtau, dxnorm, dA, iter);
    
    if (iter==0) {
        magmaFloatComplex tt = MAGMA_C_ONE;
        magmablas_clacpy(MagmaUpperLower, 1, 1, dtau, 1, T+iter+iter*ldt, 1);
        magma_csetmatrix(1,1, &tt,1, dx0,1);
    }
    else {
        /* Compute the iter-th column of T */
        magma_cgemv_kernel3<<< iter, BLOCK_SIZE, 0, magma_stream >>>( n, V, ldv, dx0, dwork, dtau );
        magma_ctrmv_kernel2<<< iter, iter,       0, magma_stream >>>( T, ldt, dwork, T+iter*ldt, dtau );
    }
}

//==============================================================================
