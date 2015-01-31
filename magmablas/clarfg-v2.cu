/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlarfg-v2.cu normal z -> c, Fri Jan 30 19:00:09 2015

*/
#include "common_magma.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define PRECISION_c


__global__
void magma_clarfg_gpu_kernel( int n, magmaFloatComplex* dx0, magmaFloatComplex* dx,
                              magmaFloatComplex *dtau, float *dxnorm, magmaFloatComplex* dAkk)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    __shared__ magmaFloatComplex scale;
    float xnorm;

    magmaFloatComplex dxi;

#if (defined(PRECISION_s) || defined(PRECISION_d))
    if( n <= 1 ) {
#else
    if( n <= 0 ) {
#endif
        *dtau = MAGMA_C_ZERO;
        *dAkk = *dx0;
        return;
    }

    if ( j < n-1)
        dxi = dx[j];

    xnorm = *dxnorm;
    magmaFloatComplex alpha = *dx0;

#if (defined(PRECISION_s) || defined(PRECISION_d))
    if ( xnorm != 0 ) {
        if (i == 0) {  
            float beta  = sqrt( alpha*alpha + xnorm*xnorm );
            beta  = -copysign( beta, alpha );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau = (beta - alpha) / beta;
            *dAkk  = beta;

            scale = 1. / (alpha - beta);
        }
#else
    float alphar = MAGMA_C_REAL(alpha);
    float alphai = MAGMA_C_IMAG(alpha);
    if ( xnorm != 0 || alphai != 0) {
        if (i == 0) {
            float beta  = sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau = MAGMA_C_MAKE((beta - alphar)/beta, -alphai/beta);
            *dAkk = MAGMA_C_MAKE(beta, 0.);

            alpha = MAGMA_C_MAKE( MAGMA_C_REAL(alpha) - beta, MAGMA_C_IMAG(alpha));
            scale = MAGMA_C_DIV( MAGMA_C_ONE, alpha);
        }
#endif

        // scale x
        __syncthreads();
        if ( xnorm != 0 && j < n-1)
            dx[j] = MAGMA_C_MUL(dxi, scale);

    } else {
        *dtau = MAGMA_C_ZERO;
        *dAkk = *dx0; 
    }
}


/*
    Generates Householder elementary reflector H = I - tau v v^T to reduce
        H [ dx0 ] = [ beta ]
          [ dx  ]   [ 0    ]
    with beta = ±norm( [dx0, dx] ) = ±dxnorm[0].
    Stores v over dx; first element of v is 1 and is not stored.
    Stores beta over dx0.
    Stores tau.  
    
    The difference with LAPACK's clarfg is that the norm of dx, and hence beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_clarfg_gpu(
    magma_int_t n,
    magmaFloatComplex_ptr dx0,
    magmaFloatComplex_ptr dx,
    magmaFloatComplex_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloatComplex_ptr dAkk)
{
    dim3 blocks((n+BLOCK_SIZE-1) / BLOCK_SIZE);
    dim3 threads( BLOCK_SIZE );

    /* recomputing the norm */
    //magmablas_scnrm2_cols(n, 1, dx0, n, dxnorm);
    magmablas_scnrm2_cols(n-1, 1, dx0+1, n, dxnorm);

    magma_clarfg_gpu_kernel<<< blocks, threads,
                               0, magma_stream >>>(n, dx0, dx, dtau, dxnorm, dAkk);
}
