/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlarfg-v2.cu normal z -> s, Fri Jan 30 19:00:09 2015

*/
#include "common_magma.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define PRECISION_s


__global__
void magma_slarfg_gpu_kernel( int n, float* dx0, float* dx,
                              float *dtau, float *dxnorm, float* dAkk)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    __shared__ float scale;
    float xnorm;

    float dxi;

#if (defined(PRECISION_s) || defined(PRECISION_d))
    if( n <= 1 ) {
#else
    if( n <= 0 ) {
#endif
        *dtau = MAGMA_S_ZERO;
        *dAkk = *dx0;
        return;
    }

    if ( j < n-1)
        dxi = dx[j];

    xnorm = *dxnorm;
    float alpha = *dx0;

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
    float alphar = MAGMA_S_REAL(alpha);
    float alphai = MAGMA_S_IMAG(alpha);
    if ( xnorm != 0 || alphai != 0) {
        if (i == 0) {
            float beta  = sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau = MAGMA_S_MAKE((beta - alphar)/beta, -alphai/beta);
            *dAkk = MAGMA_S_MAKE(beta, 0.);

            alpha = MAGMA_S_MAKE( MAGMA_S_REAL(alpha) - beta, MAGMA_S_IMAG(alpha));
            scale = MAGMA_S_DIV( MAGMA_S_ONE, alpha);
        }
#endif

        // scale x
        __syncthreads();
        if ( xnorm != 0 && j < n-1)
            dx[j] = MAGMA_S_MUL(dxi, scale);

    } else {
        *dtau = MAGMA_S_ZERO;
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
    
    The difference with LAPACK's slarfg is that the norm of dx, and hence beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_slarfg_gpu(
    magma_int_t n,
    magmaFloat_ptr dx0,
    magmaFloat_ptr dx,
    magmaFloat_ptr dtau,
    magmaFloat_ptr        dxnorm,
    magmaFloat_ptr dAkk)
{
    dim3 blocks((n+BLOCK_SIZE-1) / BLOCK_SIZE);
    dim3 threads( BLOCK_SIZE );

    /* recomputing the norm */
    //magmablas_snrm2_cols(n, 1, dx0, n, dxnorm);
    magmablas_snrm2_cols(n-1, 1, dx0+1, n, dxnorm);

    magma_slarfg_gpu_kernel<<< blocks, threads,
                               0, magma_stream >>>(n, dx0, dx, dtau, dxnorm, dAkk);
}
