/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/zlarfg-v2.cu normal z -> d, Mon May  2 23:30:32 2016

*/
#include "magma_internal.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define REAL


__global__
void magma_dlarfg_gpu_kernel( int n, double* dx0, double* dx,
                              double *dtau, double *dxnorm, double* dAkk)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    __shared__ double scale;
    double xnorm;

    double dxi;

#ifdef REAL
    if ( n <= 1 )
#else
    if ( n <= 0 )
#endif
    {
        *dtau = MAGMA_D_ZERO;
        *dAkk = *dx0;
        return;
    }

    if ( j < n-1)
        dxi = dx[j];

    xnorm = *dxnorm;
    double alpha = *dx0;

#ifdef REAL
    if ( xnorm != 0 ) {
        if (i == 0) {  
            double beta  = sqrt( alpha*alpha + xnorm*xnorm );
            beta  = -copysign( beta, alpha );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau = (beta - alpha) / beta;
            *dAkk  = beta;

            scale = 1. / (alpha - beta);
        }
#else
    double alphar = MAGMA_D_REAL(alpha);
    double alphai = MAGMA_D_IMAG(alpha);
    if ( xnorm != 0 || alphai != 0) {
        if (i == 0) {
            double beta  = sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            *dtau = MAGMA_D_MAKE((beta - alphar)/beta, -alphai/beta);
            *dAkk = MAGMA_D_MAKE(beta, 0.);

            alpha = MAGMA_D_MAKE( MAGMA_D_REAL(alpha) - beta, MAGMA_D_IMAG(alpha));
            scale = MAGMA_D_DIV( MAGMA_D_ONE, alpha);
        }
#endif

        // scale x
        __syncthreads();
        if ( xnorm != 0 && j < n-1)
            dx[j] = MAGMA_D_MUL(dxi, scale);
    }
    else {
        *dtau = MAGMA_D_ZERO;
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
    
    The difference with LAPACK's dlarfg is that the norm of dx, and hence beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_dlarfg_gpu_q(
    magma_int_t n,
    magmaDouble_ptr dx0,
    magmaDouble_ptr dx,
    magmaDouble_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDouble_ptr dAkk,
    magma_queue_t queue )
{
    dim3 blocks( magma_ceildiv( n, BLOCK_SIZE ) );
    dim3 threads( BLOCK_SIZE );

    /* recomputing the norm */
    //magmablas_dnrm2_cols(n, 1, dx0, n, dxnorm);
    magmablas_dnrm2_cols_q(n-1, 1, dx0+1, n, dxnorm, queue);

    magma_dlarfg_gpu_kernel
        <<< blocks, threads, 0, queue->cuda_stream() >>>
        (n, dx0, dx, dtau, dxnorm, dAkk);
}
