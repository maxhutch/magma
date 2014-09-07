/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "commonblas_z.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define PRECISION_z


//==============================================================================

__global__
void magma_zlarfgx_gpu_kernel( int n, magmaDoubleComplex* dx0, magmaDoubleComplex* dx, 
                               magmaDoubleComplex *dtau, double *dxnorm,
                               magmaDoubleComplex *dA, int it)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    __shared__ magmaDoubleComplex scale;
    __shared__ double xnorm;    
  
    magmaDoubleComplex dxi;

    if ( j < n-1 )
        dxi = dx[j];
  
    if ( i == 0 ) {
         xnorm = *dxnorm;
#if (defined(PRECISION_s) || defined(PRECISION_d))
        double alpha = *dx0;
        double alphai = MAGMA_Z_ZERO;
        if ( (xnorm == 0 && alphai == MAGMA_Z_ZERO ) || n == 1 )
#else
        magmaDoubleComplex alpha = *dx0;
        double alphar =  MAGMA_Z_REAL(alpha), alphai = MAGMA_Z_IMAG(alpha);
        if ( (xnorm == 0 && alphai == MAGMA_Z_ZERO ) || n == 0 )
#endif
        {
            *dtau = MAGMA_Z_ZERO;
            *dA   = *dx0;
        }
        else {

#if (defined(PRECISION_s) || defined(PRECISION_d))
            // no need to compute the norm as it is passed as input
            double beta  = xnorm; // sqrt( alpha*alpha + xnorm*xnorm );
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
            double beta  = xnorm; // sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
            beta  = -copysign( beta, alphar );

            // todo: deal with badly scaled vectors (see lapack's larfg)
            if (j==0){
                *dtau = MAGMA_Z_MAKE((beta - alphar)/beta, -alphai/beta);
                //*dx0  = MAGMA_Z_MAKE(  1., 0.); //cannot be done here because raise condition all threadblock need to read it for alpha
                *dA   = MAGMA_Z_MAKE(beta, 0.);
            }            

            alpha = MAGMA_Z_MAKE( MAGMA_Z_REAL(alpha) - beta, MAGMA_Z_IMAG(alpha));
            scale = MAGMA_Z_DIV( MAGMA_Z_ONE, alpha);
#endif
        }
    }

    // scale x
    __syncthreads();
    if ( xnorm != 0 && j < n-1)
        dx[j] = MAGMA_Z_MUL(dxi, scale);

    if (j<it){
        *( dA-it+j) = *(dx0-it+j);
        *(dx0-it+j) = MAGMA_Z_MAKE(0., 0.);
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
    
    The difference with LAPACK's zlarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_zlarfgx_gpu(magma_int_t n, magmaDoubleComplex *dx0, magmaDoubleComplex *dx, 
                  magmaDoubleComplex *dtau, double *dxnorm, 
                  magmaDoubleComplex *dA, magma_int_t it)
{
    dim3 blocks((n+BLOCK_SIZE-1) / BLOCK_SIZE);
    dim3 threads( BLOCK_SIZE );
 
    magma_zlarfgx_gpu_kernel<<< blocks, threads, 0, magma_stream >>>( n, dx0, dx, dtau, dxnorm, dA, it);
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
    
    The difference with LAPACK's zlarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*/
extern "C" void
magma_zlarfgtx_gpu(magma_int_t n, magmaDoubleComplex *dx0, magmaDoubleComplex *dx,
                   magmaDoubleComplex *dtau, double *dxnorm,
                   magmaDoubleComplex *dA, magma_int_t i, 
                   magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *T, magma_int_t ldt, 
                   magmaDoubleComplex *work)
{
    /*  Generate the elementary reflector H(i)  */
    magma_zlarfgx_gpu(n, dx0, dx, dtau, dxnorm, dA, i);
    
    if (i==0) {
        magmaDoubleComplex tt = MAGMA_Z_ONE;
        magmablas_zlacpy(MagmaUpperLower, 1, 1, dtau, 1, T+i+i*ldt, 1);
        magma_zsetmatrix(1,1, &tt,1, dx0,1);
    }
    else {
        /* Compute the i-th column of T */      
        magma_zgemv_kernel3<<< i, BLOCK_SIZE, 0, magma_stream >>>(n, V, ldv, dx0, work, dtau);
        magma_ztrmv_kernel2<<< i, i, 0, magma_stream          >>>( T, ldt, work, T+i*ldt, dtau);
    }
}

//==============================================================================
