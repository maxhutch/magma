/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from dznrm2.cu normal z -> d, Fri Jul 18 17:34:12 2014

*/
#include "common_magma.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  512
#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16

#define PRECISION_d


//==============================================================================

__global__ void
magmablas_dnrm2_kernel( int m, double *da, int ldda, double *dxnorm )
{
    const int tx = threadIdx.x;
    double *dx = da + blockIdx.x * ldda;

    __shared__ double sum[ BLOCK_SIZE ];
    double re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = tx; j < m; j += BLOCK_SIZE ) {
        #if (defined(PRECISION_s) || defined(PRECISION_d))
        re = dx[j];
        lsum += re*re;
        #else
        re = MAGMA_D_REAL( dx[j] );
        double im = MAGMA_D_IMAG( dx[j] );
        lsum += re*re + im*im;
        #endif
    }
    sum[tx] = lsum;
    magma_sum_reduce< BLOCK_SIZE >( tx, sum );
    
    if (tx==0)
        dxnorm[blockIdx.x] = sqrt(sum[0]);
}


//==============================================================================
__global__ void
magmablas_dnrm2_check_kernel( int m, double *da, int ldda, double *dxnorm, 
                               double *lsticc )
{
    const int tx = threadIdx.x;
    double *dx = da + blockIdx.x * ldda;

    __shared__ double sum[ BLOCK_SIZE ];
    double re, lsum;

    // get norm of dx only if lsticc[blockIdx+1] != 0
    if ( lsticc[blockIdx.x + 1] == 0 )
        return;

    lsum = 0;
    for( int j = tx; j < m; j += BLOCK_SIZE ) {
        #if (defined(PRECISION_s) || defined(PRECISION_d))
        re = dx[j];
        lsum += re*re;
        #else
        re = MAGMA_D_REAL( dx[j] );
        double im = MAGMA_D_IMAG( dx[j] );
        lsum += re*re + im*im;
        #endif
    }
    sum[tx] = lsum;
    magma_sum_reduce< BLOCK_SIZE >( tx, sum );
    
    if (tx==0)
        dxnorm[blockIdx.x] = sqrt(sum[0]);
}

extern "C" void
magmablas_dnrm2_check(
    magma_int_t m, magma_int_t n, double *da, magma_int_t ldda, 
    double *dxnorm, double *lsticc) 
{
    dim3  blocks( n );
    dim3 threads( BLOCK_SIZE );
    
    magmablas_dnrm2_check_kernel<<< blocks, threads >>>( m, da, ldda, dxnorm, lsticc );
}


//==============================================================================
__global__ void
magmablas_dnrm2_smkernel( int m, int n, double *da, int ldda,
                           double *dxnorm )
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    __shared__ double sum[ BLOCK_SIZEx ][ BLOCK_SIZEy + 1];
    double re, lsum;

    for( int k = ty; k < n; k += BLOCK_SIZEy ) {
        double *dx = da + k * ldda;

        // get norm of dx
        lsum = 0;
        for( int j = tx; j < m; j += BLOCK_SIZEx ) {
            #if (defined(PRECISION_s) || defined(PRECISION_d))
            re = dx[j];
            lsum += re*re;
            #else
            re = MAGMA_D_REAL( dx[j] );
            double im = MAGMA_D_IMAG( dx[j] );
            lsum += re*re + im*im;
            #endif
        }
        sum[tx][ty] = lsum;
        magma_sum_reduce_2d< BLOCK_SIZEx, BLOCK_SIZEy+1 >( tx, ty, sum );

        if (tx == 0)
            dxnorm[k] = sqrt(sum[0][ty]);
        __syncthreads();
    }
}


//==============================================================================
/*
    Compute the dnrm2 of each column of m-by-n matrix dA.
    The resulting norms are written in the dxnorm array.
    This routine uses only one SM (block).
*/
extern "C" void
magmablas_dnrm2_sm(
    magma_int_t m, magma_int_t n, double *da, magma_int_t ldda,
    double *dxnorm)
{
    dim3  blocks( 1 );
    dim3 threads( BLOCK_SIZEx, BLOCK_SIZEy );

    magmablas_dnrm2_smkernel<<< blocks, threads, 0, magma_stream >>>( m, n, da, ldda, dxnorm );
}

//==============================================================================

__global__ void
magma_dnrm2_adjust_kernel(double *xnorm, double *c)
{
    const int tx = threadIdx.x;

    __shared__ double sum[ BLOCK_SIZE ];
    double temp;

    temp = MAGMA_D_ABS( c[tx] ) / xnorm[0];
    sum[tx] = -temp * temp;
    magma_sum_reduce_n( blockDim.x, tx, sum );

    __syncthreads();
    if (tx == 0)
        xnorm[0] = xnorm[0] * sqrt(1+sum[0]);
}


/*
    Adjust the norm of c to give the norm of c[k+1:], assumin that
    c was changed with orthogonal transformations.
*/
extern "C" void
magmablas_dnrm2_adjust(magma_int_t k, double *xnorm, double *c)
{
    magma_dnrm2_adjust_kernel<<< 1, k, 0, magma_stream >>> (xnorm, c);
}

//==============================================================================

#define BS 256

__global__ void
magma_dnrm2_row_check_adjust_kernel(int n, double tol, double *xnorm, double *xnorm2, 
                                     double *c, int ldc, double *lsticc)
{
    const int tx = threadIdx.x + blockIdx.x*BS;
    lsticc[tx+1] = 0;

    if (tx < n) {
        double temp = MAGMA_D_ABS( c[tx*ldc] ) / xnorm[tx];
        temp = max( 0.0, ((1.0 + temp) * (1.0 - temp)) );
        
        
        double temp2 = xnorm[tx] / xnorm2[tx];
        temp2 = temp * (temp2 * temp2);
        
        if (temp2 <= tol) {
            lsticc[tx+1] = 1;
        } else {
            xnorm[tx] *= sqrt(temp);
        }
    }
    if (tx == 0)
        lsticc[0] = 0;
    magma_sum_reduce_n( blockDim.x, tx, lsticc );
}

/*
    Adjust the norm of c[,1:k] to give the norm of c[k+1:,1:k], assuming that
    c was changed with orthogonal transformations.
    It also do checks for QP3
*/
extern "C" void
magmablas_dnrm2_row_check_adjust(
    magma_int_t k, double tol, double *xnorm, double *xnorm2, 
    double *c, magma_int_t ldc, double *lsticc)
{
    int nblocks = (k+BS-1)/BS;
    magma_dnrm2_row_check_adjust_kernel<<< nblocks, BS >>> (k, tol, xnorm, xnorm2, c, ldc, lsticc);
}

//==============================================================================

/*
    Compute the dnrm2 of each column of m-by-n matrix dA.
    The resulting norms are written in the dxnorm array. 
    The computation can be done using n blocks (default) or on one SM (commented).
*/
extern "C" void
magmablas_dnrm2_cols(
    magma_int_t m, magma_int_t n,
    double *da, magma_int_t ldda, 
    double *dxnorm) 
{
    dim3  blocks( n );
    dim3 threads( BLOCK_SIZE );
    
    magmablas_dnrm2_kernel<<< blocks, threads, 0, magma_stream >>>( m, da, ldda, dxnorm );

    // The following would do the computation on one SM
    // magmablas_dnrm2_sm(m, n, da, ldda, dxnorm);
}

//==============================================================================
