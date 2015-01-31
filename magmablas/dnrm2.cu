/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from dznrm2.cu normal z -> d, Fri Jan 30 19:00:09 2015

*/
#include "common_magma.h"
#include "commonblas_d.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  512
#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16

#define PRECISION_d


//==============================================================================

__global__ void
magmablas_dnrm2_kernel( int m, double *dA, int ldda, double *dxnorm )
{
    const int tx = threadIdx.x;
    double *dx = dA + blockIdx.x * ldda;

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
magmablas_dnrm2_check_kernel( int m, double *dA, int ldda, double *dxnorm, 
                               double *lsticc )
{
    const int tx = threadIdx.x;
    double *dx = dA + blockIdx.x * ldda;

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
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc) 
{
    dim3  blocks( n );
    dim3 threads( BLOCK_SIZE );
    
    magmablas_dnrm2_check_kernel<<< blocks, threads >>>( m, dA, ldda, dxnorm, dlsticc );
}


//==============================================================================
__global__ void
magmablas_dnrm2_smkernel( int m, int n, double *dA, int ldda,
                           double *dxnorm )
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    __shared__ double sum[ BLOCK_SIZEx ][ BLOCK_SIZEy + 1];
    double re, lsum;

    for( int k = ty; k < n; k += BLOCK_SIZEy ) {
        double *dx = dA + k * ldda;

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
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *dxnorm)
{
    dim3  blocks( 1 );
    dim3 threads( BLOCK_SIZEx, BLOCK_SIZEy );

    magmablas_dnrm2_smkernel<<< blocks, threads, 0, magma_stream >>>( m, n, dA, ldda, dxnorm );
}

//==============================================================================
extern "C"
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
    Adjust the norm of c to give the norm of c[k+1:], assuming that
    c was changed with orthogonal transformations.
*/
extern "C" void
magmablas_dnrm2_adjust(magma_int_t k, magmaDouble_ptr dxnorm, magmaDouble_ptr dc)
{
    magma_dnrm2_adjust_kernel<<< 1, k, 0, magma_stream >>> (dxnorm, dc);
}

//==============================================================================

#define BS 256

__global__ void
magma_dnrm2_row_check_adjust_kernel(
    int n, double tol, double *xnorm, double *xnorm2, 
    double *C, int ldc, double *lsticc)
{
    const int tx = threadIdx.x + blockIdx.x*BS;
    lsticc[tx+1] = 0;

    if (tx < n) {
        double temp = MAGMA_D_ABS( C[tx*ldc] ) / xnorm[tx];
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
    Adjust the norm of C[,1:k] to give the norm of C[k+1:,1:k], assuming that
    C was changed with orthogonal transformations.
    It also do checks for QP3
*/
extern "C" void
magmablas_dnrm2_row_check_adjust(
    magma_int_t k, double tol,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dxnorm2, 
    magmaDouble_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dlsticc)
{
    int nblocks = (k+BS-1)/BS;
    magma_dnrm2_row_check_adjust_kernel<<< nblocks, BS >>> (k, tol, dxnorm, dxnorm2, dC, lddc, dlsticc);
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
    magmaDouble_ptr dA, magma_int_t ldda, 
    magmaDouble_ptr dxnorm) 
{
    dim3  blocks( n );
    dim3 threads( BLOCK_SIZE );
    
    magmablas_dnrm2_kernel<<< blocks, threads, 0, magma_stream >>>( m, dA, ldda, dxnorm );

    // The following would do the computation on one SM
    // magmablas_dnrm2_sm(m, n, dA, ldda, dxnorm);
}

//==============================================================================
