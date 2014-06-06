/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal z -> s d c

*/
#include "common_magma.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  512
#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16

#define PRECISION_z

// ----------------------------------------
// Does sum reduction of array x, leaving total in x[0].
// Contents of x are destroyed in the process.
// With k threads, can reduce array up to 2*k in size.
// Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
// Having n as template parameter allows compiler to evaluate some conditions at compile time.
template< int n >
__device__ void sum_reduce( /*int n,*/ int i, double* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce


template< int n >
__device__ void sum_reduce_2d( /*int n,*/ int i, int c, double x[][BLOCK_SIZEy+1] )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i][c] += x[i+1024][c]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i][c] += x[i+ 512][c]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i][c] += x[i+ 256][c]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i][c] += x[i+ 128][c]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i][c] += x[i+  64][c]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i][c] += x[i+  32][c]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i][c] += x[i+  16][c]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i][c] += x[i+   8][c]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i][c] += x[i+   4][c]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i][c] += x[i+   2][c]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i][c] += x[i+   1][c]; }  __syncthreads(); }
}
// end sum_reduce


//==============================================================================

__global__ void
magmablas_dznrm2_kernel( int m, magmaDoubleComplex *da, int ldda, double *dxnorm )
{
    const int i = threadIdx.x;
    magmaDoubleComplex *dx = da + blockIdx.x * ldda;

    __shared__ double sum[ BLOCK_SIZE ];
    double re, lsum;

    // get norm of dx
    lsum = 0;
    for( int j = i; j < m; j += BLOCK_SIZE ) {

#if (defined(PRECISION_s) || defined(PRECISION_d))
        re = dx[j];
        lsum += re*re;
#else
        re = MAGMA_Z_REAL( dx[j] );
        double im = MAGMA_Z_IMAG( dx[j] );
        lsum += re*re + im*im;
#endif

    }
    sum[i] = lsum;
    sum_reduce< BLOCK_SIZE >( i, sum );
    
    if (i==0)
        dxnorm[blockIdx.x] = sqrt(sum[0]);
}


//==============================================================================
__global__ void
magmablas_dznrm2_check_kernel( int m, magmaDoubleComplex *da, int ldda, double *dxnorm, 
                               double *lsticc )
{
    const int i = threadIdx.x;
    magmaDoubleComplex *dx = da + blockIdx.x * ldda;

    __shared__ double sum[ BLOCK_SIZE ];
    double re, lsum;

    // get norm of dx only if lsticc[blockIdx+1] != 0
    if( lsticc[blockIdx.x + 1] == 0 ) return;

    lsum = 0;
    for( int j = i; j < m; j += BLOCK_SIZE ) {

#if (defined(PRECISION_s) || defined(PRECISION_d))
        re = dx[j];
        lsum += re*re;
#else
        re = MAGMA_Z_REAL( dx[j] );
        double im = MAGMA_Z_IMAG( dx[j] );
        lsum += re*re + im*im;
#endif

    }
    sum[i] = lsum;
    sum_reduce< BLOCK_SIZE >( i, sum );
    
    if (i==0)
        dxnorm[blockIdx.x] = sqrt(sum[0]);
}

extern "C" void
magmablas_dznrm2_check(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *da, magma_int_t ldda, 
    double *dxnorm, double *lsticc) 
{
    dim3  blocks( n );
    dim3 threads( BLOCK_SIZE );
    
    magmablas_dznrm2_check_kernel<<< blocks, threads >>>( m, da, ldda, dxnorm, lsticc );
}


//==============================================================================
__global__ void
magmablas_dznrm2_smkernel( int m, int n, magmaDoubleComplex *da, int ldda,
                           double *dxnorm )
{
    const int i = threadIdx.x, c= threadIdx.y;
    __shared__ double sum[ BLOCK_SIZEx ][ BLOCK_SIZEy + 1];
    double re, lsum;

    for( int k = c; k < n; k+= BLOCK_SIZEy) 
    {
        magmaDoubleComplex *dx = da + k * ldda;

        // get norm of dx
        lsum = 0;
        for( int j = i; j < m; j += BLOCK_SIZEx ) {

#if (defined(PRECISION_s) || defined(PRECISION_d))
                re = dx[j];
                lsum += re*re;
#else
                re = MAGMA_Z_REAL( dx[j] );
                double im = MAGMA_Z_IMAG( dx[j] );
                lsum += re*re + im*im;
#endif

        }
        sum[i][c] = lsum;
        sum_reduce_2d< BLOCK_SIZEx >( i, c, sum );

        if (i==0)
                dxnorm[k] = sqrt(sum[0][c]);
        __syncthreads();
    }
}


//==============================================================================
/*
    Compute the dznrm2 of each column of m-by-n matrix dA.
    The resulting norms are written in the dxnorm array.
    This routine uses only one SM (block).
*/
extern "C" void
magmablas_dznrm2_sm(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *da, magma_int_t ldda,
    double *dxnorm)
{
    dim3  blocks( 1 );
    dim3 threads( BLOCK_SIZEx, BLOCK_SIZEy );

    magmablas_dznrm2_smkernel<<< blocks, threads, 0, magma_stream >>>( m, n, da, ldda, dxnorm );
}

//==============================================================================

static
__device__ void dsum_reduce( int n, int i, double* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce

__global__ void
magma_dznrm2_adjust_kernel(double *xnorm, magmaDoubleComplex *c)
{
    const int i = threadIdx.x;

    __shared__ double sum[ BLOCK_SIZE ];
    double temp;

    temp = MAGMA_Z_ABS( c[i] ) / xnorm[0];
    sum[i] = -temp * temp;
    dsum_reduce( blockDim.x, i, sum );

    __syncthreads();
    if (i==0)
        xnorm[0] = xnorm[0] * sqrt(1+sum[0]);
}


/*
    Adjust the norm of c to give the norm of c[k+1:], assumin that
    c was changed with orthogonal transformations.
*/
extern "C" void
magmablas_dznrm2_adjust(magma_int_t k, double *xnorm, magmaDoubleComplex *c)
{
    magma_dznrm2_adjust_kernel<<< 1, k, 0, magma_stream >>> (xnorm, c);
}

//==============================================================================

#define BS 256

__global__ void
magma_dznrm2_row_check_adjust_kernel(int n, double tol, double *xnorm, double *xnorm2, 
                                     magmaDoubleComplex *c, int ldc, double *lsticc)
{
    const int i = threadIdx.x + blockIdx.x*BS;
    lsticc[i+1] = 0;

    if (i<n){
        double temp = MAGMA_Z_ABS( c[i*ldc] ) / xnorm[i];
        temp = max( 0.0, ((1.0 + temp) * (1.0 - temp)) );
        
        
        double temp2 = xnorm[i] / xnorm2[i];
        temp2 = temp * (temp2 * temp2);
        
        if (temp2 <= tol) {
            lsticc[i+1] = 1;
        } else {
            xnorm[i] *= sqrt(temp);
        }
    }
    if( i==0 ) lsticc[0] = 0;
    dsum_reduce( blockDim.x, i, lsticc );
}

/*
    Adjust the norm of c[,1:k] to give the norm of c[k+1:,1:k], assuming that
    c was changed with orthogonal transformations.
    It also do checks for QP3
*/
extern "C" void
magmablas_dznrm2_row_check_adjust(
    magma_int_t k, double tol, double *xnorm, double *xnorm2, 
    magmaDoubleComplex *c, magma_int_t ldc, double *lsticc)
{
    int nblocks = (k+BS-1)/BS;
    magma_dznrm2_row_check_adjust_kernel<<< nblocks, BS >>> (k, tol, xnorm, xnorm2, c, ldc, lsticc);
}

//==============================================================================

/*
    Compute the dznrm2 of each column of m-by-n matrix dA.
    The resulting norms are written in the dxnorm array. 
    The computation can be done using n blocks (default) or on one SM (commented).
*/
extern "C" void
magmablas_dznrm2_cols(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *da, magma_int_t ldda, 
    double *dxnorm) 
{
    dim3  blocks( n );
    dim3 threads( BLOCK_SIZE );
    
    magmablas_dznrm2_kernel<<< blocks, threads, 0, magma_stream >>>( m, da, ldda, dxnorm );

    // The following would do the computation on one SM
    // magmablas_dznrm2_sm(m, n, da, ldda, dxnorm);
}

//==============================================================================
