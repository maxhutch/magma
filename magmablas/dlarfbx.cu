/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlarfbx.cu normal z -> d, Fri Jan 30 19:00:08 2015

*/
#include "common_magma.h"
#include "commonblas_d.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512




//==============================================================================
extern "C"
__global__ void 
magma_dgemv_kernel1(int m, const double * __restrict__ V, int ldv, 
                    const double * __restrict__ c, 
                    double *dwork)
{
    const int i = threadIdx.x;
    const double *dV = V + (blockIdx.x) * ldv;

    __shared__ double sum[ BLOCK_SIZE ];
    double lsum;

    /*  lsum := v**H * C  */
    lsum = MAGMA_D_ZERO;
    for( int j = i; j < m; j += BLOCK_SIZE )
       lsum += MAGMA_D_MUL( MAGMA_D_CNJG( dV[j] ), c[j] );
    
    sum[i] = lsum;
    magma_sum_reduce< BLOCK_SIZE >( i, sum );

    __syncthreads();
    if (i==0)
       dwork [blockIdx.x] = sum[0];
}

//==============================================================================
/*  ----------------------------------------------------------------------------- 
    Call 
        magma_dgemv_kernel3<<< n, BLOCK_SIZE>>>(m, V, ldv, c, dwork, tau)
    to compute
        DGEMV( "Conjugate transpose", m, n, -tau[0], V, ldv, c, 1, zero, dwork, 1)
        and to set c[0] to 1.
    i.e., 
        work = -tau[0] V**H c
    ----------------------------------------------------------------------------- */
extern "C"
__global__ void
magma_dgemv_kernel3(int m, const double * __restrict__ V, int ldv, double *c,
                    double *dwork, double *tau)
{
    const int i = threadIdx.x;
    const double *dV = V + (blockIdx.x) * ldv;

    __shared__ double sum[ BLOCK_SIZE ];
    double lsum;

    if (i==0)
       c[0] = MAGMA_D_ONE;           

    /*  lsum := v**H * C  */
    lsum = MAGMA_D_ZERO;
    for( int j = i; j < m; j += BLOCK_SIZE )
       lsum += MAGMA_D_MUL( MAGMA_D_CNJG( dV[j] ), c[j] );

    sum[i] = lsum;
    magma_sum_reduce< BLOCK_SIZE >( i, sum );

    __syncthreads();
    if (i==0)
       dwork [blockIdx.x] = -tau[0]*sum[0];
}

//==============================================================================
extern "C"
__global__ void
magma_dgemv_kernel2(int m, int n, const double * __restrict__ V, int ldv, 
                    const double * __restrict__ x, double *c)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    double lsum;

    V += j;

    lsum = MAGMA_D_ZERO;
    if (j < m){
       for(int k=0; k<n; k++)
          lsum += MAGMA_D_MUL( V[k*ldv], x[k]);
       
       c[j] -= lsum;
    }
}

//==============================================================================

/*
    Apply a real block reflector H to a real vector C from the left
    (i.e., C = H C). H is represented in the form
          H = I - V T V**H
    where T is the real k-by-k upper triangular matrix in the 
    representation of the block reflector, and V is a real block of
    k elementary reflectors. 
*/
extern "C" void
magma_dlarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaDouble_ptr V,  magma_int_t ldv,
    magmaDouble_ptr dT, magma_int_t ldt,
    magmaDouble_ptr c,
    magmaDouble_ptr dwork)
{
    /* dwork = V**H c     */
    magma_dgemv_kernel1<<< k, BLOCK_SIZE, 0, magma_stream >>>(m, V, ldv, c, dwork); 

    /* dwork = T**H dwork */
    magma_dtrmv_tkernel<<< k, k, 0, magma_stream >>>( dT, ldt, dwork, dwork+k);
 
    /* c = c - V dwork    */
    dim3  blocks3( (m + BLOCK_SIZE-1) / BLOCK_SIZE );
    dim3 threads3( BLOCK_SIZE );     
    magma_dgemv_kernel2<<< blocks3, threads3, 0, magma_stream >>>( m, k, V, ldv, dwork+k, c);
}

//==============================================================================
