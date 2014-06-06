/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated ds Tue Dec 17 13:18:44 2013

*/
#include "common_magma.h"

#define BLOCK_SIZE 64

// adds   X += R (including conversion to double)  --and--
// copies W = B
// each thread does one index, X[i] and W[i]
extern "C" __global__ void
dsaxpycp_kernel(
    int M, float *R, double *X,
    const double *B, double *W )
{
    const int i = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if ( i < M ) {
        X[i] = MAGMA_D_ADD( X[i], (double)( R[i] ) );
        W[i] = B[i];
    }
}


// adds   X += R  --and--
// copies R = B
// each thread does one index, X[i] and R[i]
extern "C" __global__ void
daxpycp_kernel(
    int M, double *R, double *X,
    const double *B)
{
    const int i = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if ( i < M ) {
        X[i] = MAGMA_D_ADD( X[i], R[i] );
        R[i] = B[i];
    }
}


// adds   X += R (including conversion to double)  --and--
// copies W = B
extern "C" void
magmablas_dsaxpycp(
    magma_int_t M, float *R, double *X,
    const double *B, double *W)
{
    dim3 threads( BLOCK_SIZE );
    dim3 grid( (M + BLOCK_SIZE - 1)/BLOCK_SIZE );
    dsaxpycp_kernel <<< grid, threads, 0, magma_stream >>> ( M, R, X, B, W );
}


// adds   X += R  --and--
// copies R = B
extern "C" void
magmablas_daxpycp(
    magma_int_t M, double *R, double *X,
    const double *B)
{
    dim3 threads( BLOCK_SIZE );
    dim3 grid( (M + BLOCK_SIZE - 1)/BLOCK_SIZE );
    daxpycp_kernel <<< grid, threads, 0, magma_stream >>> ( M, R, X, B );
}
