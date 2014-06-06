/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions mixed zc -> ds

*/
#include "common_magma.h"

#define BLOCK_SIZE 64

// adds   X += R (including conversion to double)  --and--
// copies W = B
// each thread does one index, X[i] and W[i]
extern "C" __global__ void
zcaxpycp_kernel(
    int M, magmaFloatComplex *R, magmaDoubleComplex *X,
    const magmaDoubleComplex *B, magmaDoubleComplex *W )
{
    const int i = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if ( i < M ) {
        X[i] = MAGMA_Z_ADD( X[i], cuComplexFloatToDouble( R[i] ) );
        W[i] = B[i];
    }
}


// adds   X += R  --and--
// copies R = B
// each thread does one index, X[i] and R[i]
extern "C" __global__ void
zaxpycp_kernel(
    int M, magmaDoubleComplex *R, magmaDoubleComplex *X,
    const magmaDoubleComplex *B)
{
    const int i = threadIdx.x + blockIdx.x*BLOCK_SIZE;
    if ( i < M ) {
        X[i] = MAGMA_Z_ADD( X[i], R[i] );
        R[i] = B[i];
    }
}


// adds   X += R (including conversion to double)  --and--
// copies W = B
extern "C" void
magmablas_zcaxpycp(
    magma_int_t M, magmaFloatComplex *R, magmaDoubleComplex *X,
    const magmaDoubleComplex *B, magmaDoubleComplex *W)
{
    dim3 threads( BLOCK_SIZE );
    dim3 grid( (M + BLOCK_SIZE - 1)/BLOCK_SIZE );
    zcaxpycp_kernel <<< grid, threads, 0, magma_stream >>> ( M, R, X, B, W );
}


// adds   X += R  --and--
// copies R = B
extern "C" void
magmablas_zaxpycp(
    magma_int_t M, magmaDoubleComplex *R, magmaDoubleComplex *X,
    const magmaDoubleComplex *B)
{
    dim3 threads( BLOCK_SIZE );
    dim3 grid( (M + BLOCK_SIZE - 1)/BLOCK_SIZE );
    zaxpycp_kernel <<< grid, threads, 0, magma_stream >>> ( M, R, X, B );
}
