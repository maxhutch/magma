/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal d

*/
#include "common_magma.h"

#define num_threads 64
#define gemv_bs     64

__global__ void
dgemv_kernel_MLU(
    int m, int n, int n1,
    const double * __restrict__ A, int lda,
    const double * __restrict__ x,
    double       * __restrict__ y)
{
    int ind = blockIdx.x*num_threads + threadIdx.x;

    A += ind;
    x += threadIdx.x;

    double res = 0;

    __shared__ double buff[gemv_bs];

    for( int i=0; i < n1; i += gemv_bs ) {
        __syncthreads();
        buff[threadIdx.x] = x[i];

        __syncthreads();
        #pragma unroll
        for(int j=0; j < gemv_bs; j++) {
            res += A[0]*buff[j];
            A += lda;
        }
    }
    __syncthreads();

    if ( n > n1 ) {
        if( (threadIdx.x + n1) >= n ) {
            x += (n - threadIdx.x - 1);
        }
        else{
            x += n1;
        }
        n = n - n1;
        /*
            Note
            ====
            Stan ............
            This is going to give segmentation fault or Error in GPU for illegal memory access. -- I am talking about x index
            buff[threadIdx.x]  = x[n1];
        */
        buff[threadIdx.x] = x[0];

        __syncthreads();
        for(int j=0; j < n; j++) {
            res += A[0]*buff[j];
            A += lda;
        }
    }

    if ( ind < m )
        y[ind] -= res;
}

extern "C" void
magmablas_dgemv_MLU(
    magma_int_t m, magma_int_t n,
    const double *A, magma_int_t lda,
    const double *x,
    double *y )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======

    This routine computes y = y - Ax on the GPU.

    M       (input) INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    N       (input) INTEGER.
            On entry, N specifies the number of columns of the matrix A

    A       (input) DOUBLE PRECISION array of dimension ( LDA, n ) on the GPU.

    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) DOUBLE PRECISION array of dimension n.

    Y       (output) DOUBLE PRECISION array of dimension n.
            On exit Y = Y - A X.

    ===================================================================== */

    magma_int_t blocks = (m - 1)/num_threads + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);

    dgemv_kernel_MLU<<< grid, threads, 0, magma_stream >>>
        (m, n, (n / gemv_bs)*gemv_bs, A, lda, x, y);
}

#undef num_threads
#undef gemv_bs
