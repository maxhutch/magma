/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

*/
#include "common_magma.h"

#define num_threads 64
#define gemv_bs 64

__global__ void
sgemv_kernel_tesla(
    int m, int n, int n1,
    const float * __restrict__ A, int lda,
    const float * __restrict__ x,
    float       * __restrict__ y )
{
    int ind = blockIdx.x*num_threads + threadIdx.x;
    
    A += ind;
    x += threadIdx.x;
    
    float res = 0;
    
    __shared__ float buff[gemv_bs];
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
        buff[threadIdx.x] = x[n1];
        
        __syncthreads();
        for(int j=0; j < (n-n1); j++) {
            res += A[0]*buff[j];
            A += lda;
        }
    }
    
    if ( ind < m )
        y[ind] = res;
}

__global__ void
sgemv_kernel2_tesla(
    int m, int n, int n1,
    const float * __restrict__ A, int lda,
    const float * __restrict__ x, int incx,
    float       * __restrict__ y )
{
    int ind = blockIdx.x*num_threads + threadIdx.x;
    
    A += ind;
    x += threadIdx.x * incx;
    
    float res = 0;
    
    __shared__ float buff[gemv_bs];
    for( int i=0; i < n1; i += gemv_bs ) {
        __syncthreads();
        buff[threadIdx.x] = x[i*incx];
        
        __syncthreads();
        #pragma unroll
        for(int j=0; j < gemv_bs; j++) {
            res += A[0]*buff[j];
            A += lda;
        }
    }
    __syncthreads();
    
    if ( n > n1 ) {
        buff[threadIdx.x] = x[n1*incx];
        
        __syncthreads();
        for(int j=0; j < (n-n1); j++) {
            res += A[0]*buff[j];
            A += lda;
        }
    }
    
    if ( ind < m )
        y[ind] = res;
}

extern "C" void
magmablas_sgemvt_tesla(
    magma_int_t m, magma_int_t n, float alpha,
    const float *A, magma_int_t lda,
    const float *x,
    float       *y );

extern "C" void
magmablas_sgemv_tesla(
    char trans, magma_int_t m, magma_int_t n,
    float alpha,
    const float *A, magma_int_t lda,
    const float *x, magma_int_t incx,
    float beta,
    float       *y, magma_int_t incy)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    This routine computes:
    1) y =       A   x      if trans == 'N' or 'n', alpha == 1, beta == 0,
                            and incx == incy == 1 (using magmablas code)
    2) y = alpha A^T x      if trans == 'T' or 't', beta == 0,
                            and incx == incy == 1 (using magmablas code)
    3) y = alpha A^TRANS x + beta y
                            otherwise, using CUBLAS.

    Arguments
    ==========
    TRANS   CHARACTER*1
            On entry, TRANS specifies the operation to be performed as
            follows:
                TRANS = 'N' or 'n'   y := alpha*A  *x + beta*y
                TRANS = 'T' or 't'   y := alpha*A^T*x + beta*y
            
    M       (input) INTEGER
            On entry, M specifies the number of rows of the matrix A.
            
    N       (input) INTEGER
            On entry, N specifies the number of columns of the matrix A
            
    ALPHA   REAL
            On entry, ALPHA specifies the scalar alpha.
            Unchanged on exit.
            
    A       (input) REAL array of dimension (LDA, N) on the GPU.
            
    LDA     (input) INTEGER
            LDA specifies the leading dimension of A.
            
    X       (input) REAL array of dimension
            n if trans == 'n'
            m if trans == 't'
            
    INCX    (input) Specifies the increment for the elements of X.
            INCX must not be zero. Unchanged on exit.
            
    BETA    REAL
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.
            Unchanged on exit.
            
    Y       (output) REAL array of dimension
            m if trans == 'n'
            n if trans == 't'
            
    INCY    (input) Specifies the increment for the elements of Y.
            INCY must not be zero. Unchanged on exit.
    ===================================================================== */

    if ( incx == 1 && incy == 1 && beta == 0 ) {
        if ( trans == 'n' || trans == 'N' ) {
            if ( alpha == 1. ) {
                magma_int_t blocks = (m - 1)/num_threads + 1;
                dim3 grid( blocks, 1, 1 );
                dim3 threads( num_threads, 1, 1 );
                sgemv_kernel_tesla<<< grid, threads, 0, magma_stream >>>
                    (m, n, (n/gemv_bs)*gemv_bs, A, lda, x, y);
            }
            else {
                cublasSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
            }
        }
        else {
            magmablas_sgemvt_tesla(m, n, alpha, A, lda, x, y);
        }
    }
    else {
        cublasSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
}

extern "C" void
magmablas_sgemv2_tesla(
    magma_int_t m, magma_int_t n,
    const float *A, magma_int_t lda,
    const float *x, magma_int_t incx,
    float       *y )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    This routine computes y = A x on the GPU.
    This version has INCX as an argument.

    M       (input) INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    N       (input) INTEGER.
            On entry, N specifies the number of columns of the matrix A

    A       (input) REAL array of dimension (LDA, N) on the GPU.

    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) REAL array of dimension N.

    INCX    (input) Specifies the increment for the elements of X.
            INCX must not be zero. Unchanged on exit.

    Y       (output) REAL array of dimension M.
            On exit Y = A X.
    ===================================================================== */

    magma_int_t blocks = (m - 1)/num_threads + 1;
    dim3 grid( blocks, 1, 1 );
    dim3 threads( num_threads, 1, 1 );

    sgemv_kernel2_tesla<<< grid, threads, 0, magma_stream >>>
        (m, n, (n / gemv_bs)*gemv_bs, A, lda, x, incx, y);
}

__global__ void
sgemvt_kernel1_tesla(
    int m, int n, float alpha, int m1,
    const float * __restrict__ A, int lda,
    const float * __restrict__ x,
    float       * __restrict__ y )
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    
    int ind  = iny + __mul24(blockIdx.x,32);
        ind  = inx + __mul24(ind,lda);
    int ind2 = inx + __mul24(iny,32);
    
    A += ind;
    x += ind2;
    
    float res = 0;
    
    __shared__ float buff[gemv_bs];
    __shared__ float la[32][33];
    
    for( int i=0; i < m1; i += gemv_bs ) {
        buff[ind2] = x[i];
        #pragma unroll
        for(int j=0; j < 16; j++)
            la[iny+__mul24(j,2)][inx] = A[j*__mul24(2,lda)];
        
        __syncthreads();
        #pragma unroll
        for(int j=0; j < 16; j++)
            res += la[inx][iny*16+j]*buff[j+iny*16];
        
        A += 32;
        
        //===============================================
        #pragma unroll
        for(int j=0; j < 16; j++)
            la[iny+__mul24(j,2)][inx] = A[j*__mul24(2,lda)];
        
        __syncthreads();
        
        #pragma unroll
        for(int j=0; j < 16; j++)
            res += la[inx][iny*16+j]*buff[j+32+iny*16];
        A += 32;
    }
    
    if ( m > m1 ) {
        if ( ind2 >= (m-m1) )
            buff[ind2] = 0;
        else
            buff[ind2] = x[m1];
        
        #pragma unroll
        for(int j=0; j < 16; j++)
            la[iny+__mul24(j,2)][inx] = A[j*__mul24(2,lda)];
        
        __syncthreads();
        
        if ( m-m1 > 16 ) {
            #pragma unroll
            for(int j=0; j < 16; j++)
                res += la[inx][iny*16+j]*buff[j+iny*16];
            
            A += 32;
            #pragma unroll
            for(int j=0; j < 16; j++)
                la[iny+__mul24(j,2)][inx] = A[j*__mul24(2,lda)];
            
            __syncthreads();
            
            #pragma unroll
            for(int j=0; j < 16; j++)
                res += la[inx][iny*16+j]*buff[j+32+iny*16];
        }
        else {
            #pragma unroll
            for(int j=0; j < 16; j++)
                res += la[inx][iny*16+j]*buff[j+iny*16];
        }
    }
    ind = inx + __mul24(blockIdx.x,32);
    
    la[inx][iny] = res;
    if ( ind < n ) {
        res = la[inx][0] + la[inx][1];
        y[ind] = alpha*res;
    }
}

__global__ void
sgemvt_kernel2_tesla(
    int m, int n, float alpha, int m1,
    const float * __restrict__ A, int lda,
    const float * __restrict__ x,
    float       * __restrict__ y )
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    
    int ind  = iny + __mul24(blockIdx.x,16);
        ind  = inx + __mul24(ind,lda);
    int ind2 = inx + __mul24(iny,16);
    if ( ind2 > 31 )
        ind2 -= 32;
    
    A += ind;
    x += ind2;
    // dgemv_tesla has:
    // if ( ind2 > 31 )
    //     ind2 -= 32;
    
    float res = 0;
    
    __shared__ float buff[32];
    __shared__ float la[16][17];
    
    for( int i=0; i < m1; i += 32 ) {
        buff[ind2] = x[i];
        #pragma unroll
        for(int j=0; j < 4; j++)
            la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];
        
        __syncthreads();
        #pragma unroll
        for(int j=0; j < 4; j++)
            res += la[inx][iny*4+j]*buff[j+iny*4];
        
        A += 16;
        __syncthreads();
        //===========================================
        #pragma unroll
        for(int j=0; j < 4; j++)
            la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];
        
        __syncthreads();
        
        #pragma unroll
        for(int j=0; j < 4; j++)
            res += la[inx][iny*4+j]*buff[j+16+iny*4];
        A += 16;
    }
    
    __syncthreads(); // 1
    if ( m > m1 ) {
        if ( ind2 >= (m-m1) )
            buff[ind2] = 0;
        else
            buff[ind2] = x[m1];
        
        __syncthreads();
        #pragma unroll
        for(int j=0; j < 4; j++) {
            if ( inx >= (m-m1) )
                la[iny+__mul24(j,4)][inx] = 0;
            else
                la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];
        }
        
        __syncthreads();
        if ( m-m1 > 4 ) {
            #pragma unroll
            for(int j=0; j < 4; j++) {
                ind = j+iny*4;
                res += la[inx][ind]*buff[ind];
            }
            
            A += 16;
            __syncthreads();
            #pragma unroll
            for(int j=0; j < 4; j++) {
                if ( inx+16 >= (m-m1) )
                    la[iny+__mul24(j,4)][inx] = 0;
                else
                    la[iny+__mul24(j,4)][inx] = A[j*__mul24(4,lda)];
            }
            
            __syncthreads();
            
            #pragma unroll
            for(int j=0; j < 4; j++) {
                ind = j+iny*4;
                res += la[inx][ind]*buff[16+ind];
            }
        }
        else {
            #pragma unroll
            for(int j=0; j < 4; j++) {
                ind = j+iny*4;
                res += la[inx][ind]*buff[ind];
            }
        }
    }
    
    __syncthreads();
    ind = inx + __mul24(blockIdx.x,16);
    la[inx][iny] = res;
    __syncthreads();
    if ( ind < n && iny == 0 ) {
        res = la[inx][0] + la[inx][1] + la[inx][2] + la[inx][3];
        y[ind] = alpha*res;
    }
}

extern "C" void
magmablas_sgemvt1_tesla(
    magma_int_t m, magma_int_t n, float alpha,
    const float *A, magma_int_t lda,
    const float *x,
    float       *y )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    This routine computes y = alpha A^T x on the GPU.
    Recommended for large M and N.

    M       (input) INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    N       (input) INTEGER.
            On entry, N specifies the number of columns of the matrix A

    A       (input) REAL array of dimension (LDA, N) on the GPU.

    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) REAL array of dimension m.

    Y       (output) REAL array of dimension n.
            On exit Y = alpha A^T X.
    ===================================================================== */

    magma_int_t blocks = (n - 1)/32 + 1;
    dim3 grid( blocks, 1, 1 );
    dim3 threads( 32, 2, 1 );
    sgemvt_kernel1_tesla<<< grid, threads, 0, magma_stream >>>
        (m, n, alpha, (m / gemv_bs)*gemv_bs, A, lda, x, y);
}

extern "C" void
magmablas_sgemvt2_tesla(
    magma_int_t m, magma_int_t n, float alpha,
    const float *A, magma_int_t lda,
    const float *x,
    float       *y )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    This routine computes y = alpha A^T x on the GPU. Used in least squares
    solver for N small (e.g. = BS, a block size of order 64, 128, etc).

    M       (input) INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    N       (input) INTEGER.
            On entry, N specifies the number of columns of the matrix A

    A       (input) REAL array of dimension (LDA, N) on the GPU.

    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) REAL array of dimension m.

    Y       (output) REAL array of dimension n.
            On exit Y = alpha A^T X.
    ===================================================================== */

    magma_int_t blocks = (n - 1)/16 + 1;
    dim3 grid( blocks, 1, 1 );
    dim3 threads( 16, 4, 1 );
    sgemvt_kernel2_tesla<<< grid, threads, 0, magma_stream >>>
        (m, n, alpha, (m / 32)*32, A, lda, x, y);
}

extern "C" void
magmablas_sgemvt_tesla(
    magma_int_t m, magma_int_t n, float alpha,
    const float *A, magma_int_t lda,
    const float *x,
    float       *y )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    This routine computes y = alpha A^T x on the GPU.

    M       (input) INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    N       (input) INTEGER.
            On entry, N specifies the number of columns of the matrix A

    A       (input) REAL array of dimension (LDA, N) on the GPU.

    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) REAL array of dimension m.

    Y       (output) REAL array of dimension n.
            On exit Y = alpha A^T X.
    ===================================================================== */

    if ( n <= 128 )
        magmablas_sgemvt2_tesla(m, n, alpha, A, lda, x, y);
    else
        magmablas_sgemvt1_tesla(m, n, alpha, A, lda, x, y);
}

#undef num_threads
#undef gemv_bs
