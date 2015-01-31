/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

*/
#include "common_magma.h"

#define num_threads 64
#define gemv_bs 64

__global__ void
dgemv_kernel_tesla(
    int m, int n, int n1,
    const double * __restrict__ A, int lda,
    const double * __restrict__ x,
    double       * __restrict__ y )
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

extern "C" void
magmablas_dgemvt_tesla(
    magma_int_t m, magma_int_t n, double alpha,
    const double *A, magma_int_t lda,
    const double *x,
    double       *y );

/**
    Purpose
    -------
    This routine computes:
    1) y =       A   x      if trans == MagmaNoTrans, alpha == 1, beta == 0,
                            and incx == incy == 1 (using magmablas code)
    2) y = alpha A^T x      if trans == MagmaTrans, beta == 0,
                            and incx == incy == 1 (using magmablas code)
    3) y = alpha A^TRANS x + beta y
                            otherwise, using CUBLAS.

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^T*x + beta*y
            
    @param[in]
    m       INTEGER
            On entry, M specifies the number of rows of the matrix A.
            
    @param[in]
    n       INTEGER
            On entry, N specifies the number of columns of the matrix A
            
    @param[in]
    alpha   DOUBLE PRECISION
            On entry, ALPHA specifies the scalar alpha.
            
    @param[in]
    A       DOUBLE PRECISION array of dimension (LDA, N) on the GPU.
            
    @param[in]
    lda     INTEGER
            LDA specifies the leading dimension of A.
            
    @param[in]
    x       DOUBLE PRECISION array of dimension
            n if trans == 'n'
            m if trans == 't'
            
    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.
        
    @param[in]
    beta    DOUBLE PRECISION
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.
            
    @param[out]
    y       DOUBLE PRECISION array of dimension
            m if trans == 'n'
            n if trans == 't'
            
    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @ingroup magma_dblas2
    ********************************************************************/
extern "C" void
magmablas_dgemv_tesla(
    magma_trans_t trans, magma_int_t m, magma_int_t n,
    double alpha,
    const double *A, magma_int_t lda,
    const double *x, magma_int_t incx,
    double beta,
    double       *y, magma_int_t incy)
{
    magma_int_t info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( lda < m )
        info = -6;
    else if ( incx == 0 )
        info = -8;
    else if ( incy == 0 )
        info = -11;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( incx == 1 && incy == 1 && beta == 0 ) {
        if ( trans == MagmaNoTrans ) {
            if ( alpha == 1. ) {
                magma_int_t blocks = (m - 1)/num_threads + 1;
                dim3 grid( blocks, 1, 1 );
                dim3 threads( num_threads, 1, 1 );
                dgemv_kernel_tesla<<< grid, threads, 0, magma_stream >>>
                    (m, n, (n/gemv_bs)*gemv_bs, A, lda, x, y);
            }
            else {
                magma_dgemv( trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
            }
        }
        else {
            magmablas_dgemvt_tesla(m, n, alpha, A, lda, x, y);
        }
    }
    else {
        magma_dgemv( trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
}

__global__ void
dgemvt_kernel1_tesla(
    int m, int n, double alpha, int m1,
    const double * __restrict__ A, int lda,
    const double * __restrict__ x,
    double       * __restrict__ y )
{
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    
    int ind  = iny + __mul24(blockIdx.x,32);
        ind  = inx + __mul24(ind,lda);
    int ind2 = inx + __mul24(iny,32);
    
    A += ind;
    x += ind2;
    
    double res = 0;
    
    __shared__ double buff[gemv_bs];
    __shared__ double la[32][33];
    
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
dgemvt_kernel2_tesla(
    int m, int n, double alpha, int m1,
    const double * __restrict__ A, int lda,
    const double * __restrict__ x,
    double       * __restrict__ y )
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
    if ( ind2 > 31 )
        ind2 -= 32;
    
    double res = 0;
    
    __shared__ double buff[32];
    __shared__ double la[16][17];
    
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
    
    // sgemv_tesla has:
    // __syncthreads();
    if ( m > m1 ) {
        if ( ind2 >= (m-m1) )
            buff[ind2] = 0;
        else
            buff[ind2] = x[m1];
        
        __syncthreads();
        #pragma unroll
        for(int j=0; j < 4; j++) {
            // sgemv_tesla has:
            // if ( inx >= (m-m1) )
            //     la[iny+__mul24(j,4)][inx] = 0;
            // else
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
                // sgemv_tesla has:
                // if ( inx+16 >= (m-m1) )
                //     la[iny+__mul24(j,4)][inx] = 0;
                // else
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
    // sgemv_tesla has:
    // if ( ind < n && iny == 0 ) {
    if ( ind < n ) {
        res = la[inx][0] + la[inx][1] + la[inx][2] + la[inx][3];
        y[ind] = alpha*res;
    }
}

/**
    Purpose
    -------
    This routine computes y = alpha A^T x on the GPU.
    Recommended for large M and N.

    @param[in]
    m       INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER.
            On entry, N specifies the number of columns of the matrix A

    @param[in]
    alpha   DOUBLE PRECISION.
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    A       DOUBLE PRECISION array of dimension (LDA, N) on the GPU.

    @param[in]
    lda     INTEGER.
            LDA specifies the leading dimension of A.

    @param[in]
    x       DOUBLE PRECISION array of dimension m.

    @param[out]
    y       DOUBLE PRECISION array of dimension n.
            On exit Y = alpha A^T X.

    @ingroup magma_dblas2_internal
    ********************************************************************/
extern "C" void
magmablas_dgemvt1_tesla(
    magma_int_t m, magma_int_t n, double alpha,
    const double *A, magma_int_t lda,
    const double *x,
    double       *y )
{
    magma_int_t blocks = (n - 1)/32 + 1;
    dim3 grid( blocks, 1, 1 );
    dim3 threads( 32, 2, 1 );
    dgemvt_kernel1_tesla<<< grid, threads, 0, magma_stream >>>
        (m, n, alpha, (m / gemv_bs)*gemv_bs, A, lda, x, y);
}


/**
    Purpose
    -------
    This routine computes y = alpha A^T x on the GPU. Used in least squares
    solver for N small (e.g. = BS, a block size of order 64, 128, etc).

    @param[in]
    m       INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER.
            On entry, N specifies the number of columns of the matrix A

    @param[in]
    alpha   DOUBLE PRECISION.
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    A       DOUBLE PRECISION array of dimension (LDA, N) on the GPU.

    @param[in]
    lda     INTEGER.
            LDA specifies the leading dimension of A.

    @param[in]
    x       DOUBLE PRECISION array of dimension m.

    @param[out]
    y       DOUBLE PRECISION array of dimension n.
            On exit Y = alpha A^T X.

    @ingroup magma_dblas2_internal
    ********************************************************************/
extern "C" void
magmablas_dgemvt2_tesla(
    magma_int_t m, magma_int_t n, double alpha,
    const double *A, magma_int_t lda,
    const double *x,
    double       *y )
{
    magma_int_t blocks = (n - 1)/16 + 1;
    dim3 grid( blocks, 1, 1 );
    dim3 threads( 16, 4, 1 );
    dgemvt_kernel2_tesla<<< grid, threads, 0, magma_stream >>>
        (m, n, alpha, (m / 32)*32, A, lda, x, y);
}


/**
    Purpose
    -------
    This routine computes y = alpha A^T x on the GPU.

    @param[in]
    m       INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER.
            On entry, N specifies the number of columns of the matrix A

    @param[in]
    alpha   DOUBLE PRECISION.
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    A       DOUBLE PRECISION array of dimension (LDA, N) on the GPU.

    @param[in]
    lda     INTEGER.
            LDA specifies the leading dimension of A.

    @param[in]
    x       DOUBLE PRECISION array of dimension m.

    @param[out]
    y       DOUBLE PRECISION array of dimension n.
            On exit Y = alpha A^T X.

    @ingroup magma_dblas2_internal
    ********************************************************************/
extern "C" void
magmablas_dgemvt_tesla(
    magma_int_t m, magma_int_t n, double alpha,
    const double *A, magma_int_t lda,
    const double *x,
    double       *y )
{
    if ( n <= 128 )
        magmablas_dgemvt2_tesla(m, n, alpha, A, lda, x, y);
    else
        magmablas_dgemvt1_tesla(m, n, alpha, A, lda, x, y);
}

#undef num_threads
#undef gemv_bs
