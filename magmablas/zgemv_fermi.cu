/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal z -> c
*/
#include "common_magma.h"
#include "commonblas_z.h"

#define PRECISION_z

#define num_threads 128
#define gemv_bs      32
#define threadSize  128

__global__ void
zgemvn_kernel1_fermi(
    int m, int n, int n1, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y)
{
#if (__CUDA_ARCH__ >= 200)
    int ind = blockIdx.x*num_threads + threadIdx.x;
    
    A += ind;
    
    magmaDoubleComplex res = MAGMA_Z_ZERO;
    
    for( int i=0; i < n1; i += gemv_bs ) {
        #pragma unroll
        for(int j=0; j < gemv_bs; j++) {
            res += A[0] * x[j];
            A   += lda;
        }
        x += gemv_bs;
    }
    
    if ( n > n1 ) {
        for(int j=0; j < (n-n1); j++) {
            res += A[0] * x[j];
            A   += lda;
        }
    }
    
    if ( ind < m )
        y[ind] = alpha * res + beta * y[ind];
#endif /* (__CUDA_ARCH__ >= 200) */
}

__global__ void
zgemvn_kernel2_fermi(
    int m, int n, int n1, magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y)
{
#if (__CUDA_ARCH__ >= 200)
    int ind = blockIdx.x*num_threads + threadIdx.x;
    
    A += ind;
    x += threadIdx.x;
    
    magmaDoubleComplex res = MAGMA_Z_ZERO;
    
    __shared__ magmaDoubleComplex buff[num_threads];
    for( int i=0; i < n1; i += num_threads ) {
        __syncthreads();
        buff[threadIdx.x] = x[i];
        
        __syncthreads();
        #pragma unroll
        for(int j=0; j < num_threads; j++) {
            res += A[0]*buff[j];
            A += lda;
        }
    }
    __syncthreads();
    
    if ( n > n1 ) {
        buff[threadIdx.x] = x[n1];
        
        __syncthreads();
        for(int j=0; j<(n-n1); j++) {
            res += A[0]*buff[j];
            A += lda;
        }
    }
    
    if ( ind < m )
        y[ind] = alpha * res + beta * y[ind];
#endif /* (__CUDA_ARCH__ >= 200) */
}

extern "C" void
magmablas_zgemvn_fermi(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    const magmaDoubleComplex *x, magmaDoubleComplex beta,
    magmaDoubleComplex       *y)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======

    This routine computes Y = alpha A x on the GPU.

    M       (input) INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    N       (input) INTEGER.
            On entry, N specifies the number of columns of the matrix A

    A       (input) COMPLEX*16 array of dimension ( LDA, n ) on the GPU.
   
    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) COMPLEX*16 array of dimension n.
     
    Y       (output) COMPLEX*16 array of dimension n.
            On exit Y = alpha A X.

    ===================================================================== */

    magma_int_t blocks = (m - 1)/num_threads + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);
    /*
    if ( m <= 8500 )
        zgemvn_kernel1_fermi<<< grid, threads, 0, magma_stream >>>
            (m, n, (n / gemv_bs)*gemv_bs, alpha, A, lda, x, y);
    else
    */
        zgemvn_kernel2_fermi<<< grid, threads, 0, magma_stream >>>
            (m, n, (n / num_threads)*num_threads, alpha, A, lda, x, beta, y);
}

__global__ void
zgemvt_kernel_fermi(
    int m, int n, magmaDoubleComplex alpha, int n1,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y)
{
#if (__CUDA_ARCH__ >= 200)
    unsigned int tx = threadIdx.x;

    __shared__ magmaDoubleComplex sdata[threadSize];

    magmaDoubleComplex res = MAGMA_Z_ZERO;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
 
    for(int i=0; i < n1; i += threadSize) {
        res += A[tx + i + lda * blockIdx.y] * x[tx + i];
    }

    if ( m > n1 ) {
        if ( tx + n1 < m ) {
            res += A[tx + n1 + lda*blockIdx.y] * x[tx + n1];
        }
        else {
            res += c_zero;
        }
    }

    sdata[tx] = res;
    __syncthreads();

    for(int s=blockDim.x/2; s > 32; s /= 2) {
        if ( tx < s ) {
            sdata[tx] += sdata[tx+s];
        }
        __syncthreads();
    }

    if ( tx < 32 ) {
        sdata[tx] += sdata[tx + 32];
    }

    if ( tx == 0 ) {
        for(int i=1; i < 32; i++) {
            sdata[tx] += sdata[tx + i];
        }
    }

    if ( tx == 0 ) {
        if ( blockIdx.y < n ) {
            y[blockIdx.y] = sdata[0] * alpha + beta * y[blockIdx.y];
        }
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}

extern "C" void
magmablas_zgemvt_fermi(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    const magmaDoubleComplex *x, magmaDoubleComplex beta,
    magmaDoubleComplex       *y)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======

    This routine computes y = alpha * A^T * x on the GPU.

    M       (input) INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    N       (input) INTEGER.
            On entry, N specifies the number of columns of the matrix A

    A       (input) COMPLEX*16 array of dimension ( LDA, n ) on the GPU.

    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) COMPLEX*16 array of dimension m.

    Y       (output) COMPLEX*16 array of dimension n.
            On exit Y = alpha A^T X.

    ===================================================================== */

    dim3 grid    ( 1, n, 1 );
    dim3 threads ( threadSize, 1, 1 );
    zgemvt_kernel_fermi<<< grid, threads, 0, magma_stream >>>
        (m, n, alpha, (m / threadSize) * threadSize, A, lda, x, beta, y );
}

__global__ void
zgemvc_kernel_fermi(
    int m, int n, magmaDoubleComplex alpha, int n1,
    const magmaDoubleComplex * __restrict__ A, int lda,
    const magmaDoubleComplex * __restrict__ x, magmaDoubleComplex beta,
    magmaDoubleComplex       * __restrict__ y)
{
#if (__CUDA_ARCH__ >= 200)
    unsigned int tx = threadIdx.x;

    __shared__ magmaDoubleComplex sdata[threadSize];

    magmaDoubleComplex res = MAGMA_Z_ZERO;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
 
    for(int i=0; i < n1; i += threadSize) {
        res += cuConj(A[tx + i + lda * blockIdx.y]) * x[tx + i];
    }

    if ( m > n1 ) {
        if ( tx + n1 < m ) {
            res += cuConj(A[tx + n1 + lda*blockIdx.y]) * x[tx + n1];
        }
        else {
            res += c_zero;
        }
    }

    sdata[tx] = res;
    __syncthreads();

    /*
    if ( tx < 128 ) {
        sdata[tx] += sdata[tx + 128];
    }
    __syncthreads();
    */

    if ( tx < 64 ) {
        sdata[tx] += sdata[tx + 64];
    }
    __syncthreads();

    if ( tx < 32 ) {
        sdata[tx] += sdata[tx + 32];
    }

    if ( tx == 0 ) {
        for(int i=1; i < 32; i++) {
            sdata[tx] += sdata[tx + i];
        }
    }

    if ( tx == 0 ) {
        if ( blockIdx.y < n ) {
            y[blockIdx.y] = sdata[0] * alpha + beta * y[blockIdx.y];
        }
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}

extern "C" void
magmablas_zgemvc_fermi(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    const magmaDoubleComplex *x,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======

    This routine computes y = alpha * A^H * x on the GPU.

    M       (input) INTEGER.
            On entry, M specifies the number of rows of the matrix A.

    N       (input) INTEGER.
            On entry, N specifies the number of columns of the matrix A

    A       (input) COMPLEX*16 array of dimension ( LDA, n ) on the GPU.

    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) COMPLEX*16 array of dimension m.

    Y       (output) COMPLEX*16 array of dimension n.
            On exit Y = alpha A^H X.

    ===================================================================== */

    dim3 grid    ( 1, n, 1 );
    dim3 threads ( threadSize, 1, 1 );
    zgemvc_kernel_fermi<<< grid, threads, 0, magma_stream >>>
        (m, n, alpha, (m / threadSize) * threadSize, A, lda, x, beta, y);
}

extern "C" void
magmablas_zgemv(
    char trans, magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    const magmaDoubleComplex *x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, magma_int_t incy)
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // call CUDA ARCH 1.x version
        // magmablas for [sd] precisions, cublas for [zc] precisions.
        #if defined(PRECISION_z) || defined(PRECISION_c)
        cublasZgemv( trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
        #else
        magmablas_zgemv_tesla( trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
        #endif
        return;
    }
    
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( incx == 1 && incy == 1 ) {
        if ( trans == 'N' || trans == 'n' ) {
            if ( m < 7000 ) {
                cublasZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
            }
            else {
                magmablas_zgemvn_fermi(m, n, alpha, A, lda, x, beta, y);
            }
        }
        else if ( trans == 'T' || trans == 't' ) {
            magmablas_zgemvt_fermi(m, n, alpha, A, lda, x, beta, y);
        }
        else if ( trans == 'C' || trans == 'c' ) {
            magmablas_zgemvc_fermi(m, n, alpha, A, lda, x, beta, y);
        }
        else {
            fprintf( stderr, "trans = %c is invalid\n", trans );
        }
    }
    else {
        cublasZgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
}

#undef num_threads
#undef gemv_bs
#undef threadSize
