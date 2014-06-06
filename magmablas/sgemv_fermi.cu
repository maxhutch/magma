/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

*/
#include "common_magma.h"
#include "commonblas_s.h"

#define PRECISION_s

#define num_threads 128
#define gemv_bs      32
#define threadSize  128

__global__ void
sgemvn_kernel1_fermi(
    int m, int n, int n1, float alpha,
    const float * __restrict__ A, int lda,
    const float * __restrict__ x, float beta,
    float       * __restrict__ y)
{
#if (__CUDA_ARCH__ >= 200)
    int ind = blockIdx.x*num_threads + threadIdx.x;
    
    A += ind;
    
    float res = 0.f;
    
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
sgemvn_kernel2_fermi(
    int m, int n, int n1, float alpha,
    const float * __restrict__ A, int lda,
    const float * __restrict__ x, float beta,
    float       * __restrict__ y)
{
#if (__CUDA_ARCH__ >= 200)
    int ind = blockIdx.x*num_threads + threadIdx.x;
    
    A += ind;
    x += threadIdx.x;
    
    float res = 0.f;
    
    __shared__ float buff[num_threads];
    for( int i=0; i < n1; i += num_threads ) {
        __syncthreads();
        buff[threadIdx.x]  = x[i];
        
        __syncthreads();
        #pragma unroll
        for(int j=0; j < num_threads; j++) {
            res += A[0]*buff[j];
            A += lda;
        }
    }
    __syncthreads();
    
    if ( n > n1 ) {
        buff[threadIdx.x]  = x[n1];
        
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
magmablas_sgemvn_fermi(
    magma_int_t m, magma_int_t n, float alpha,
    const float *A, magma_int_t lda,
    const float *x, float beta,
    float       *y)
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

    A       (input) REAL array of dimension ( LDA, n ) on the GPU.
   
    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) REAL array of dimension n.
     
    Y       (output) REAL array of dimension n.
            On exit Y = alpha A X.

    ===================================================================== */

    magma_int_t blocks = (m - 1)/num_threads + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);
    if ( m <= 8500 )
        sgemvn_kernel1_fermi<<< grid, threads, 0, magma_stream >>>
            (m, n, (n / gemv_bs)*gemv_bs, alpha, A, lda, x, beta, y);
    else
        sgemvn_kernel2_fermi<<< grid, threads, 0, magma_stream >>>
            (m, n, (n / num_threads)*num_threads, alpha, A, lda, x, beta, y);
}

__global__ void
sgemvt_kernel1_fermi(
    int m, int n, float alpha, int m1,
    const float * __restrict__ A, int lda,
    const float * __restrict__ x, float beta,
    float       * __restrict__ y)
{
#if (__CUDA_ARCH__ >= 200)
    int tx = threadIdx.x;

    __shared__ float sdata[threadSize];
    
    volatile float *smem;

    float res;
    res = 0.0f;
 
    for(int i=0; i < m1; i += threadSize) {
        res += A[tx + i + lda * blockIdx.y] * x[tx + i];
    }

    if ( m > m1 ) {
        if ( tx + m1 <  m ) {
            res  += A[tx + m1 + lda*blockIdx.y] * x[tx + m1];
        }
        else {
            res  += 0.0f;
        }
    }

    sdata[tx] = res;
    __syncthreads();

    for(int s=blockDim.x/2; s > 32; s /= 2)  {
        if ( tx < s ) {
            sdata[tx] += sdata[tx + s];
        }
        __syncthreads();
    }

    if ( tx < 32 ) {
        smem = sdata;
        smem[tx] += smem[tx + 32];
        smem[tx] += smem[tx + 16];
        smem[tx] += smem[tx +  8];
        smem[tx] += smem[tx +  4];
        smem[tx] += smem[tx +  2];
        smem[tx] += smem[tx +  1];
    }

    if ( tx == 0 )  {
        if ( blockIdx.y < n ) {
            y[blockIdx.y] = sdata[0] * alpha + beta * y[blockIdx.y];
        }
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}

__global__ void
sgemvt_kernel2_fermi(
    int m, int n, float alpha, int n1,
    const float * __restrict__ A, int lda,
    const float * __restrict__ x, float beta,
    float       * __restrict__ y)
{
#if (__CUDA_ARCH__ >= 200)
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    
    int ind  = iny + blockIdx.x * 16;
    ind = inx + ind * lda;
    int ind2 = inx + iny * 16;
    if ( ind2 > 31 )
        ind2 -= 32;
    
    A += ind;
    x += ind2;
    
    float res = 0.f;
    
    __shared__ float buff[32];
    __shared__ float la[16][17];
    
    for( int i=0; i < n1; i += 32 ) {
        buff[ind2]  = x[i];
        #pragma unroll
        for(int j=0; j < 4; j++)
            la[iny + j * 4][inx] = A[j* 4 * lda];
        
        __syncthreads();
        #pragma unroll
        for(int j=0; j < 4; j++)
            res += la[inx][iny*4+j]*buff[j+iny*4];
        
        A += 16;
        
        __syncthreads();
        //===========================================
        #pragma unroll
        for(int j=0; j < 4; j++)
            la[iny+ j * 4][inx] = A[j* 4 * lda];
        
        __syncthreads();
        
        #pragma unroll
        for(int j=0; j < 4; j++)
            res += la[inx][iny*4+j]*buff[j+16+iny*4];
        A += 16;
    }
    
    __syncthreads(); // 1
    if ( n > n1 ) {
        if ( ind2 >= (n-n1) )
            buff[ind2]=0.;
        else
            buff[ind2]  = x[n1];
        
        __syncthreads();
        #pragma unroll
        for(int j=0; j < 4; j++)
            if ( inx >= (n-n1) )
                la[iny + j * 4][inx] =  0.f;
            else
                la[iny + j * 4][inx] = A[j* 4 * lda];
        
        __syncthreads();
        if ( n-n1 > 4 ) {
            #pragma unroll
            for(int j=0; j < 4; j++) {
                ind =  j+iny*4;
                res += la[inx][ind]*buff[ind];
            }
            A += 16;
            __syncthreads();
            #pragma unroll
            for(int j=0; j < 4; j++)
                if ( inx+16>=(n-n1) )
                    la[iny+ j * 4][inx] = 0.f;
                else
                    la[iny+ j * 4][inx] = A[j* 4* lda];
            
            __syncthreads();
            
            #pragma unroll
            for(int j=0; j < 4; j++) {
                ind = j+4*iny;
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
    ind = inx + blockIdx.x * 16;
    la[inx][iny] = res;
    __syncthreads();
    if ( ind < n && iny == 0 ) {
        res = la[inx][0] + la[inx][1] + la[inx][2] + la[inx][3];
        y[ind] = alpha*res + beta * y[ind];
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}

extern "C" void
magmablas_sgemvt1_fermi(
    magma_int_t m, magma_int_t n, float alpha,
    const float *A, magma_int_t lda,
    const float *x, float beta,
    float       *y)
{
    dim3 grid    ( 1, n, 1 );
    dim3 threads ( threadSize, 1, 1 );
    sgemvt_kernel1_fermi<<< grid, threads, 0, magma_stream >>>
        (m, n, alpha, (m / threadSize)*threadSize, A, lda, x, beta, y);
}

extern "C" void
magmablas_sgemvt2_fermi(
    magma_int_t m, magma_int_t n, float alpha,
    const float *A, magma_int_t lda,
    const float *x, float beta,
    float       *y)
{
    magma_int_t blocks = (n - 1)/16 + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(16, 4, 1);
    sgemvt_kernel2_fermi<<< grid, threads, 0, magma_stream >>>
        (m, n, alpha, (m / 32)*32, A, lda, x, beta, y);
}

extern "C" void
magmablas_sgemvt_fermi(
    magma_int_t m, magma_int_t n, float alpha,
    const float *A, magma_int_t lda,
    const float *x, float beta,
    float       *y)
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

    A       (input) REAL array of dimension ( LDA, n ) on the GPU.

    LDA     (input) INTEGER.
            LDA specifies the leading dimension of A.

    X       (input) REAL array of dimension m.

    Y       (output) REAL array of dimension n.
            On exit Y = alpha A^T X.

    ===================================================================== */

    magmablas_sgemvt1_fermi(m, n, alpha, A, lda, x, beta, y);
}

extern "C" void
magmablas_sgemv(
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
    3) y = alpha A^trans x + beta y
                            otherwise, using CUBLAS.

    Arguments
    ==========
    TRANS   CHARACTER*1
            On entry, TRANS specifies the operation to be performed as
            follows:
                TRANS = 'N' or 'n'   y := alpha*A  *x + beta*y
                TRANS = 'T' or 't'   y := alpha*A^T*x + beta*y

    M       (input) INTEGER
            On entry, m specifies the number of rows of the matrix A.

    N       (input) INTEGER
            On entry, n specifies the number of columns of the matrix A
 
    ALPHA   REAL
            On entry, ALPHA specifies the scalar alpha.
            Unchanged on exit.

    A       (input) REAL array of dimension ( LDA, n ) on the GPU.
   
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

    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // call CUDA ARCH 1.x version
        // magmablas for [sd] precisions, cublas for [zc] precisions.
        #if defined(PRECISION_z) || defined(PRECISION_c)
        cublasSgemv( trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
        #else
        magmablas_sgemv_tesla( trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
        #endif
        return;
    }
    
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    if ( incx == 1 && incy == 1 ) {
        if ( trans == 'n' || trans == 'N' )
            magmablas_sgemvn_fermi(m, n, alpha, A, lda, x, beta, y);
        else if (trans == 't' || trans == 'T' || trans == 'c' || trans == 'C')
            magmablas_sgemvt_fermi(m, n, alpha, A, lda, x, beta, y);
        else
            fprintf( stderr, "trans = %c is invalid\n", trans );
    }
    else {
        cublasSgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }
}

#undef num_threads
#undef gemv_bs
#undef threadSize
