/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Stan Tomov

       @generated from magmablas/zgemv_conj.cu normal z -> s, Mon May  2 23:30:29 2016
*/
#include "magma_internal.h"
#include "commonblas_s.h"

#define num_threads 256


__global__ void
sgemv_conj_kernel(
    int m, int n, float alpha,
    const float * __restrict__ A, int lda,
    const float * __restrict__ x, int incx, float beta,
    float *       __restrict__ y, int incy)
{
    int ind = blockIdx.x*num_threads + threadIdx.x;
    
    A += ind;

    if ( ind < m ) {
        float res = MAGMA_S_ZERO;
        
        #pragma unroll
        for( int i=0; i < n; i ++ ) {
            res += A[0] * MAGMA_S_CONJ(x[0]);
            A += lda;
            x += incx;
        }
        
        y[ind*incy] = alpha * res + beta * y[ind*incy];
    }
}


/**
    Purpose
    -------
    SGEMV_CONJ performs the matrix-vector operation
    
        y := alpha*A*conj(x)    + beta*y, 
    
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ----------
    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of the matrix A

    @param[in]
    alpha   REAL
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA      REAL array of dimension ( LDDA, n ) on the GPU.

    @param[in]
    ldda    INTEGER
            LDDA specifies the leading dimension of A.

    @param[in]
    dx      REAL array of dimension n

    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.

    @param[in]
    beta    DOUBLE REAL
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy      REAL array of dimension m

    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_sblas2
    ********************************************************************/
extern "C" void
magmablas_sgemv_conj_q(
    magma_int_t m, magma_int_t n, float alpha,
    magmaFloat_const_ptr dA, magma_int_t ldda,
    magmaFloat_const_ptr dx, magma_int_t incx,
    float beta,
    magmaFloat_ptr dy, magma_int_t incy,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < m )
        info = -5;
    else if ( incx == 0 )
        info = -7;
    else if ( incy == 0 )
        info = -10;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    magma_int_t blocks = magma_ceildiv( m, num_threads );
    dim3 grid(blocks, 1, 1);
    dim3 threads(num_threads, 1, 1);

    sgemv_conj_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
            (m, n, alpha, dA, ldda, dx, incx, beta, dy, incy);
}
