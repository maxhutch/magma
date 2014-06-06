/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal z -> s d c
*/

#include <stdio.h>
#include "common_magma.h"
#include "magmablas.h"

#define PRECISION_z

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

#define zswap_bs 64

//#if (GPUSHMEM < 200)
#define zgeru_bs 512  // 512 is max threads for 1.x cards
//#else
//#define zgeru_bs 1024
//#endif

void magma_zswap(
    magma_int_t n, magmaDoubleComplex *x, magma_int_t i, magma_int_t j, magma_int_t incx);

void magma_zscal_zgeru(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda);


extern "C" magma_int_t
magma_zgetf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t* info )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    ZGETF2 computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    Arguments
    =========

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0 and N <= 1024.
            On CUDA architecture 1.x cards, N <= 512.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the m by n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value
            > 0: if INFO = k, U(k,k) is exactly zero. The factorization
                 has been completed, but the factor U is exactly
                 singular, and division by zero will occur if it is used
                 to solve a system of equations.

    ===================================================================== */

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > zgeru_bs) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return *info;
    }

    magma_int_t min_mn = min(m, n);
    magma_int_t j, jp;
    
    for( j=0; j < min_mn; j++ ) {
        cudaDeviceSetCacheConfig( cudaFuncCachePreferShared );

        // Find pivot and test for singularity.
        jp = j - 1 + cublasIzamax(m-j, A(j,j), 1);
        ipiv[j] = jp + 1;  // ipiv uses Fortran one-based index
        // Can't check value of A since it is on GPU
        //if ( A(jp, j) != 0.0) {
            cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
            
            // Apply the interchange to columns 1:N.
            // TODO: replace with pre-existing BLAS-standard zswap routine,
            // e.g., magmablas_zswap or cublasZswap
            if (jp != j) {
                magma_zswap(n, A, j, jp, lda);
                //magmablas_zswap( n, A(j,0), lda, A(jp,0), lda );
                //cublasZswap( n, A(j,0), lda, A(jp,0), lda );
            }
            
            // Compute elements J+1:M of J-th column.
            if (j < m) {
                magma_zscal_zgeru(m-j, n-j, A(j, j), lda);
            }
        //}
        //else if (*info == 0) {
        //    *info = j;
        //}
    }

    return *info;
}


__global__
void kernel_zswap(int n, magmaDoubleComplex *x, int i, int j, int incx)
{
    int id = blockIdx.x * zswap_bs + threadIdx.x;

    if (id < n) {
        magmaDoubleComplex tmp = x[i + incx*id];
        x[i + incx*id] = x[j + incx*id];
        x[j + incx*id] = tmp;
    }
}


void magma_zswap(magma_int_t n, magmaDoubleComplex *x, magma_int_t i, magma_int_t j, magma_int_t incx)
{
/*
    zswap two row vectors: ith and jth
*/
    dim3 threads(zswap_bs, 1, 1);
    int num_blocks = (n - 1)/zswap_bs + 1;
    dim3 grid(num_blocks,1);
    kernel_zswap<<< grid, threads, 0, magma_stream >>>(n, x, i, j, incx);
}


// dynamically allocated shared memory, set to size n when the kernel is launched.
// See CUDA Guide B.2.3
extern __shared__ magmaDoubleComplex shared_data[];

__global__
void kernel_zscal_zgeru(int m, int n, magmaDoubleComplex *A, int lda)
{
    magmaDoubleComplex *shared_y = shared_data;

    int tid = blockIdx.x * zgeru_bs + threadIdx.x;

    magmaDoubleComplex reg = MAGMA_Z_ZERO;

    if (threadIdx.x < n) {
        shared_y[threadIdx.x] = A[lda * threadIdx.x];
    }

    __syncthreads();

    if (tid < m && tid > 0) {
        reg = A[tid];

        reg *= MAGMA_Z_DIV(MAGMA_Z_ONE, shared_y[0]);

        A[tid] = reg;

        #pragma unroll
        for(int i=1; i < n; i++) {
            A[tid + i*lda] += (MAGMA_Z_NEG_ONE) * shared_y[i] * reg;
        }
    }
}


void magma_zscal_zgeru(magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda)
{
/*

    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where 
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);

*/
    dim3 threads(zgeru_bs, 1, 1);
    int num_blocks = (m - 1)/zgeru_bs + 1;
    dim3 grid(num_blocks,1);
    size_t shared_size = sizeof(magmaDoubleComplex)*(n);
    kernel_zscal_zgeru<<< grid, threads, shared_size, magma_stream>>>(m, n, A, lda);
}
