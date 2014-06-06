/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:45 2013
*/

#include <stdio.h>
#include "common_magma.h"
#include "magmablas.h"

#define PRECISION_c

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

#define cswap_bs 64

//#if (GPUSHMEM < 200)
#define cgeru_bs 512  // 512 is max threads for 1.x cards
//#else
//#define cgeru_bs 1024
//#endif

void magma_cswap(
    magma_int_t n, magmaFloatComplex *x, magma_int_t i, magma_int_t j, magma_int_t incx);

void magma_cscal_cgeru(
    magma_int_t m, magma_int_t n, magmaFloatComplex *A, magma_int_t lda);


extern "C" magma_int_t
magma_cgetf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t* info )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    CGETF2 computes an LU factorization of a general m-by-n matrix A
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

    A       (input/output) COMPLEX array, dimension (LDA,N)
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
    } else if (n < 0 || n > cgeru_bs) {
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
        jp = j - 1 + cublasIcamax(m-j, A(j,j), 1);
        ipiv[j] = jp + 1;  // ipiv uses Fortran one-based index
        // Can't check value of A since it is on GPU
        //if ( A(jp, j) != 0.0) {
            cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
            
            // Apply the interchange to columns 1:N.
            // TODO: replace with pre-existing BLAS-standard cswap routine,
            // e.g., magmablas_cswap or cublasCswap
            if (jp != j) {
                magma_cswap(n, A, j, jp, lda);
                //magmablas_cswap( n, A(j,0), lda, A(jp,0), lda );
                //cublasCswap( n, A(j,0), lda, A(jp,0), lda );
            }
            
            // Compute elements J+1:M of J-th column.
            if (j < m) {
                magma_cscal_cgeru(m-j, n-j, A(j, j), lda);
            }
        //}
        //else if (*info == 0) {
        //    *info = j;
        //}
    }

    return *info;
}


__global__
void kernel_cswap(int n, magmaFloatComplex *x, int i, int j, int incx)
{
    int id = blockIdx.x * cswap_bs + threadIdx.x;

    if (id < n) {
        magmaFloatComplex tmp = x[i + incx*id];
        x[i + incx*id] = x[j + incx*id];
        x[j + incx*id] = tmp;
    }
}


void magma_cswap(magma_int_t n, magmaFloatComplex *x, magma_int_t i, magma_int_t j, magma_int_t incx)
{
/*
    cswap two row vectors: ith and jth
*/
    dim3 threads(cswap_bs, 1, 1);
    int num_blocks = (n - 1)/cswap_bs + 1;
    dim3 grid(num_blocks,1);
    kernel_cswap<<< grid, threads, 0, magma_stream >>>(n, x, i, j, incx);
}


// dynamically allocated shared memory, set to size n when the kernel is launched.
// See CUDA Guide B.2.3
extern __shared__ magmaFloatComplex shared_data[];

__global__
void kernel_cscal_cgeru(int m, int n, magmaFloatComplex *A, int lda)
{
    magmaFloatComplex *shared_y = shared_data;

    int tid = blockIdx.x * cgeru_bs + threadIdx.x;

    magmaFloatComplex reg = MAGMA_C_ZERO;

    if (threadIdx.x < n) {
        shared_y[threadIdx.x] = A[lda * threadIdx.x];
    }

    __syncthreads();

    if (tid < m && tid > 0) {
        reg = A[tid];

        reg *= MAGMA_C_DIV(MAGMA_C_ONE, shared_y[0]);

        A[tid] = reg;

        #pragma unroll
        for(int i=1; i < n; i++) {
            A[tid + i*lda] += (MAGMA_C_NEG_ONE) * shared_y[i] * reg;
        }
    }
}


void magma_cscal_cgeru(magma_int_t m, magma_int_t n, magmaFloatComplex *A, magma_int_t lda)
{
/*

    Specialized kernel which merged cscal and cgeru the two kernels
    1) cscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a cgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where 
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);

*/
    dim3 threads(cgeru_bs, 1, 1);
    int num_blocks = (m - 1)/cgeru_bs + 1;
    dim3 grid(num_blocks,1);
    size_t shared_size = sizeof(magmaFloatComplex)*(n);
    kernel_cscal_cgeru<<< grid, threads, shared_size, magma_stream>>>(m, n, A, lda);
}
