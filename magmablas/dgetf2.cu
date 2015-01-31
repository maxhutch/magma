/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zgetf2.cu normal z -> d, Fri Jan 30 19:00:10 2015
*/
#include "common_magma.h"

#define PRECISION_d

#define dswap_bs 64

//#if (GPUSHMEM < 200)
#define dger_bs 512  // 512 is max threads for 1.x cards
//#else
//#define dger_bs 1024
//#endif

void magma_dgetf2_swap(
    magma_int_t n, double *x, magma_int_t i, magma_int_t j, magma_int_t incx);

void magma_dscal_dger(
    magma_int_t m, magma_int_t n, double *A, magma_int_t lda);


/**
    DGETF2 computes an LU factorization of a general m-by-n matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    Arguments
    ---------

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0 and N <= 1024.
            On CUDA architecture 1.x cards, N <= 512.

    @param[in,out]
    A       DOUBLE_PRECISION array, dimension (LDA,N)
            On entry, the m by n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -k, the k-th argument had an illegal value
      -     > 0: if INFO = k, U(k,k) is exactly zero. The factorization
                 has been completed, but the factor U is exactly
                 singular, and division by zero will occur if it is used
                 to solve a system of equations.

    @ingroup magma_dgesv_aux
    ********************************************************************/
extern "C" magma_int_t
magma_dgetf2_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info )
{
    #define dA(i, j)  (dA + (i) + (j)*ldda)

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > dger_bs) {
        *info = -2;
    } else if (ldda < max(1,m)) {
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
        jp = j - 1 + magma_idamax(m-j, dA(j,j), 1);
        ipiv[j] = jp + 1;  // ipiv uses Fortran one-based index
        // Can't check value of dA since it is on GPU
        //if ( dA(jp, j) != 0.0) {
            cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
            
            // Apply the interchange to columns 1:N.
            if (jp != j) {
                magma_dgetf2_swap(n, dA, j, jp, ldda);
            }
            
            // Compute elements J+1:M of J-th column.
            if (j < m) {
                magma_dscal_dger(m-j, n-j, dA(j, j), ldda);
            }
        //}
        //else if (*info == 0) {
        //    *info = j;
        //}
    }

    return *info;
}


__global__
void kernel_dswap(int n, double *x, int i, int j, int incx)
{
    int id = blockIdx.x * dswap_bs + threadIdx.x;

    if (id < n) {
        double tmp = x[i + incx*id];
        x[i + incx*id] = x[j + incx*id];
        x[j + incx*id] = tmp;
    }
}


void magma_dgetf2_swap(magma_int_t n, double *x, magma_int_t i, magma_int_t j, magma_int_t incx)
{
/*
    dswap two row vectors: ith and jth
*/
    dim3 threads(dswap_bs, 1, 1);
    int num_blocks = (n - 1)/dswap_bs + 1;
    dim3 grid(num_blocks,1);
    kernel_dswap<<< grid, threads, 0, magma_stream >>>(n, x, i, j, incx);
}


// dynamically allocated shared memory, set to size n when the kernel is launched.
// See CUDA Guide B.2.3
extern __shared__ double shared_data[];

__global__
void kernel_dscal_dger(int m, int n, double *A, int lda)
{
    double *shared_y = shared_data;

    int tid = blockIdx.x * dger_bs + threadIdx.x;

    double reg = MAGMA_D_ZERO;

    if (threadIdx.x < n) {
        shared_y[threadIdx.x] = A[lda * threadIdx.x];
    }

    __syncthreads();

    if (tid < m && tid > 0) {
        reg = A[tid];

        reg *= MAGMA_D_DIV(MAGMA_D_ONE, shared_y[0]);

        A[tid] = reg;

        #pragma unroll
        for(int i=1; i < n; i++) {
            A[tid + i*lda] += (MAGMA_D_NEG_ONE) * shared_y[i] * reg;
        }
    }
}


void magma_dscal_dger(magma_int_t m, magma_int_t n, double *A, magma_int_t lda)
{
/*

    Specialized kernel which merged dscal and dger the two kernels
    1) dscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a dger Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where 
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);

*/
    dim3 threads(dger_bs, 1, 1);
    int num_blocks = (m - 1)/dger_bs + 1;
    dim3 grid(num_blocks,1);
    size_t shared_size = sizeof(double)*(n);
    kernel_dscal_dger<<< grid, threads, shared_size, magma_stream>>>(m, n, A, lda);
}
