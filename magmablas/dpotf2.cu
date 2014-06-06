/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
       
       @generated d Tue Dec 17 13:18:45 2013
*/

#include <stdio.h>
#include "common_magma.h"
#include "magmablas.h"

#define PRECISION_d

//#if (GPUSHMEM < 200)
#define ddot_max_bs 512  // 512 is max threads for 1.x cards
//#else
//#define ddot_max_bs 1024
//#endif

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

void dpotf2_dscal(magma_int_t n, double *x, magma_int_t incx);
void dpotf2_ddot(magma_int_t n, double *x, magma_int_t incx);

#if defined(PRECISION_z) || defined(PRECISION_c)
void dlacgv(magma_int_t n, double *x, magma_int_t incx);
#endif

extern "C" magma_int_t
magma_dpotf2_gpu(
    magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======

    dpotf2 computes the Cholesky factorization of a real symmetric
    positive definite matrix A.

    The factorization has the form
        A = U' * U , if UPLO = 'U', or
        A = L  * L', if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the unblocked version of the algorithm, calling Level 2 BLAS.

    Arguments
    =========

    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is stored.
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the matrix A.  N >= 0 and N <= 1024.
            On CUDA architecture 1.x cards, N <= 512.

    A       (input/output) DOUBLE_PRECISION array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            n by n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading n by n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U'*U  or A = L*L'.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value
            > 0: if INFO = k, the leading minor of order k is not
                 positive definite, and the factorization could not be
                 completed.

    ===================================================================== */

    magma_int_t j;

    *info = 0;
    if ( uplo != 'U' && uplo != 'u' && uplo != 'L' && uplo != 'l') {
        *info = -1;
    } else if (n < 0 || n > ddot_max_bs) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if (n == 0) {
        return *info;
    }

    double alpha = MAGMA_D_NEG_ONE;
    double beta  = MAGMA_D_ONE;

    if (uplo == 'U' || uplo == 'u') {
        for(j = 0; j < n; j++) {
            dpotf2_ddot(j, A(0,j), 1); // including ddot product and update a(j,j)
            if (j < n) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                dlacgv(j, A(0, j), 1);
                #endif
                cublasDgemv( MagmaTrans, j, n-j-1,
                             alpha, A(0, j+1), lda,
                                    A(0, j),   1,
                             beta,  A(j, j+1), lda); // cublas is better in upper case

                #if defined(PRECISION_z) || defined(PRECISION_c)
                dlacgv(j, A(0, j), 1);
                #endif
                dpotf2_dscal(n-j, A(j,j), lda);
            }
        }
    }
    else {
        for(j = 0; j < n; j++) {
            dpotf2_ddot(j, A(j,0), lda); // including ddot product and update a(j,j)
            if (j < n) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                dlacgv(j, A(j, 0), lda);
                #endif
                magmablas_dgemv( MagmaNoTrans, n-j-1, j,
                                 alpha, A(j+1, 0), lda,
                                        A(j,0),    lda,
                                 beta,  A(j+1, j), 1 );// magmablas is better in lower case

                #if defined(PRECISION_z) || defined(PRECISION_c)
                dlacgv(j, A(j, 0), lda);
                #endif
                dpotf2_dscal(n-j, A(j,j), 1);
            }
        }
    }

    return *info;
}

#define dscal_bs  32
#define ddot_bs  512
#define dlacgv_bs 512

// dynamically allocated shared memory, set to size number of threads when the kernel is launched.
// See CUDA Guide B.2.3
extern __shared__ double shared_data[];

__global__ void kernel_ddot(int n, double *x, int incx, int threadSize)
{
    int tx = threadIdx.x;

    double *sdata = shared_data;

    double res = MAGMA_D_ZERO;

    if (tx < n) {
       res = x[tx*incx];
    }

    sdata[tx] = MAGMA_D_REAL(res * MAGMA_D_CNJG(res));

    __syncthreads();

    for(int s = blockDim.x/2; s > 32; s >>= 1 ) {
        if (tx < s) {
            sdata[tx] += sdata[tx+s];
        }
        __syncthreads();
    }

    if (tx < 32) {
        volatile double* smem = sdata;
        smem[tx] += smem[tx+32];
        smem[tx] += smem[tx+16];
        smem[tx] += smem[tx+8];
        smem[tx] += smem[tx+4];
        smem[tx] += smem[tx+2];
        smem[tx] += smem[tx+1];
    }

    if (tx == 0) {
        double xreal = MAGMA_D_REAL(x[n*incx]);
        x[n*incx] = MAGMA_D_MAKE( sqrt(xreal - sdata[0]), 0 );
    }
}

void dpotf2_ddot(magma_int_t n, double *x, magma_int_t incx)
{
/*
    Specialized Ddot
    1) performs ddot sum = x[0:n-1]*(x[0:n-1])
    2) updates x[n] = sqrt(x[n]-sum);

*/
    if (n > ddot_max_bs) {
        printf("n = %d > %d is not supported in dpotf2_ddot\n", (int) n, (int) ddot_max_bs);
        exit(1);
    }
    int threadSize;

    if (n <= 1024 && n > 512) {
        threadSize = 1024;
    }
    else if (n <= 512 && n > 256 ) {
        threadSize = 512;
    }
    else if (n <= 256 && n > 128) {
        threadSize = 256;
    }
    else if (n <= 128 && n > 64) {
        threadSize = 128;
    }
    else {
        threadSize = 64;
    }

    kernel_ddot<<< 1, threadSize, threadSize * sizeof(double), magma_stream>>> (n, x, incx, threadSize);
}

__global__ void kernel_dscal(int n, double *x, int incx)
{
    int id = blockIdx.x * dscal_bs + threadIdx.x;

    __shared__ double factor;

    if (threadIdx.x == 0) {
        factor = MAGMA_D_MAKE(1.0/MAGMA_D_REAL(x[0]), 0.0);
    }

    __syncthreads();

    if ( id < n && id >0) {
        x[id*incx] = x[id*incx] * factor;
    }
}


void dpotf2_dscal(magma_int_t n, double *x, magma_int_t incx)
{
/*
    Specialized Zdscal perform x[1:n-1]/x[0]

*/
    dim3 threads(dscal_bs, 1, 1);
    int num_blocks = (n - 1)/dscal_bs + 1;
    dim3 grid(num_blocks,1);
    kernel_dscal<<< grid, threads, 0, magma_stream >>> (n, x, incx);
}


#if defined(PRECISION_z) || defined(PRECISION_c)

__global__ void kernel_dlacgv(int n, double *x, int incx)
{
    int id = blockIdx.x * dlacgv_bs + threadIdx.x;

    if ( id < n ) {
        x[id*incx] = MAGMA_D_CNJG(x[id*incx]);
    }
}


void dlacgv(magma_int_t n, double *x, magma_int_t incx)
{
/*
    Purpose
    =======

    DLACGV conjugates a real vector of length N.

    Arguments
    =========

    N       (input) INTEGER
            The length of the vector X.  N >= 0.

    X       (input/output) DOUBLE PRECISION array, dimension
                           (1+(N-1)*abs(INCX))
            On entry, the vector of length N to be conjugated.
            On exit, X is overwritten with conjg(X).

    INCX    (input) INTEGER
            The spacing between successive elements of X.

    ===================================================================== */

    dim3 threads(dlacgv_bs, 1, 1);
    int num_blocks = (n - 1)/dlacgv_bs + 1;
    dim3 grid(num_blocks,1);
    kernel_dlacgv<<< grid, threads, 0, magma_stream >>> (n, x, incx);
}

#endif // defined(PRECISION_z) || defined(PRECISION_c)
