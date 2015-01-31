/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zclaswp.cu mixed zc -> ds, Fri Jan 30 19:00:08 2015

*/
#include "common_magma.h"

#define NB 64

// TODO check precision, as in dlag2s?

__global__ void
dslaswp_kernel(int n, double *A, int lda, float *SA, int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    float res;
    
    if (ind < m) {
        SA   += ind;
        ipiv += ind;
        
        newind = ipiv[0];
        
        for(int i=0; i < n; i++) {
            res = MAGMA_S_MAKE( (float)(A[newind+i*lda]),
                                (float)(A[newind+i*lda]) );
            SA[i*lda] = res; 
        }
    }
}

__global__ void
dslaswp_inv_kernel(int n, double *A, int lda, float *SA, int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    double res;

    if (ind < m) {
        A    += ind;
        ipiv += ind;

        newind = ipiv[0];

        for(int i=0; i < n; i++) {
            res = MAGMA_D_MAKE( (double)(SA[newind+i*lda]),
                                (double)(SA[newind+i*lda]) );
            A[i*lda] = res;
        }
    }
}


/**
    Purpose
    -------
    Row i of  A is cast to single precision in row ipiv[i] of SA (incx > 0), or
    row i of SA is cast to double precision in row ipiv[i] of  A (incx < 0),
    for 0 <= i < M.

    @param[in]
    n       INTEGER.
            On entry, N specifies the number of columns of the matrix A.

    @param[in,out]
    A       DOUBLE PRECISION array on the GPU, dimension (LDA,N)
            On entry, the M-by-N matrix to which the row interchanges will be applied.
            TODO update docs

    @param[in]
    lda     INTEGER.
            LDA specifies the leading dimension of A.

    @param[in,out]
    SA      REAL array on the GPU, dimension (LDA,N)
            On exit, the single precision, permuted matrix.
            TODO update docs
        
    @param[in]
    m       The number of rows to be interchanged.

    @param[in]
    ipiv    INTEGER array on the GPU, dimension (M)
            The vector of pivot indices. Row i of A is cast to single 
            precision in row ipiv[i] of SA, for 0 <= i < m. 

    @param[in]
    incx    INTEGER
            If INCX is negative, the pivots are applied in reverse order,
            otherwise in straight-forward order.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dslaswp_q(
    magma_int_t n,
    magmaDouble_ptr A, magma_int_t lda,
    magmaFloat_ptr SA, magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx,
    magma_queue_t queue )
{
    int blocks = (m - 1)/NB + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(NB, 1, 1);

    if (incx >= 0)
        dslaswp_kernel<<< grid, threads, 0, queue >>>(n, A, lda, SA, m, ipiv);
    else
        dslaswp_inv_kernel<<< grid, threads, 0, queue >>>(n, A, lda, SA, m, ipiv);
}


/**
    @see magmablas_dslaswp_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dslaswp(
    magma_int_t n,
    magmaDouble_ptr A, magma_int_t lda,
    magmaFloat_ptr SA, magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx )
{
    magmablas_dslaswp_q( n, A, lda, SA, m, ipiv, incx, magma_stream );
}
