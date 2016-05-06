/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/zclaswp.cu mixed zc -> ds, Mon May  2 23:30:34 2016

*/
#include "magma_internal.h"

#define NB 64

// TODO check precision, as in dlag2s?

__global__ void
dslaswp_kernel(
    int n,
    double *A, int lda,
    float *SA, int ldsa,
    int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    float res;
    
    if (ind < m) {
        SA   += ind;
        ipiv += ind;
        
        newind = ipiv[0];
        
        for (int i=0; i < n; i++) {
            res = MAGMA_S_MAKE( (float)MAGMA_D_REAL( A[newind+i*lda] ),
                                (float)MAGMA_D_IMAG( A[newind+i*lda] ));
            SA[i*ldsa] = res; 
        }
    }
}

__global__ void
dslaswp_inv_kernel(
    int n,
    double *A, int lda,
    float *SA, int ldsa,
    int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    double res;

    if (ind < m) {
        A    += ind;
        ipiv += ind;

        newind = ipiv[0];

        for (int i=0; i < n; i++) {
            res = MAGMA_D_MAKE( (double)MAGMA_S_REAL( SA[newind+i*ldsa] ),
                                (double)MAGMA_S_IMAG( SA[newind+i*ldsa] ));
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
    SA      REAL array on the GPU, dimension (LDSA,N)
            On exit, the single precision, permuted matrix.
            TODO update docs

    @param[in]
    ldsa    INTEGER.
            LDSA specifies the leading dimension of SA.
        
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
    magmaFloat_ptr SA, magma_int_t ldsa,
    magma_int_t m,
    const magma_int_t *ipiv, magma_int_t incx,
    magma_queue_t queue )
{
    int blocks = magma_ceildiv( m, NB );
    dim3 grid( blocks );
    dim3 threads( NB );

    if (incx >= 0)
        dslaswp_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n, A, lda, SA, ldsa, m, ipiv);
    else
        dslaswp_inv_kernel<<< grid, threads, 0, queue->cuda_stream() >>>(n, A, lda, SA, ldsa, m, ipiv);
}
