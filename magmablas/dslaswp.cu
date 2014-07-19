/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zclaswp.cu mixed zc -> ds, Fri Jul 18 17:34:12 2014

*/
#include "common_magma.h"

#define NB 64

__global__ void
dslaswp_kernel(int n, double *a, int lda, float *sa, int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    float res;
    
    if (ind < m) {
        sa   += ind;
        ipiv += ind;
        
        newind = ipiv[0];
        
        for(int i=0; i < n; i++) {
            res = MAGMA_S_MAKE( (float)(a[newind+i*lda]),
                                (float)(a[newind+i*lda]) );
            sa[i*lda] = res; 
        }
    }
}

__global__ void
dslaswp_inv_kernel(int n, double *a, int lda, float *sa, int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    double res;

    if (ind < m) {
        a    += ind;
        ipiv += ind;

        newind = ipiv[0];

        for(int i=0; i < n; i++) {
            res = MAGMA_D_MAKE( (double)(sa[newind+i*lda]),
                                (double)(sa[newind+i*lda]) );
            a[i*lda] = res;
        }
    }
}


/**
    Purpose
    -------
    Row i of A is cast to single precision in row ipiv[i] of SA, for 0 <= i < M.

    N      - (input) INTEGER.
             On entry, N specifies the number of columns of the matrix A.

    A      - (input) DOUBLE PRECISION array on the GPU, dimension (LDA,N)
             On entry, the M-by-N matrix to which the row interchanges will be applied.

    LDA    - (input) INTEGER.
             LDA specifies the leading dimension of A.

    SA     - (output) REAL array on the GPU, dimension (LDA,N)
             On exit, the single precision, permuted matrix.
        
    M      - (input) The number of rows to be interchanged.

    IPIV   - (input) INTEGER array on the GPU, dimension (M)
             The vector of pivot indices. Row i of A is cast to single 
             precision in row ipiv[i] of SA, for 0 <= i < m. 

    INCX   - (input) INTEGER
             If IPIV is negative, the pivots are applied in reverse 
             order, otherwise in straight-forward order.

    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dslaswp( magma_int_t n, double *a, magma_int_t lda,
                   float *sa, magma_int_t m,
                   const magma_int_t *ipiv, magma_int_t incx )
{
    int blocks = (m - 1)/NB + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(NB, 1, 1);

    if (incx >= 0)
        dslaswp_kernel<<< grid, threads, 0, magma_stream >>>(n, a, lda, sa, m, ipiv);
    else
        dslaswp_inv_kernel<<< grid, threads, 0, magma_stream >>>(n, a, lda, sa, m, ipiv);
}
