/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @precisions mixed zc -> ds

*/
#include "common_magma.h"

#define NB 64

__global__ void
zclaswp_kernel(int n, magmaDoubleComplex *a, int lda, magmaFloatComplex *sa, int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    magmaFloatComplex res;
    
    if (ind < m) {
        sa   += ind;
        ipiv += ind;
        
        newind = ipiv[0];
        
        for(int i=0; i < n; i++) {
            res = MAGMA_C_MAKE( (float)cuCreal(a[newind+i*lda]),
                                (float)cuCimag(a[newind+i*lda]) );
            sa[i*lda] = res; 
        }
    }
}

__global__ void
zclaswp_inv_kernel(int n, magmaDoubleComplex *a, int lda, magmaFloatComplex *sa, int m, const magma_int_t *ipiv)
{
    int ind = blockIdx.x*NB + threadIdx.x;
    int newind;
    magmaDoubleComplex res;

    if (ind < m) {
        a    += ind;
        ipiv += ind;

        newind = ipiv[0];

        for(int i=0; i < n; i++) {
            res = MAGMA_Z_MAKE( (double)cuCrealf(sa[newind+i*lda]),
                                (double)cuCimagf(sa[newind+i*lda]) );
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

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zclaswp( magma_int_t n, magmaDoubleComplex *a, magma_int_t lda,
                   magmaFloatComplex *sa, magma_int_t m,
                   const magma_int_t *ipiv, magma_int_t incx )
{
    int blocks = (m - 1)/NB + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(NB, 1, 1);

    if (incx >= 0)
        zclaswp_kernel<<< grid, threads, 0, magma_stream >>>(n, a, lda, sa, m, ipiv);
    else
        zclaswp_inv_kernel<<< grid, threads, 0, magma_stream >>>(n, a, lda, sa, m, ipiv);
}
