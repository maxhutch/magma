/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal z -> s d c
       
       @author Stan Tomov
       @author Mathieu Faverge
       @author Ichitaro Yamazaki
       @author Mark Gates
*/
#include "common_magma.h"

// MAX_PIVOTS is maximum number of pivots to apply in each kernel launch
// NTHREADS is number of threads in a block
// 64 and 256 are better on Kepler; 
//#define MAX_PIVOTS 64
//#define NTHREADS   256
#define MAX_PIVOTS 32
#define NTHREADS   64

typedef struct {
    magmaDoubleComplex *dAT;
    int n, lda, j0, npivots;
    int ipiv[MAX_PIVOTS];
} zlaswp_params_t;


// Matrix A is stored row-wise in dAT.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__global__ void zlaswp_kernel( zlaswp_params_t params )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if( tid < params.n ) {
        int lda = params.lda;
        magmaDoubleComplex *dAT = params.dAT + tid + params.j0*lda;
        magmaDoubleComplex *A1  = dAT;
        
        for( int i1 = 0; i1 < params.npivots; ++i1 ) {
            int i2 = params.ipiv[i1];
            magmaDoubleComplex *A2 = dAT + i2*lda;
            magmaDoubleComplex temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += lda;  // A1 = dA + i1*ldx
        }
    }
}


// Launch zlaswp kernel with ceil( n / NTHREADS ) blocks of NTHREADS threads each.
extern "C" void zlaswp_launch( zlaswp_params_t &params )
{
    int blocks = (params.n + NTHREADS - 1) / NTHREADS;
    zlaswp_kernel<<< blocks, NTHREADS, 0, magma_stream >>>( params );
}


// Swap rows of A, stored row-wise.
// This version updates each entry of ipiv by adding ind.
// It is used in zgetrf, zgetrf_gpu, zgetrf_mgpu, zgetrf_ooc.
extern "C" void
magmablas_zpermute_long2( magma_int_t n, magmaDoubleComplex *dAT, magma_int_t lda,
                          magma_int_t *ipiv, magma_int_t nb, magma_int_t ind )
{
    for( int k = 0; k < nb; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, nb-k );
        // fields are:             dAT  n  lda  j0       npivots
        zlaswp_params_t params = { dAT, n, lda, ind + k, npivots };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[ind + k + j] - k - 1;
            ipiv[ind + k + j] += ind;
        }
        zlaswp_launch( params );
    }
}


// Swap rows of A, stored row-wise.
// This version assumes ind has already been added to ipiv.
// It is used in zgetrf_mgpu, zgetrf_ooc.
extern "C" void
magmablas_zpermute_long3( magmaDoubleComplex *dAT, magma_int_t lda,
                          const magma_int_t *ipiv, magma_int_t nb, magma_int_t ind )
{
    for( int k = 0; k < nb; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, nb-k );
        // fields are:             dAT  n    lda  j0       npivots
        zlaswp_params_t params = { dAT, lda, lda, ind + k, npivots };
        for( int j = 0; j < MAX_PIVOTS; ++j ) {
            params.ipiv[j] = ipiv[ind + k + j] - k - 1 - ind;
        }
        zlaswp_launch( params );
    }
}


// Swap rows of A, stored row-wise.
// This interface is identical to LAPACK's laswp interface.
// It is used in zgessm, zgetrf_incpiv.
extern "C" void
magmablas_zlaswp( magma_int_t n, magmaDoubleComplex *dAT, magma_int_t lda,
                  magma_int_t i1, magma_int_t i2,
                  const magma_int_t *ipiv, magma_int_t inci )
{
    for( int k = i1-1; k < i2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, i2-k );
        // fields are:             dAT        n  lda  j0 npivots
        zlaswp_params_t params = { dAT+k*lda, n, lda, 0, npivots };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        zlaswp_launch( params );
    }
}


// ------------------------------------------------------------
// Extended version has stride in both directions (ldx, ldy)
// to handle both row-wise and column-wise storage.

typedef struct {
    magmaDoubleComplex *dA;
    int n, ldx, ldy, j0, npivots;
    int ipiv[MAX_PIVOTS];
} zlaswpx_params_t;


// Matrix A is stored row-wise in dA.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__global__ void zlaswpx_kernel( zlaswpx_params_t params )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if( tid < params.n ) {
        int ldx = params.ldx;
        magmaDoubleComplex *dA = params.dA + tid*params.ldy + params.j0*ldx;
        magmaDoubleComplex *A1  = dA;
        
        for( int i1 = 0; i1 < params.npivots; ++i1 ) {
            int i2 = params.ipiv[i1];
            magmaDoubleComplex *A2 = dA + i2*ldx;
            magmaDoubleComplex temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += ldx;  // A1 = dA + i1*ldx
        }
    }
}


// Launch zlaswpx kernel with ceil( n / NTHREADS ) blocks of NTHREADS threads each.
extern "C" void zlaswpx( zlaswpx_params_t &params )
{
    int blocks = (params.n + NTHREADS - 1) / NTHREADS;
    zlaswpx_kernel<<< blocks, NTHREADS, 0, magma_stream >>>( params );
}


// Swap rows of A.
// For A stored row-wise,    set ldx=lda and ldy=1.
// For A stored column-wise, set ldx=1   and ldy=lda.
// Otherwise, this interface is identical to LAPACK's laswp interface.
extern "C" void
magmablas_zlaswpx( magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldx, magma_int_t ldy,
                   magma_int_t i1, magma_int_t i2,
                   const magma_int_t *ipiv, magma_int_t inci )
{
    for( int k = i1-1; k < i2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, i2-k );
        // fields are:              dA        n  ldx  ldy  j0 npivots
        zlaswpx_params_t params = { dA+k*ldx, n, ldx, ldy, 0, npivots };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        zlaswpx( params );
    }
}


// ------------------------------------------------------------
// This version takes d_ipiv on the GPU. Thus it does not pass pivots
// as an argument using a structure, avoiding all the argument size
// limitations of CUDA and OpenCL. It also needs just one kernel launch
// with all the pivots, instead of multiple kernel launches with small
// batches of pivots. On Fermi, it is faster than magmablas_zlaswp
// (including copying pivots to the GPU).

__global__ void zlaswp2_kernel( int n, magmaDoubleComplex *dAT, int lda, int npivots, const magma_int_t* d_ipiv )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if( tid < n ) {
        dAT += tid;
        magmaDoubleComplex *A1  = dAT;
        
        for( int i1 = 0; i1 < npivots; ++i1 ) {
            int i2 = d_ipiv[i1] - 1;  // Fortran index
            magmaDoubleComplex *A2 = dAT + i2*lda;
            magmaDoubleComplex temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += lda;  // A1 = dA + i1*ldx
        }
    }
}

// Swap rows of A, stored row-wise.
// d_ipiv is vector of pivots stored on the GPU,
// unlike magmablas_zlaswp where ipiv is stored on the CPU.
// This interface is identical to LAPACK's laswp interface.
extern "C" void
magmablas_zlaswp2( magma_int_t n, magmaDoubleComplex* dAT, magma_int_t lda,
                   magma_int_t i1, magma_int_t i2,
                   const magma_int_t *d_ipiv )
{
    int blocks = (n + NTHREADS - 1) / NTHREADS;
    zlaswp2_kernel<<< blocks, NTHREADS, 0, magma_stream >>>(
        n, dAT + (i1-1)*lda, lda, i2-(i1-1), d_ipiv );
}
