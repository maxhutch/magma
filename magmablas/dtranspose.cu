/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from ztranspose.cu normal z -> d, Fri Jan 30 19:00:10 2015

       @author Stan Tomov
       @author Mark Gates
*/
#include "common_magma.h"

#define PRECISION_d

#if defined(PRECISION_z)
    #define NX 16
#else
    #define NX 32
#endif

#define NB 32
#define NY 8


// tile M-by-N matrix with ceil(M/NB) by ceil(N/NB) tiles sized NB-by-NB.
// uses NX-by-NY threads, where NB/NX, NB/NY, NX/NY evenly.
// subtile each NB-by-NB tile with (NB/NX) subtiles sized NX-by-NB
// for each subtile
//     load NX-by-NB subtile transposed from A into sA, as (NB/NY) blocks sized NX-by-NY
//     save NB-by-NX subtile from sA into AT,   as (NB/NX)*(NX/NY) blocks sized NX-by-NY
//     A  += NX
//     AT += NX*ldat
//
// e.g., with NB=32, NX=32, NY=8 ([sdc] precisions)
//     load 32x32 subtile as 4   blocks of 32x8 columns: (A11  A12  A13  A14 )
//     save 32x32 subtile as 1*4 blocks of 32x8 columns: (AT11 AT12 AT13 AT14)
//
// e.g., with NB=32, NX=16, NY=8 (z precision)
//     load 16x32 subtile as 4   blocks of 16x8 columns: (A11  A12  A13  A14)
//     save 32x16 subtile as 2*2 blocks of 16x8 columns: (AT11 AT12)
//                                                       (AT21 AT22)
static __device__ void
dtranspose_device(
    int m, int n,
    const double *A, int lda,
    double *AT,      int ldat)
{
    __shared__ double sA[NB][NX+1];

    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int ibx = blockIdx.x*NB;
    int iby = blockIdx.y*NB;
    int i, j;
    
    A  += ibx + tx + (iby + ty)*lda;
    AT += iby + tx + (ibx + ty)*ldat;
    
    #pragma unroll
    for( int tile=0; tile < NB/NX; ++tile ) {
        // load NX-by-NB subtile transposed from A into sA
        i = ibx + tx + tile*NX;
        j = iby + ty;
        if (i < m) {
            #pragma unroll
            for( int j2=0; j2 < NB; j2 += NY ) {
                if (j + j2 < n) {
                    sA[ty + j2][tx] = A[j2*lda];
                }
            }
        }
        __syncthreads();
        
        // save NB-by-NX subtile from sA into AT
        i = iby + tx;
        j = ibx + ty + tile*NX;
        #pragma unroll
        for( int i2=0; i2 < NB; i2 += NX ) {
            if (i + i2 < n) {
                #pragma unroll
                for( int j2=0; j2 < NX; j2 += NY ) {
                    if (j + j2 < m) {
                        AT[i2 + j2*ldat] = sA[tx + i2][ty + j2];
                    }
                }
            }
        }
        __syncthreads();
        
        // move to next subtile
        A  += NX;
        AT += NX*ldat;
    }
}


/*
    kernel wrapper to call the device function.
*/
__global__
void dtranspose_kernel(
    int m, int n,
    const double *A, int lda,
    double *AT,      int ldat)
{
    dtranspose_device(m, n, A, lda, AT, ldat);
}

__global__
void dtranspose_kernel_batched(
    int m, int n,
    double **dA_array, int lda,
    double **dAT_array,      int ldat)
{
    int batchid = blockIdx.z;
    dtranspose_device(m, n, dA_array[batchid], lda, dAT_array[batchid], ldat);
}


/**
    Purpose
    -------
    dtranspose_q copies and transposes a matrix dA to matrix dAT.
    
    Same as dtranspose, but adds queue argument.
        
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA      DOUBLE_PRECISION array, dimension (LDDA,N)
            The M-by-N matrix dA.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= M.
    
    @param[in]
    dAT     DOUBLE_PRECISION array, dimension (LDDAT,M)
            The N-by-M matrix dAT.
    
    @param[in]
    lddat   INTEGER
            The leading dimension of the array dAT.  LDDAT >= N.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dtranspose_q(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA,  magma_int_t ldda,
    magmaDouble_ptr       dAT, magma_int_t lddat,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < m )
        info = -4;
    else if ( lddat < n )
        info = -6;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    dim3 threads( NX, NY );
    dim3 grid( (m+NB-1)/NB, (n+NB-1)/NB );
    dtranspose_kernel<<< grid, threads, 0, queue >>>
        ( m, n, dA, ldda, dAT, lddat );
}


/**
    @see magmablas_dtranspose_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dtranspose(
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA,  magma_int_t ldda,
    magmaDouble_ptr       dAT, magma_int_t lddat )
{
    magmablas_dtranspose_q( m, n, dA, ldda, dAT, lddat, magma_stream );
}




/**
    Purpose
    -------
    dtranspose_batched_q copies and transposes a matrix dA_array[i] to matrix dAT_array[i].
    
    Same as dtranspose_batched, but adds queue argument.
        
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA_array 
            DOUBLE_PRECISION* array, dimension (batchCount)
            array of pointers to the matrices dA, where each dA is of dimension (LDDA,N)
            The M-by-N matrix dA.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= M.
    
    @param[in]
    dAT_array     
            DOUBLE_PRECISION* array, dimension (batchCount)
            array of pointers to the matrices dAT, where each dAT is of dimension (LDDAT,M)
            The N-by-M matrix dAT.
    
    @param[in]
    lddat   INTEGER
            The leading dimension of the array dAT.  LDDAT >= N.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @param[in]
    batchCount  Number of matrices in dA_array and dAT_array

    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dtranspose_batched_q(
    magma_int_t m, magma_int_t n,
    double **dA_array,  magma_int_t ldda,
    double **dAT_array, magma_int_t lddat, magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < m )
        info = -4;
    else if ( lddat < n )
        info = -6;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    dim3 threads( NX, NY );
    dim3 grid( (m+NB-1)/NB, (n+NB-1)/NB, batchCount );
    dtranspose_kernel_batched<<< grid, threads, 0, queue >>>
        ( m, n, dA_array, ldda, dAT_array, lddat );
}


/**
    @see magmablas_dtranspose_batched_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dtranspose_batched(
    magma_int_t m, magma_int_t n,
    double **dA_array,  magma_int_t ldda,
    double **dAT_array, magma_int_t lddat, magma_int_t batchCount )
{
    magmablas_dtranspose_batched_q( m, n, dA_array, ldda, dAT_array, lddat, batchCount, magma_stream );
}

