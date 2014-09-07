/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from zlaswp.cu normal z -> d, Tue Sep  2 12:38:16 2014
       
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
    double *dAT;
    int n, lda, j0, npivots;
    int ipiv[MAX_PIVOTS];
} dlaswp_params_t;


// Matrix A is stored row-wise in dAT.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__global__ void dlaswp_kernel( dlaswp_params_t params )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if( tid < params.n ) {
        int lda = params.lda;
        double *dAT = params.dAT + tid + params.j0*lda;
        double *A1  = dAT;
        
        for( int i1 = 0; i1 < params.npivots; ++i1 ) {
            int i2 = params.ipiv[i1];
            double *A2 = dAT + i2*lda;
            double temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += lda;  // A1 = dA + i1*ldx
        }
    }
}


// Launch dlaswp kernel with ceil( n / NTHREADS ) blocks of NTHREADS threads each.
extern "C" void dlaswp_launch( dlaswp_params_t &params, magma_queue_t queue )
{
    int blocks = (params.n + NTHREADS - 1) / NTHREADS;
    dlaswp_kernel<<< blocks, NTHREADS, 0, queue >>>( params );
}


// @deprecated
// Swap rows of A, stored row-wise.
// This version updates each entry of ipiv by adding ind.
// (In contrast, LAPACK applies laswp, then updates ipiv.)
// It is used in dgetrf, dgetrf_gpu, dgetrf_mgpu, dgetrf_ooc.
extern "C" void
magmablas_dpermute_long2( magma_int_t n, double *dAT, magma_int_t lda,
                          magma_int_t *ipiv, magma_int_t nb, magma_int_t ind )
{
    for( int k = 0; k < nb; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, nb-k );
        // fields are:             dAT  n  lda  j0       npivots
        dlaswp_params_t params = { dAT, n, lda, ind + k, npivots };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[ind + k + j] - k - 1;
            ipiv[ind + k + j] += ind;
        }
        dlaswp_launch( params, magma_stream );
    }
}


// @deprecated
// Swap rows of A, stored row-wise.
// This version assumes ind has already been added to ipiv.
// (In contrast, LAPACK applies laswp, then updates ipiv.)
// It is used in dgetrf_mgpu, dgetrf_ooc.
extern "C" void
magmablas_dpermute_long3( double *dAT, magma_int_t lda,
                          const magma_int_t *ipiv, magma_int_t nb, magma_int_t ind )
{
    for( int k = 0; k < nb; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, nb-k );
        // fields are:             dAT  n    lda  j0       npivots
        dlaswp_params_t params = { dAT, lda, lda, ind + k, npivots };
        for( int j = 0; j < MAX_PIVOTS; ++j ) {
            params.ipiv[j] = ipiv[ind + k + j] - k - 1 - ind;
        }
        dlaswp_launch( params, magma_stream );
    }
}


/**
    Purpose:
    =============
    DLASWP performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored row-wise (hence dAT). **
    Otherwise, this is identical to LAPACK's interface.
    
    Arguments:
    ==========
    \param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    \param[in,out]
    dAT      DOUBLE PRECISION array on GPU, stored row-wise, dimension (LDA,N)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    \param[in]
    lda      INTEGER
             The leading dimension of the array A. lda >= n.
    
    \param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (Fortran one-based index: 1 <= k1 <= n.)
    
    \param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (Fortran one-based index: 1 <= k2 <= n.)
    
    \param[in]
    ipiv     INTEGER array, on CPU, dimension (K2*abs(INCI))
             The vector of pivot indices.  Only the elements in positions
             K1 through K2 of IPIV are accessed.
             IPIV(K) = L implies rows K and L are to be interchanged.
    
    \param[in]
    inci     INTEGER
             The increment between successive values of IPIV.
             Currently, IPIV > 0.
             TODO: If IPIV is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_daux2
    ********************************************************************/
// It is used in dgessm, dgetrf_incpiv.
extern "C" void
magmablas_dlaswp_q(
    magma_int_t n, double *dAT, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 1 || k1 > n )
        info = -4;
    else if ( k2 < 1 || k2 > n )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, k2-k );
        // fields are:             dAT        n  lda  j0 npivots
        dlaswp_params_t params = { dAT+k*lda, n, lda, 0, npivots };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        dlaswp_launch( params, queue );
    }
}


/**
    @see magmablas_dlaswp_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaswp( magma_int_t n, double *dAT, magma_int_t lda,
                  magma_int_t k1, magma_int_t k2,
                  const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_dlaswp_q( n, dAT, lda, k1, k2, ipiv, inci, magma_stream );
}






// ------------------------------------------------------------
// Extended version has stride in both directions (ldx, ldy)
// to handle both row-wise and column-wise storage.

typedef struct {
    double *dA;
    int n, ldx, ldy, j0, npivots;
    int ipiv[MAX_PIVOTS];
} dlaswpx_params_t;


// Matrix A is stored row or column-wise in dA.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__global__ void dlaswpx_kernel( dlaswpx_params_t params )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if( tid < params.n ) {
        int ldx = params.ldx;
        double *dA = params.dA + tid*params.ldy + params.j0*ldx;
        double *A1  = dA;
        
        for( int i1 = 0; i1 < params.npivots; ++i1 ) {
            int i2 = params.ipiv[i1];
            double *A2 = dA + i2*ldx;
            double temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += ldx;  // A1 = dA + i1*ldx
        }
    }
}


// Launch dlaswpx kernel with ceil( n / NTHREADS ) blocks of NTHREADS threads each.
extern "C" void dlaswpx( dlaswpx_params_t &params, magma_queue_t queue )
{
    int blocks = (params.n + NTHREADS - 1) / NTHREADS;
    dlaswpx_kernel<<< blocks, NTHREADS, 0, queue >>>( params );
}


/**
    Purpose:
    =============
    DLASWPX performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored either row-wise or column-wise,
       depending on ldx and ldy. **
    Otherwise, this is identical to LAPACK's interface.
    
    Arguments:
    ==========
    \param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    \param[in,out]
    dA       DOUBLE PRECISION array on GPU, dimension (*,*)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    \param[in]
    ldx      INTEGER
             Stride between elements in same column.
    
    \param[in]
    ldy      INTEGER
             Stride between elements in same row.
             For A stored row-wise,    set ldx=lda and ldy=1.
             For A stored column-wise, set ldx=1   and ldy=lda.
    
    \param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    ipiv     INTEGER array, on CPU, dimension (K2*abs(INCI))
             The vector of pivot indices.  Only the elements in positions
             K1 through K2 of IPIV are accessed.
             IPIV(K) = L implies rows K and L are to be interchanged.
    
    \param[in]
    inci     INTEGER
             The increment between successive values of IPIV.
             Currently, IPIV > 0.
             TODO: If IPIV is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaswpx_q(
    magma_int_t n, double *dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 0 )
        info = -4;  
    else if ( k2 < 0 || k2 < k1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, k2-k );
        // fields are:              dA        n  ldx  ldy  j0 npivots
        dlaswpx_params_t params = { dA+k*ldx, n, ldx, ldy, 0, npivots };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        dlaswpx( params, queue );
    }
}


/**
    @see magmablas_dlaswpx_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaswpx( magma_int_t n, double *dA, magma_int_t ldx, magma_int_t ldy,
                   magma_int_t k1, magma_int_t k2,
                   const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_dlaswpx_q( n, dA, ldx, ldy, k1, k2, ipiv, inci, magma_stream );
}






// ------------------------------------------------------------
// This version takes d_ipiv on the GPU. Thus it does not pass pivots
// as an argument using a structure, avoiding all the argument size
// limitations of CUDA and OpenCL. It also needs just one kernel launch
// with all the pivots, instead of multiple kernel launches with small
// batches of pivots. On Fermi, it is faster than magmablas_dlaswp
// (including copying pivots to the GPU).

__global__ void dlaswp2_kernel(
    int n, double *dAT, int lda, int npivots,
    const magma_int_t* d_ipiv, magma_int_t inci )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if( tid < n ) {
        dAT += tid;
        double *A1  = dAT;
        
        for( int i1 = 0; i1 < npivots; ++i1 ) {
            int i2 = d_ipiv[i1*inci] - 1;  // Fortran index
            double *A2 = dAT + i2*lda;
            double temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += lda;  // A1 = dA + i1*ldx
        }
    }
}


/**
    Purpose:
    =============
    DLASWP2 performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored row-wise (hence dAT). **
    Otherwise, this is identical to LAPACK's interface.
    
    Here, d_ipiv is passed in GPU memory.
    
    Arguments:
    ==========
    \param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    \param[in,out]
    dAT      DOUBLE PRECISION array on GPU, stored row-wise, dimension (LDA,*)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    \param[in]
    lda      INTEGER
             The leading dimension of the array A.
             (I.e., stride between elements in a column.)
    
    \param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    d_ipiv   INTEGER array, on GPU, dimension (K2*abs(INCI))
             The vector of pivot indices.  Only the elements in positions
             K1 through K2 of IPIV are accessed.
             IPIV(K) = L implies rows K and L are to be interchanged.
    
    \param[in]
    inci     INTEGER
             The increment between successive values of IPIV.
             Currently, IPIV > 0.
             TODO: If IPIV is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaswp2_q(
    magma_int_t n, double* dAT, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *d_ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 0 )
        info = -4;  
    else if ( k2 < 0 || k2 < k1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    int blocks = (n + NTHREADS - 1) / NTHREADS;
    dlaswp2_kernel<<< blocks, NTHREADS, 0, queue >>>(
        n, dAT + (k1-1)*lda, lda, k2-(k1-1), d_ipiv, inci );
}


/**
    @see magmablas_dlaswp2_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaswp2( magma_int_t n, double* dAT, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   const magma_int_t *d_ipiv, magma_int_t inci )
{
    magmablas_dlaswp2_q( n, dAT, lda, k1, k2, d_ipiv, inci, magma_stream );
}
