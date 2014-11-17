/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from zlacpy.cu normal z -> d, Sat Nov 15 19:53:59 2014
       @author Mark Gates
       @author Azzam Haidar
*/
#include "common_magma.h"

#define BLK_X 64
#define BLK_Y 32

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to dlaset.
*/
static __device__
void dlacpy_full_device(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}


/*
    Similar to dlacpy_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.

    Code similar to dlaset.
*/
static __device__
void dlacpy_lower_device(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m && ind + BLK_X > iby ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n && ind >= iby+j; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
    }
}


/*
    Similar to dlacpy_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to dlaset.
*/
static __device__
void dlacpy_upper_device(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < m && ind < iby + BLK_Y ) {
        dA += ind + iby*ldda;
        dB += ind + iby*lddb;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                dB[j*lddb] = dA[j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( ind <= iby+j ) {
                    dB[j*lddb] = dA[j*ldda];
                }
            }
        }
    }
}

/*
    kernel wrapper to call the device function.
*/
__global__
void dlacpy_full_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    dlacpy_full_device(m, n, dA, ldda, dB, lddb);
}

__global__
void dlacpy_lower_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    dlacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

__global__
void dlacpy_upper_kernel(
    int m, int n,
    const double *dA, int ldda,
    double       *dB, int lddb )
{
    dlacpy_upper_device(m, n, dA, ldda, dB, lddb);
}


/*
    kernel wrapper to call the device function for the batched routine.
*/
__global__
void dlacpy_full_kernel_batched(
    int m, int n,
    double const * const *dAarray, int ldda,
    double **dBarray, int lddb )
{
    int batchid = blockIdx.z;
    dlacpy_full_device(m, n, dAarray[batchid], ldda, dBarray[batchid], lddb);
}

__global__
void dlacpy_lower_kernel_batched(
    int m, int n,
    double const * const *dAarray, int ldda,
    double **dBarray, int lddb )
{
    int batchid = blockIdx.z;
    dlacpy_lower_device(m, n, dAarray[batchid], ldda, dBarray[batchid], lddb);
}

__global__
void dlacpy_upper_kernel_batched(
    int m, int n,
    double const * const *dAarray, int ldda,
    double **dBarray, int lddb )
{
    int batchid = blockIdx.z;
    dlacpy_upper_device(m, n, dAarray[batchid], ldda, dBarray[batchid], lddb);
}


/**
    Purpose
    -------
    DLACPY_Q copies all or part of a two-dimensional matrix dA to another
    matrix dB.
    
    This is the same as DLACPY, but adds queue argument.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be copied to dB.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:  All of the matrix dA
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA      DOUBLE_PRECISION array, dimension (LDDA,N)
            The m by n matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[out]
    dB      DOUBLE_PRECISION array, dimension (LDDB,N)
            The m by n matrix dB.
            On exit, dB = dA in the locations specified by UPLO.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlacpy_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || n == 0 )
        return;
    
    dim3 threads( BLK_X, 1 );
    dim3 grid( (m + BLK_X - 1)/BLK_X, (n + BLK_Y - 1)/BLK_Y );
    
    if ( uplo == MagmaLower ) {
        dlacpy_lower_kernel<<< grid, threads, 0, queue >>> ( m, n, dA, ldda, dB, lddb );
    }
    else if ( uplo == MagmaUpper ) {
        dlacpy_upper_kernel<<< grid, threads, 0, queue >>> ( m, n, dA, ldda, dB, lddb );
    }
    else {
        dlacpy_full_kernel <<< grid, threads, 0, queue >>> ( m, n, dA, ldda, dB, lddb );
    }
}

/**
    @see magmablas_dlacpy_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlacpy(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr       dB, magma_int_t lddb )
{
    magmablas_dlacpy_q( uplo, m, n, dA, ldda, dB, lddb, magma_stream );
}


/**
    Purpose
    -------
    DLACPY_BATCHED_Q copies all or part of each two-dimensional matrix
    dAarray[i] to matrix dBarray[i], for 0 <= i < batchcount.
    
    This is the same as DLACPY_BATCHED, but adds queue argument.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of each matrix dA to be copied to dB.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:  All of each matrix dA
    
    @param[in]
    m       INTEGER
            The number of rows of each matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of each matrix dA.  N >= 0.
    
    @param[in]
    dAarray DOUBLE_PRECISION* array, dimension (batchCount)
            array of pointers to the matrices dA, where each dA is of dimension (LDDA,N)
            The m by n matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of each array dA.  LDDA >= max(1,M).
    
    @param[out]
    dBarray DOUBLE_PRECISION* array, dimension (batchCount)
            array of pointers to the matrices dB, where each dB is of dimension (LDDB,N)
            The m by n matrix dB.
            On exit, dB = dA in the locations specified by UPLO.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of each array dB.  LDDB >= max(1,M).
    
    @param[in]
    batchCount  Number of matrices in dAarray and dBarray.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlacpy_batched_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const dAarray[], magma_int_t ldda,
    magmaDouble_ptr             dBarray[], magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    else if ( batchCount < 0 )
        info = -8;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || n == 0 || batchCount == 0 )
        return;
    
    dim3 threads( BLK_X, 1, 1 );
    dim3 grid( (m + BLK_X - 1)/BLK_X, (n + BLK_Y - 1)/BLK_Y, batchCount );
    
    if ( uplo == MagmaLower ) {
        dlacpy_lower_kernel_batched<<< grid, threads, 0, queue >>> ( m, n, dAarray, ldda, dBarray, lddb );
    }
    else if ( uplo == MagmaUpper ) {
        dlacpy_upper_kernel_batched<<< grid, threads, 0, queue >>> ( m, n, dAarray, ldda, dBarray, lddb );
    }
    else {
        dlacpy_full_kernel_batched <<< grid, threads, 0, queue >>> ( m, n, dAarray, ldda, dBarray, lddb );
    }
}


/**
    @see magmablas_dlacpy_batched_q
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlacpy_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr const dAarray[], magma_int_t ldda,
    magmaDouble_ptr             dBarray[], magma_int_t lddb,
    magma_int_t batchCount )
{
    magmablas_dlacpy_batched_q( uplo, m, n, dAarray, ldda, dBarray, lddb, batchCount, magma_stream );
}
