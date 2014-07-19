/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @author Mark Gates
       
       @generated from zlaset.cu normal z -> d, Fri Jul 18 17:34:12 2014

*/
#include "common_magma.h"

#define BLK_X 64
#define BLK_Y 32

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.
*/
__global__
void dlaset_full(
    int m, int n,
    double offdiag, double diag,
    double *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* full block-column && (below diag || above diag || offdiag == diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y || ind + BLK_X <= iby || offdiag == diag));
    /* do only rows inside matrix */
    if ( ind < m ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block or offdiag == diag
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else
                    A[j*lda] = offdiag;
            }
        }
    }
}


/*
    Similar to dlaset_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.
*/
__global__
void dlaset_lower(
    int m, int n,
    double offdiag, double diag,
    double *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m && ind + BLK_X > iby ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else if ( ind > iby+j )
                    A[j*lda] = offdiag;
            }
        }
    }
}


/*
    Similar to dlaset_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.
*/
__global__
void dlaset_upper(
    int m, int n,
    double offdiag, double diag,
    double *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < m && ind < iby + BLK_Y ) {
        A += ind + iby*lda;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = offdiag;
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( iby+j == ind )
                    A[j*lda] = diag;
                else if ( ind < iby+j )
                    A[j*lda] = offdiag;
            }
        }
    }
}


/**
    Purpose
    -------
    DLASET_STREAM initializes a 2-D array A to DIAG on the diagonal and
    OFFDIAG on the off-diagonals.
    
    This is the same as DLASET, but adds stream argument.
    
    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be set.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:         All of the matrix dA is set.
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    offdiag DOUBLE_PRECISION
            The scalar OFFDIAG. (In LAPACK this is called ALPHA.)
    
    @param[in]
    diag    DOUBLE_PRECISION
            The scalar DIAG. (In LAPACK this is called BETA.)
    
    @param[in]
    dA      DOUBLE_PRECISION array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
            On exit, A(i,j) = OFFDIAG, 1 <= i <= m, 1 <= j <= n, i != j;
                     A(i,i) = DIAG,    1 <= i <= min(m,n)
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    stream  magma_queue_t
            Stream to execute in.
    
    @ingroup magma_daux2
    ********************************************************************/
extern "C"
void magmablas_dlaset_stream(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    double *dA, magma_int_t ldda,
    magma_queue_t stream)
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m) )
        info = -7;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    dim3 threads( BLK_X, 1 );
    dim3 grid( (m-1)/BLK_X + 1, (n-1)/BLK_Y + 1 );
    
    if (uplo == MagmaLower)
        dlaset_lower<<< grid, threads, 0, stream >>> (m, n, offdiag, diag, dA, ldda);
    else if (uplo == MagmaUpper)
        dlaset_upper<<< grid, threads, 0, stream >>> (m, n, offdiag, diag, dA, ldda);
    else
        dlaset_full <<< grid, threads, 0, stream >>> (m, n, offdiag, diag, dA, ldda);
}


/**
    @see magmablas_dlaset_stream
    @ingroup magma_daux2
    ********************************************************************/
extern "C"
void magmablas_dlaset(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    double offdiag, double diag,
    double *dA, magma_int_t ldda )
{
    magmablas_dlaset_stream( uplo, m, n, offdiag, diag, dA, ldda, magma_stream );
}


#define LASET_BAND_NB 64

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting the k-1 super-diagonals to OFFDIAG
    and the main diagonal to DIAG.
    Divides matrix into min( ceil((m+k-1)/nb), ceil(n/nb) ) block-columns,
    with k threads in each block.
    Each thread iterates across one diagonal.
    Thread k-1 does the main diagonal, thread k-2 the first super-diagonal, etc.

      block 0           block 1
      0                           => skip above matrix
      1 0                         => skip above matrix
      2 1 0                       => skip above matrix
    [ 3 2 1 0         |         ]
    [   3 2 1 0       |         ]
    [     3 2 1 0     |         ]
    [       3 2 1 0   |         ]
    [         3 2 1 0 |         ]
    [           3 2 1 | 0       ]
    [             3 2 | 1 0     ]
    [               3 | 2 1 0   ]
    [                 | 3 2 1 0 ]
    [                 |   3 2 1 ]
                      |     3 2   => skip below matrix
                              3   => skip below matrix
    
    Thread assignment for m=10, n=12, k=4, nb=8. Each column is done in parallel.
    
    @author Raffaele Solca
    @author Mark Gates
 */
__global__
void dlaset_band_upper(
    int m, int n,
    double offdiag, double diag,
    double *A, int lda)
{
    int k   = blockDim.x;
    int ibx = blockIdx.x * LASET_BAND_NB;
    int ind = ibx + threadIdx.x - k + 1;
    
    A += ind + ibx*lda;
    
    double value = offdiag;
    if (threadIdx.x == k-1)
        value = diag;

    #pragma unroll
    for (int j=0; j < LASET_BAND_NB; j++) {
        if (ibx + j < n && ind + j >= 0 && ind + j < m) {
            A[j*(lda+1)] = value;
        }
    }
}

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting the k-1 sub-diagonals to OFFDIAG
    and the main diagonal to DIAG.
    Divides matrix into min( ceil(m/nb), ceil(n/nb) ) block-columns,
    with k threads in each block.
    Each thread iterates across one diagonal.
    Thread 0 does the main diagonal, thread 1 the first sub-diagonal, etc.
    
      block 0           block 1
    [ 0               |         ]
    [ 1 0             |         ]
    [ 2 1 0           |         ]
    [ 3 2 1 0         |         ]
    [   3 2 1 0       |         ]
    [     3 2 1 0     |         ]
    [       3 2 1 0   |         ]
    [         3 2 1 0 |         ]
    [           3 2 1 | 0       ]
    [             3 2 | 1 0     ]
    [               3 | 2 1 0   ]
    [                   3 2 1 0 ]
    [                     3 2 1 ]
                            3 2   => skip below matrix
                              3   => skip below matrix
    
    Thread assignment for m=13, n=12, k=4, nb=8. Each column is done in parallel.
    
    @author Raffaele Solca
    @author Mark Gates
 */

__global__
void dlaset_band_lower(
    int m, int n,
    double offdiag, double diag,
    double *A, int lda)
{
    //int k   = blockDim.x;
    int ibx = blockIdx.x * LASET_BAND_NB;
    int ind = ibx + threadIdx.x;
    
    A += ind + ibx*lda;
    
    double value = offdiag;
    if (threadIdx.x == 0)
        value = diag;

    #pragma unroll
    for (int j=0; j < LASET_BAND_NB; j++) {
        if (ibx + j < n && ind + j < m) {
            A[j*(lda+1)] = value;
        }
    }
}


/**
    Purpose
    -------
    DLASET_BAND_STREAM initializes the main diagonal of dA to DIAG,
    and the K-1 sub- or super-diagonals to OFFDIAG.
    
    This is the same as DLASET_BAND, but adds stream argument.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be set.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix dA.  N >= 0.
    
    @param[in]
    k       INTEGER
            The number of diagonals to set, including the main diagonal.  K >= 0.
            Currently, K <= 1024 due to CUDA restrictions (max. number of threads per block).
    
    @param[in]
    offdiag DOUBLE_PRECISION
            Off-diagonal elements in the band are set to OFFDIAG.
    
    @param[in]
    diag    DOUBLE_PRECISION
            All the main diagonal elements are set to DIAG.
    
    @param[in]
    dA      DOUBLE_PRECISION array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
            On exit, A(i,j) = ALPHA, 1 <= i <= m, 1 <= j <= n where i != j, abs(i-j) < k;
                     A(i,i) = BETA , 1 <= i <= min(m,n)
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    stream  magma_queue_t
            Stream to execute DLASET in.
    
    @author Raffaele Solca
    @author Mark Gates
    
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaset_band_stream(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    double offdiag, double diag,
    double *dA, magma_int_t ldda, magma_queue_t stream)
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 || k > 1024 )
        info = -4;
    else if ( ldda < max(1,m) )
        info = -6;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if (uplo == MagmaUpper) {
        dim3 threads( min(k,n) );
        dim3 grid( (min(m+k-1,n) - 1)/LASET_BAND_NB + 1 );
        dlaset_band_upper<<< grid, threads, 0, stream >>> (m, n, offdiag, diag, dA, ldda);
}
    else if (uplo == MagmaLower) {
        dim3 threads( min(k,m) );
        dim3 grid( (min(m,n) - 1)/LASET_BAND_NB + 1 );
        dlaset_band_lower<<< grid, threads, 0, stream >>> (m, n, offdiag, diag, dA, ldda);
    }
}


/**
    @see magmablas_dlaset_band_stream
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_dlaset_band(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    double offdiag, double diag,
    double *dA, magma_int_t ldda)
{
    magmablas_dlaset_band_stream(uplo, m, n, k, offdiag, diag, dA, ldda, magma_stream);
}
