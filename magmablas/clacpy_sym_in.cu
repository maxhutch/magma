/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Mark Gates
       @author Azzam Haidar
       
       @generated from magmablas/zlacpy_sym_in.cu normal z -> c, Mon May  2 23:30:31 2016

*/
#include "magma_internal.h"

#define BLK_X 64
#define BLK_Y 32

/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.

    Code similar to claset, clacpy, clag2z, clag2z, cgeadd.
*/
static __device__
void clacpy_sym_in_full_device(
    int m, int n,
    const magmaFloatComplex *dA, int ldda,
    magmaFloatComplex       *dB, int lddb )
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
    Similar to clacpy_full, but updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.

    Code similar to claset, clacpy, zlat2c, clat2z.
*/
static __device__
void clacpy_sym_in_lower_device(
    int m, int n, magma_int_t *rows, magma_int_t *perm,
    const magmaFloatComplex *dA, int ldda,
    magmaFloatComplex       *dB, int lddb )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x; // row
    int iby = blockIdx.y*BLK_Y;               // col

    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n);
    for (int jj=0; jj < n; jj++) {
        perm[rows[2*jj+1]] = rows[2*jj];
    }
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < m ) {
        if ( full ) {
            // full block-column, off-diagonal block
            //#pragma unroll
            for( int jj=0; jj < BLK_Y; ++jj ) 
            {
                int j = rows[2*(iby+jj)];
                if (perm[ind] <= j)
                    dB[ind + (iby+jj)*lddb] = MAGMA_C_CONJ( dA[j + perm[ind]*ldda] );
                else
                    dB[ind + (iby+jj)*lddb] = dA[perm[ind] + j*ldda];
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int jj=0; jj < BLK_Y && iby+jj < n; ++jj ) 
            {
                int j = rows[2*(iby+jj)];
                if (perm[ind] <= j)
                    dB[ind + (iby+jj)*lddb] = MAGMA_C_CONJ( dA[j + perm[ind]*ldda] );
                else
                    dB[ind + (iby+jj)*lddb] = dA[perm[ind] + j*ldda];
            }
        }
    }
}


/*
    Similar to clacpy_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.

    Code similar to claset, clacpy, zlat2c, clat2z.
*/
static __device__
void clacpy_sym_in_upper_device(
    int m, int n,
    const magmaFloatComplex *dA, int ldda,
    magmaFloatComplex       *dB, int lddb )
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


//////////////////////////////////////////////////////////////////////////////////////
/*
    kernel wrappers to call the device functions.
*/
__global__
void clacpy_sym_in_full_kernel(
    int m, int n,
    const magmaFloatComplex *dA, int ldda,
    magmaFloatComplex       *dB, int lddb )
{
    clacpy_sym_in_full_device(m, n, dA, ldda, dB, lddb);
}

__global__
void clacpy_sym_in_lower_kernel(
    int m, int n, magma_int_t *rows, magma_int_t *perm,
    const magmaFloatComplex *dA, int ldda,
    magmaFloatComplex       *dB, int lddb )
{
    clacpy_sym_in_lower_device(m, n, rows, perm, dA, ldda, dB, lddb);
}

__global__
void clacpy_sym_in_upper_kernel(
    int m, int n,
    const magmaFloatComplex *dA, int ldda,
    magmaFloatComplex       *dB, int lddb )
{
    clacpy_sym_in_upper_device(m, n, dA, ldda, dB, lddb);
}


/**
    Purpose
    -------
    CLACPY_Q copies all or part of a two-dimensional matrix dA to another
    matrix dB.
    
    This is the same as CLACPY, but adds queue argument.
    
    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be copied to dB.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
      -     = MagmaFull:       All of the matrix dA
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of rows that are swapped.  N >= 0.
 
    @param[in]
    rows     INTEGER array, on GPU, dimension (2*n)
             On entry, it stores the new pivots such that rows[i]-th and rows[n+i]-th
             rows are swapped.

    @param[in,out]
    perm     INTEGER array, on GPU, dimension (m)
             On entry, it stores the identity permutation array.
             On exit, it is updated with the new pivots given by rows such that
             i-th row will be the original perm[i]-th row after the pivots are applied. 

    @param[in]
    dA      COMPLEX array, dimension (LDDA,N)
            The M-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle or trapezoid is accessed;
            if UPLO = MagmaLower, only the lower triangle or trapezoid is accessed.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[out]
    dB      COMPLEX array, dimension (LDDB,N)
            On exit, dB = stores the columns after the pivots are applied.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_caux2
    ********************************************************************/
extern "C" void
magmablas_clacpy_sym_in_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < max(1,m))
        info = -5;
    else if ( lddb < max(1,m))
        info = -7;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    if ( m == 0 || n == 0 ) {
        return;
    }
    
    dim3 threads( BLK_X, 1 );
    dim3 grid( magma_ceildiv(m, BLK_X), magma_ceildiv(n, BLK_Y) );
    
    if ( uplo == MagmaLower ) {
        clacpy_sym_in_lower_kernel<<< grid, threads, 0, queue->cuda_stream() >>> ( m, n, rows, perm, dA, ldda, dB, lddb );
    }
    else if ( uplo == MagmaUpper ) {
        clacpy_sym_in_upper_kernel<<< grid, threads, 0, queue->cuda_stream() >>> ( m, n, dA, ldda, dB, lddb );
    }
    else {
        clacpy_sym_in_full_kernel <<< grid, threads, 0, queue->cuda_stream() >>> ( m, n, dA, ldda, dB, lddb );
    }
}
