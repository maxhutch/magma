/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from magmablas/clat2z.cu mixed zc -> ds, Mon May  2 23:30:34 2016
       @author Mark Gates
*/
#include "magma_internal.h"

#define BLK_X 64
#define BLK_Y 32


/*
    Divides matrix into ceil( n/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.
    Updates only the diagonal and below.
    Blocks that are fully above the diagonal exit immediately.
    
    Code similar to dlag2s and zlaset.
*/
__global__
void slat2d_lower(
    int n,
    const float *SA, int ldsa,
    double      *A,  int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (below diag) */
    bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
    /* do only rows inside matrix, and blocks not above diag */
    if ( ind < n && ind + BLK_X > iby ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ),
                                         MAGMA_S_IMAG( SA[j*ldsa] ) );
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n && ind >= iby+j; ++j ) {
                A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ),
                                         MAGMA_S_IMAG( SA[j*ldsa] ) );
            }
        }
    }
}


/*
    Similar to slat2d_full, but updates only the diagonal and above.
    Blocks that are fully below the diagonal exit immediately.
    
    Code similar to dlag2s and zlaset.
*/
__global__
void slat2d_upper(
    int n,
    const float *SA, int ldsa,
    double      *A,  int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column && (above diag) */
    bool full = (iby + BLK_Y <= n && (ind + BLK_X <= iby));
    /* do only rows inside matrix, and blocks not below diag */
    if ( ind < n && ind < iby + BLK_Y ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column, off-diagonal block
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ),
                                         MAGMA_S_IMAG( SA[j*ldsa] ) );
            }
        }
        else {
            // either partial block-column or diagonal block
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                if ( ind <= iby+j ) {
                    A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ),
                                             MAGMA_S_IMAG( SA[j*ldsa] ) );
                }
            }
        }
    }
}


/**
    Purpose
    -------
    SLAT2D converts a single-real matrix, SA,
                 to a double-real matrix, A.

    Note that while it is possible to overflow while converting
    from double to single, it is not possible to overflow when
    converting from single to double.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix A to be converted.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  n >= 0.
    
    @param[in]
    A       DOUBLE PRECISION array, dimension (LDA,n)
            On entry, the n-by-n coefficient matrix A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,n).
    
    @param[out]
    SA      SINGLE PRECISION array, dimension (LDSA,n)
            On exit, if INFO=0, the n-by-n coefficient matrix SA;
            if INFO > 0, the content of SA is unspecified.
    
    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,n).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_daux2
    ********************************************************************/
extern "C" void
magmablas_slat2d_q(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr      A,  magma_int_t lda,
    magma_queue_t queue,
    magma_int_t *info )
{
    *info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( lda < max(1,n) )
        *info = -4;
    else if ( ldsa < max(1,n) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return; //*info;
    }

    /* quick return */
    if ( n == 0 ) {
        return;
    }
    
    dim3 threads( BLK_X, 1 );
    dim3 grid( magma_ceildiv( n, BLK_X ), magma_ceildiv( n, BLK_Y ) );
    
    if (uplo == MagmaLower) {
        slat2d_lower<<< grid, threads, 0, queue->cuda_stream() >>> (n, SA, ldsa, A, lda);
    }
    else if (uplo == MagmaUpper) {
        slat2d_upper<<< grid, threads, 0, queue->cuda_stream() >>> (n, SA, ldsa, A, lda);
    }
}
