/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from clag2z.cu mixed zc -> ds, Fri Jan 30 19:00:07 2015
       @author Mark Gates
*/
#include "common_magma.h"

#define PRECISION_d

#define BLK_X 64
#define BLK_Y 32


/*
    Divides matrix into ceil( m/BLK_X ) x ceil( n/BLK_Y ) blocks.
    Each block has BLK_X threads.
    Each thread loops across one row, updating BLK_Y entries.
    
    Code similar to slat2d and zlaset.
*/
__global__
void slag2d_kernel(
    int m, int n,
    const float *SA, int ldsa,
    double       *A, int lda )
{
    int ind = blockIdx.x*BLK_X + threadIdx.x;
    int iby = blockIdx.y*BLK_Y;
    /* check if full block-column */
    bool full = (iby + BLK_Y <= n);
    /* do only rows inside matrix */
    if ( ind < m ) {
        A  += ind + iby*lda;
        SA += ind + iby*ldsa;
        if ( full ) {
            // full block-column
            #pragma unroll
            for( int j=0; j < BLK_Y; ++j ) {
                A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ), MAGMA_S_IMAG( SA[j*ldsa] ));
            }
        }
        else {
            // partial block-column
            for( int j=0; j < BLK_Y && iby+j < n; ++j ) {
                A[j*lda] = MAGMA_D_MAKE( MAGMA_S_REAL( SA[j*ldsa] ), MAGMA_S_IMAG( SA[j*ldsa] ));
            }
        }
    }
}


/**
    Purpose
    -------
    SLAG2D_STREAM converts a single-real matrix, SA,
                        to a double-real matrix, A.

    Note that while it is possible to overflow while converting
    from double to single, it is not possible to overflow when
    converting from single to double.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of lines of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in]
    SA      REAL array, dimension (LDSA,N)
            On entry, the M-by-N coefficient matrix SA.

    @param[in]
    ldsa    INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).

    @param[out]
    A       DOUBLE PRECISION array, dimension (LDA,N)
            On exit, the M-by-N coefficient matrix A.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slag2d_q(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr       A, magma_int_t lda,
    magma_int_t *info,
    magma_queue_t queue)
{
    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( ldsa < max(1,m) )
        *info = -4;
    else if ( lda < max(1,m) )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return; //*info;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    dim3 threads( BLK_X, 1 );
    dim3 grid( (m+BLK_X-1)/BLK_X, (n+BLK_Y-1)/BLK_Y );
    slag2d_kernel<<< grid, threads, 0, queue >>> ( m, n, SA, ldsa, A, lda );
}


/**
    @see magmablas_slag2d_q
    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slag2d(
    magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr SA, magma_int_t ldsa,
    magmaDouble_ptr       A, magma_int_t lda,
    magma_int_t *info)
{
    magmablas_slag2d_q( m, n, SA, ldsa, A, lda, info, magma_stream );
}
