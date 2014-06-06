/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from clag2z.cu mixed zc -> ds, Fri May 30 10:40:38 2014
       @author Mark Gates
*/
#include "common_magma.h"

#define blksize 64

__global__ void
slag2d_array(  int m, int n,
               const float *SA, int ldsa,
               double       *A, int lda )
{
    int i = blockIdx.x*blksize + threadIdx.x;
    if ( i < m ) {
        A  += i;
        SA += i;
        const double *Aend = A + lda*n;
        while( A < Aend ) {
            *A = (double)( *SA );
            A  += lda;
            SA += ldsa;
        }
    }
}


__global__ void
slag2d_vector( int m,
               const float *SA,
               double       *A )
{
    int i = blockIdx.x*blksize + threadIdx.x;
    if ( i < m ) {
        A  += i;
        SA += i;
        *A = (double)( *SA );
    }
}


/**
    Purpose
    -------
    SLAG2D converts a single-real matrix, SA,
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

    @ingroup magma_saux2
    ********************************************************************/
extern "C" void
magmablas_slag2d(
    magma_int_t m, magma_int_t n,
    const float *SA, magma_int_t ldsa,
    double       *A, magma_int_t lda,
    magma_int_t *info)
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
        //return *info;
    }

    /* quick return */
    if ( m == 0 || n == 0 ) {
        return;
    }

    dim3 threads( blksize );
    dim3 grid( (m+blksize-1)/blksize );
    if( n > 1 ) {
        slag2d_array<<< grid, threads, 0, magma_stream >>> ( m, n, SA, ldsa, A, lda );
    }
    else{
        slag2d_vector<<< grid, threads, 0, magma_stream >>> ( m, SA, A );
    }
}
