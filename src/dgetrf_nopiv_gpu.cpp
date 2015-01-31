/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zgetrf_nopiv_gpu.cpp normal z -> d, Fri Jan 30 19:00:13 2015

*/
#include "common_magma.h"

/**
    Purpose
    -------
    DGETRF_NOPIV_GPU computes an LU factorization of a general M-by-N
    matrix A without any pivoting.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      DOUBLE_PRECISION array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @ingroup magma_dgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dgetrf_nopiv_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info)
{
#define dA(i,j) (dA + (i)*nb + (j)*nb*ldda)

    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;

    magma_int_t iinfo, nb;
    magma_int_t maxm, mindim;
    magma_int_t i, rows, s, lddwork;
    double *work;

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    mindim = min(m, n);
    nb     = magma_get_dgetrf_nb(m);
    s      = mindim / nb;

    if (nb <= 1 || nb >= min(m,n)) {
        /* Use CPU code. */
        magma_dmalloc_cpu( &work, m * n );
        if ( work == NULL ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        magma_dgetmatrix( m, n, dA, ldda, work, m );
        magma_dgetrf_nopiv( m, n, work, m, info);
        magma_dsetmatrix( m, n, work, m, dA, ldda );
        magma_free_cpu(work);
    }
    else {
        /* Use hybrid blocked code. */
        maxm = ((m + 31)/32)*32;

        lddwork = maxm;

        if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, maxm*nb )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }

        /* Define user stream if current stream is NULL */
        magma_queue_t stream[2];
        
        magma_queue_t orig_stream;
        magmablasGetKernelStream( &orig_stream );

        magma_queue_create( &stream[0] );
        if (orig_stream == NULL) {
            magma_queue_create( &stream[1] );
            magmablasSetKernelStream(stream[1]);
        }
        else {
            stream[1] = orig_stream;
        }

        for( i=0; i < s; i++ ) {
            // download i-th panel
            magma_queue_sync( stream[1] );
            magma_dgetmatrix_async( m-i*nb, nb, dA(i,i), ldda, work, lddwork, stream[0] );
            
            if ( i > 0 ) {
                magma_dtrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                             nb, n - (i+1)*nb,
                             c_one, dA(i-1,i-1), ldda,
                             dA(i-1,i+1), ldda );
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             m-i*nb, n-(i+1)*nb, nb,
                             c_neg_one, dA(i,  i-1), ldda, dA(i-1,i+1), ldda,
                             c_one,     dA(i,  i+1), ldda );
            }

            // do the cpu part
            rows = m - i*nb;
            magma_queue_sync( stream[0] );
            magma_dgetrf_nopiv( rows, nb, work, lddwork, &iinfo );
            if ( (*info == 0) && (iinfo > 0) )
                *info = iinfo + i*nb;

            // upload i-th panel
            magma_dsetmatrix_async( m-i*nb, nb, work, lddwork, dA(i, i), ldda, stream[0] );
            magma_queue_sync( stream[0] );

            // do the small non-parallel computations
            if ( s > (i+1) ) {
                magma_dtrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dA(i, i  ), ldda,
                             dA(i, i+1), ldda);
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             m-(i+1)*nb, nb, nb,
                             c_neg_one, dA(i+1, i  ), ldda, dA(i,   i+1), ldda,
                             c_one,     dA(i+1, i+1), ldda );
            }
            else {
                magma_dtrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                             nb, n-s*nb,
                             c_one, dA(i, i  ), ldda,
                             dA(i, i+1), ldda);
                magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                             m-(i+1)*nb, n-(i+1)*nb, nb,
                             c_neg_one, dA(i+1, i  ), ldda, dA(i,   i+1), ldda,
                             c_one,     dA(i+1, i+1), ldda );
            }
        }

        magma_int_t nb0 = min(m - s*nb, n - s*nb);
        rows = m - s*nb;
        magma_dgetmatrix( rows, nb0, dA(s,s), ldda, work, lddwork );

        // make sure that gpu queue is empty
        magma_device_sync();

        // do the cpu part
        magma_dgetrf_nopiv( rows, nb0, work, lddwork, &iinfo );
        if ( (*info == 0) && (iinfo > 0) )
            *info = iinfo + s*nb;

        // upload i-th panel
        magma_dsetmatrix( rows, nb0, work, lddwork, dA(s,s), ldda );

        magma_dtrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                     nb0, n-s*nb-nb0,
                     c_one, dA(s,s),     ldda,
                            dA(s,s)+nb0, ldda);

        magma_free_pinned( work );

        magma_queue_destroy( stream[0] );
        if (orig_stream == NULL) {
            magma_queue_destroy( stream[1] );
        }
        magmablasSetKernelStream( orig_stream );
    }

    return *info;
} /* magma_dgetrf_nopiv_gpu */

#undef dA
