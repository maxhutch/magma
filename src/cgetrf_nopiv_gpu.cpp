/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Stan Tomov
       
       @generated from src/zgetrf_nopiv_gpu.cpp normal z -> c, Mon May  2 23:30:03 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    CGETRF_NOPIV_GPU computes an LU factorization of a general M-by-N
    matrix A without any pivoting.

    The factorization has the form
        A = L * U
    where L is lower triangular with unit
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
    dA      COMPLEX array on the GPU, dimension (LDDA,N).
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

    @ingroup magma_cgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_cgetrf_nopiv_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    #ifdef HAVE_clBLAS
    #define  dA(i_, j_) dA,  (dA_offset  + (i_)*nb       + (j_)*nb*ldda)
    #else
    #define  dA(i_, j_) (dA  + (i_)*nb       + (j_)*nb*ldda)
    #endif

    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;

    magma_int_t iinfo, nb;
    magma_int_t maxm, mindim;
    magma_int_t j, rows, s, ldwork;
    magmaFloatComplex *work;

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
    mindim = min( m, n );
    nb     = magma_get_cgetrf_nb( m, n );
    s      = mindim / nb;

    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );

    if (nb <= 1 || nb >= min(m,n)) {
        /* Use CPU code. */
        if ( MAGMA_SUCCESS != magma_cmalloc_cpu( &work, m*n )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        magma_cgetmatrix( m, n, dA(0,0), ldda, work, m, queues[0] );
        magma_cgetrf_nopiv( m, n, work, m, info );
        magma_csetmatrix( m, n, work, m, dA(0,0), ldda, queues[0] );
        magma_free_cpu( work );
    }
    else {
        /* Use hybrid blocked code. */
        maxm = magma_roundup( m, 32 );

        ldwork = maxm;
        if (MAGMA_SUCCESS != magma_cmalloc_pinned( &work, ldwork*nb )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }

        for( j=0; j < s; j++ ) {
            // get j-th panel from device
            magma_queue_sync( queues[1] );
            magma_cgetmatrix_async( m-j*nb, nb, dA(j,j), ldda, work, ldwork, queues[0] );
            
            if ( j > 0 ) {
                magma_ctrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                             nb, n - (j+1)*nb,
                             c_one, dA(j-1,j-1), ldda,
                                    dA(j-1,j+1), ldda, queues[1] );
                magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                             m-j*nb, n-(j+1)*nb, nb,
                             c_neg_one, dA(j,  j-1), ldda,
                                        dA(j-1,j+1), ldda,
                             c_one,     dA(j,  j+1), ldda, queues[1] );
            }

            // do the cpu part
            rows = m - j*nb;
            magma_queue_sync( queues[0] );
            magma_cgetrf_nopiv( rows, nb, work, ldwork, &iinfo );
            if ( *info == 0 && iinfo > 0 )
                *info = iinfo + j*nb;

            // send j-th panel to device
            magma_csetmatrix_async( m-j*nb, nb, work, ldwork, dA(j, j), ldda, queues[0] );
            magma_queue_sync( queues[0] );

            // do the small non-parallel computations (next panel update)
            if ( s > j+1 ) {
                magma_ctrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dA(j, j  ), ldda,
                                    dA(j, j+1), ldda, queues[1] );
                magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                             m-(j+1)*nb, nb, nb,
                             c_neg_one, dA(j+1, j  ), ldda,
                                        dA(j,   j+1), ldda,
                             c_one,     dA(j+1, j+1), ldda, queues[1] );
            }
            else {
                magma_ctrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                             nb, n-s*nb,
                             c_one, dA(j, j  ), ldda,
                                    dA(j, j+1), ldda, queues[1] );
                magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                             m-(j+1)*nb, n-(j+1)*nb, nb,
                             c_neg_one, dA(j+1, j  ), ldda,
                                        dA(j,   j+1), ldda,
                             c_one,     dA(j+1, j+1), ldda, queues[1] );
            }
        }

        magma_int_t nb0 = min( m - s*nb, n - s*nb );
        if ( nb0 > 0 ) {
            rows = m - s*nb;
            
            magma_cgetmatrix( rows, nb0, dA(s,s), ldda, work, ldwork, queues[1] );
            
            // do the cpu part
            magma_cgetrf_nopiv( rows, nb0, work, ldwork, &iinfo );
            if ( *info == 0 && iinfo > 0 )
                *info = iinfo + s*nb;
    
            // send j-th panel to device
            magma_csetmatrix( rows, nb0, work, ldwork, dA(s,s), ldda, queues[1] );
    
            magma_ctrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                         nb0, n-s*nb-nb0,
                         c_one, dA(s,s),     ldda,
                                dA(s,s)+nb0, ldda, queues[1] );
        }
        
        magma_free_pinned( work );
    }
    
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    return *info;
} /* magma_cgetrf_nopiv_gpu */
