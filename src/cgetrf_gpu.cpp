/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @author Stan Tomov
       @generated from zgetrf_gpu.cpp normal z -> c, Fri Jul 18 17:34:15 2014
*/
#include "common_magma.h"

/**
    Purpose
    -------
    CGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    If the current stream is NULL, this version replaces it with user defined
    stream to overlap computation with communication.

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
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

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
magma_cgetrf_gpu(magma_int_t m, magma_int_t n,
                 magmaFloatComplex *dA, magma_int_t ldda,
                 magma_int_t *ipiv, magma_int_t *info)
{
    #define dAT(i,j) (dAT + (i)*nb*lddat + (j)*nb)

    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;

    magma_int_t iinfo, nb;
    magma_int_t maxm, maxn, mindim;
    magma_int_t i, rows, cols, s, lddat, lddwork;
    magmaFloatComplex *dAT, *dAP, *work;

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
    nb     = magma_get_cgetrf_nb(m);
    s      = mindim / nb;

    if (nb <= 1 || nb >= min(m,n)) {
        /* Use CPU code. */
        magma_cmalloc_cpu( &work, m * n );
        if ( work == NULL ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        magma_cgetmatrix( m, n, dA, ldda, work, m );
        lapackf77_cgetrf(&m, &n, work, &m, ipiv, info);
        magma_csetmatrix( m, n, work, m, dA, ldda );
        magma_free_cpu(work);
    }
    else {
        /* Use hybrid blocked code. */
        maxm = ((m + 31)/32)*32;
        maxn = ((n + 31)/32)*32;

        lddat   = maxn;
        lddwork = maxm;

        dAT = dA;

        if (MAGMA_SUCCESS != magma_cmalloc( &dAP, nb*maxm )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        if ( m == n ) {
            lddat = ldda;
            magmablas_ctranspose_inplace( m, dAT, ldda );
        }
        else {
            if (MAGMA_SUCCESS != magma_cmalloc( &dAT, maxm*maxn )) {
                magma_free( dAP );
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            magmablas_ctranspose( m, n, dA, ldda, dAT, lddat );
        }

        if (MAGMA_SUCCESS != magma_cmalloc_pinned( &work, maxm*nb )) {
            magma_free( dAP );
            if ( ! (m == n))
                magma_free( dAT );
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }

        /* Define user stream if current stream is NULL */
        cudaStream_t stream[2], current_stream;
        magmablasGetKernelStream(&current_stream);

        magma_queue_create( &stream[0] );
        if (current_stream == NULL) {
            magma_queue_create( &stream[1] );
            magmablasSetKernelStream(stream[1]);
        }
        else {
            stream[1] = current_stream;
        }
  
        for( i=0; i < s; i++ ) {
            // download i-th panel
            cols = maxm - i*nb;
            //magmablas_ctranspose( nb, cols, dAT(i,i), lddat, dAP, cols );
            magmablas_ctranspose( nb, m-i*nb, dAT(i,i), lddat, dAP, cols );

            // make sure that that the transpose has completed
            magma_queue_sync( stream[1] );
            magma_cgetmatrix_async( m-i*nb, nb, dAP, cols, work, lddwork,
                                    stream[0]);

            if ( i > 0 ) {
                magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n - (i+1)*nb, nb,
                             c_one, dAT(i-1,i-1), lddat,
                                    dAT(i-1,i+1), lddat );
                magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(i+1)*nb, m-i*nb, nb,
                             c_neg_one, dAT(i-1,i+1), lddat,
                                        dAT(i,  i-1), lddat,
                             c_one,     dAT(i,  i+1), lddat );
            }

            // do the cpu part
            rows = m - i*nb;
            magma_queue_sync( stream[0] );
            lapackf77_cgetrf( &rows, &nb, work, &lddwork, ipiv+i*nb, &iinfo);
            if ( (*info == 0) && (iinfo > 0) )
                *info = iinfo + i*nb;

            // upload i-th panel
            magma_csetmatrix_async( m-i*nb, nb, work, lddwork, dAP, maxm,
                                    stream[0]);

            magmablas_cpermute_long2( n, dAT, lddat, ipiv, nb, i*nb );

            magma_queue_sync( stream[0] );
            //magmablas_ctranspose( cols, nb, dAP, maxm, dAT(i,i), lddat );
            magmablas_ctranspose( m-i*nb, nb, dAP, maxm, dAT(i,i), lddat );

            // do the small non-parallel computations (next panel update)
            if ( s > (i+1) ) {
                magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dAT(i, i  ), lddat,
                                    dAT(i, i+1), lddat);
                magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                             nb, m-(i+1)*nb, nb,
                             c_neg_one, dAT(i,   i+1), lddat,
                                        dAT(i+1, i  ), lddat,
                             c_one,     dAT(i+1, i+1), lddat );
            }
            else {
                magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-s*nb, nb,
                             c_one, dAT(i, i  ), lddat,
                                    dAT(i, i+1), lddat);
                magma_cgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(i+1)*nb, m-(i+1)*nb, nb,
                             c_neg_one, dAT(i,   i+1), lddat,
                                        dAT(i+1, i  ), lddat,
                             c_one,     dAT(i+1, i+1), lddat );
            }
        }

        magma_int_t nb0 = min(m - s*nb, n - s*nb);
        rows = m - s*nb;
        cols = maxm - s*nb;

        magmablas_ctranspose( nb0, rows, dAT(s,s), lddat, dAP, maxm );
        magma_cgetmatrix( rows, nb0, dAP, maxm, work, lddwork );

        // do the cpu part
        lapackf77_cgetrf( &rows, &nb0, work, &lddwork, ipiv+s*nb, &iinfo);
        if ( (*info == 0) && (iinfo > 0) )
            *info = iinfo + s*nb;
        magmablas_cpermute_long2( n, dAT, lddat, ipiv, nb0, s*nb );

        // upload i-th panel
        magma_csetmatrix( rows, nb0, work, lddwork, dAP, maxm );
        magmablas_ctranspose( rows, nb0, dAP, maxm, dAT(s,s), lddat );

        magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                     n-s*nb-nb0, nb0,
                     c_one, dAT(s,s),     lddat,
                            dAT(s,s)+nb0, lddat);

        if ( m == n ) {
            magmablas_ctranspose_inplace( m, dAT, lddat );
        }
        else {
            magmablas_ctranspose( n, m, dAT, lddat, dA, ldda );
            magma_free( dAT );
        }

        magma_free( dAP );
        magma_free_pinned( work );
    
        magma_queue_destroy( stream[0] );
        if (current_stream == NULL) {
            magma_queue_destroy( stream[1] );
            magmablasSetKernelStream(NULL);
        }
    }
    return *info;
} /* magma_cgetrf_gpu */
