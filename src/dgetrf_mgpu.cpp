/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zgetrf_mgpu.cpp normal z -> d, Mon May  2 23:30:03 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    DGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    d_lA    DOUBLE PRECISION array of pointers on the GPU, dimension (ngpu).
            On entry, the M-by-N matrix A distributed over GPUs
            (d_lA[d] points to the local matrix on d-th GPU).
            It uses 1D block column cyclic format with the block size of nb,
            and each local matrix is stored by column.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array d_lA.  LDDA >= max(1,M).

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

    @ingroup magma_dgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dgetrf_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr d_lA[], magma_int_t ldda, magma_int_t *ipiv,
    magma_int_t *info)
{
    magma_int_t nb, n_local[MagmaMaxGPUs];
    magma_int_t maxm;
    magma_int_t i, j, d, lddat, lddwork;
    double *d_lAT[MagmaMaxGPUs];
    double *d_panel[MagmaMaxGPUs], *work;
    magma_queue_t queues[MagmaMaxGPUs][2];

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -2;
    else if (n < 0)
        *info = -3;
    else if (ldda < max(1,m))
        *info = -5;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* create the queues */
    for( d=0; d < ngpu; d++ ) {
        magma_queue_create( d, &queues[d][0] );
        magma_queue_create( d, &queues[d][1] );
    }

    /* Function Body */
    nb = magma_get_dgetrf_nb( m, n );

    if (nb <= 1 || nb >= n) {
        /* Use CPU code. */
        magma_dmalloc_cpu( &work, m * n );
        if ( work == NULL ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        magma_dgetmatrix( m, n, d_lA[0], ldda, work, m, queues[0][0] );
        lapackf77_dgetrf(&m, &n, work, &m, ipiv, info);
        magma_dsetmatrix( m, n, work, m, d_lA[0], ldda, queues[0][0] );
        magma_free_cpu(work);
    } else {
        /* Use hybrid blocked code. */
        magma_device_t orig_dev;
        magma_getdevice( &orig_dev );
        
        maxm = magma_roundup( m, 32 );
        if ( ngpu > ceil((double)n/nb) ) {
            printf( " * too many GPUs for the matrix size, using %d GPUs\n", (int) ngpu );
            *info = -1;
            return *info;
        }

        /* allocate workspace for each GPU */
        lddat = magma_roundup( ((magma_ceildiv( n, nb )/ngpu)*nb), 32 );
        lddat = magma_ceildiv( n, nb );        /* number of block columns         */
        lddat = magma_ceildiv( lddat, ngpu );  /* number of block columns per GPU */
        lddat = nb*lddat;                      /* number of columns per GPU       */
        lddat = magma_roundup( lddat, 32 );    /* make it a multiple of 32        */
        for (i=0; i < ngpu; i++) {
            magma_setdevice(i);
            
            /* local-n and local-ld */
            n_local[i] = ((n/nb)/ngpu)*nb;
            if (i < (n/nb)%ngpu)
                n_local[i] += nb;
            else if (i == (n/nb)%ngpu)
                n_local[i] += n%nb;
            
            /* workspaces */
            if (MAGMA_SUCCESS != magma_dmalloc( &d_panel[i], (3+ngpu)*nb*maxm )) {
                for( j=0; j <= i; j++ ) {
                    magma_setdevice(j);
                }
                for( j=0; j < i; j++ ) {
                    magma_setdevice(j);
                    magma_free( d_panel[j] );
                    magma_free( d_lAT[j]   );
                }
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            
            /* local-matrix storage */
            if (MAGMA_SUCCESS != magma_dmalloc( &d_lAT[i], lddat*maxm )) {
                for( j=0; j <= i; j++ ) {
                    magma_setdevice(j);
                    magma_free( d_panel[j] );
                }
                for( j=0; j < i; j++ ) {
                    magma_setdevice(j);
                    magma_free( d_lAT[j] );
                }
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            
            magmablas_dtranspose( m, n_local[i], d_lA[i], ldda, d_lAT[i], lddat, queues[i][1] );
        }
        for (i=0; i < ngpu; i++) {
            magma_setdevice(i);
            magma_queue_sync(queues[i][0]);
        }
        magma_setdevice(0);

        /* cpu workspace */
        lddwork = maxm;
        if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, lddwork*nb*ngpu )) {
            for (i=0; i < ngpu; i++ ) {
                magma_setdevice(i);
                magma_free( d_panel[i] );
                magma_free( d_lAT[i]   );
            }
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }

        /* calling multi-gpu interface with allocated workspaces and queues */
        magma_dgetrf2_mgpu(ngpu, m, n, nb, 0, d_lAT, lddat, ipiv, d_panel, work, maxm,
                           queues, info);

        /* clean up */
        for( d=0; d < ngpu; d++ ) {
            magma_setdevice(d);
            
            /* save on output */
            magmablas_dtranspose( n_local[d], m, d_lAT[d], lddat, d_lA[d], ldda, queues[d][0] );
            magma_queue_sync(queues[d][0]);
            magma_queue_sync(queues[d][1]);

            magma_free( d_lAT[d]   );
            magma_free( d_panel[d] );
        } /* end of for d=1,..,ngpu */
        magma_setdevice( orig_dev );
        magma_free_pinned( work );
    }

    /* clean up */
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        magma_queue_destroy( queues[d][0] );
        magma_queue_destroy( queues[d][1] );
    }

    return *info;
}
