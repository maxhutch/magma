/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zgeqlf.cpp normal z -> c, Fri Jan 30 19:00:16 2015

*/
#include "common_magma.h"

/**
    Purpose
    -------
    CGEQLF computes a QL factorization of a COMPLEX M-by-N matrix A:
    A = Q * L.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, if m >= n, the lower triangle of the subarray
            A(m-n+1:m,1:n) contains the N-by-N lower triangular matrix L;
            if m <= n, the elements on and below the (n-m)-th
            superdiagonal contain the M-by-N lower trapezoidal matrix L;
            the remaining elements, with the array TAU, represent the
            orthogonal matrix Q as a product of elementary reflectors
            (see Further Details).
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    tau     COMPLEX array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    work    (workspace) COMPLEX array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.
    \n
            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= max(1,N,2*NB^2).
            For optimum performance LWORK >= max(N*NB, 2*NB^2) where NB can be obtained
            through magma_get_cgeqlf_nb(M).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(k) . . . H(2) H(1), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(m-k+i+1:m) = 0 and v(m-k+i) = 1; v(1:m-k+i-1) is stored on exit in
    A(1:m-k+i-1,n-k+i), and tau in TAU(i).

    @ingroup magma_cgeqlf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_cgeqlf(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A,    magma_int_t lda, magmaFloatComplex *tau,
    magmaFloatComplex *work, magma_int_t lwork,
    magma_int_t *info)
{
    #define  A(i_,j_) ( A + (i_) + (j_)*lda)
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    #define dwork(i_) (dwork + (i_))

    magmaFloatComplex_ptr dA, dwork;
    magmaFloatComplex c_one = MAGMA_C_ONE;
    magma_int_t i, k, lddwork, old_i, old_ib, nb;
    magma_int_t rows, cols;
    magma_int_t ib, ki, kk, mu, nu, iinfo, ldda;
    int lquery;

    nb = magma_get_cgeqlf_nb(m);
    *info = 0;
    lquery = (lwork == -1);

    // silence "uninitialized" warnings
    old_ib = nb;
    old_i  = 0;
    
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    }

    k = min(m,n);
    if (*info == 0) {
        if (k == 0)
            work[0] = c_one;
        else {
            work[0] = MAGMA_C_MAKE( max(n*nb, 2*nb*nb), 0 );
        }

        if (lwork < max(max(1,n), 2*nb*nb) && ! lquery)
            *info = -7;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    /* Quick return if possible */
    if (k == 0)
        return *info;

    lddwork = ((n+31)/32)*32;
    ldda    = ((m+31)/32)*32;

    if (MAGMA_SUCCESS != magma_cmalloc( &dA, (n)*ldda + nb*lddwork )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    dwork = dA + ldda*n;

    magma_queue_t queues[2];
    magma_queue_create( &queues[0] );
    magma_queue_create( &queues[1] );

    if ( (nb > 1) && (nb < k) ) {
        /*  Use blocked code initially.
            The last kk columns are handled by the block method.
            First, copy the matrix on the GPU except the last kk columns */
        magma_csetmatrix_async( m, n-nb,
                                A(0, 0),  lda,
                                dA(0, 0), ldda, queues[0] );

        ki = ((k - nb - 1) / nb) * nb;
        kk = min(k, ki + nb);
        for (i = k - kk + ki; i >= k -kk; i -= nb) {
            ib = min(k-i,nb);

            if (i < k - kk + ki) {
                /* 1. Copy asynchronously the current panel to the CPU.
                   2. Copy asynchronously the submatrix below the panel
                   to the CPU)                                        */
                rows = m - k + i + ib;
                magma_cgetmatrix_async( rows, ib,
                                        dA(0, n-k+i), ldda,
                                        A(0, n-k+i),  lda, queues[1] );

                magma_cgetmatrix_async( m-rows, ib,
                                        dA(rows, n-k+i), ldda,
                                        A(rows, n-k+i),  lda, queues[0] );

                /* Apply H' to A(1:m-k+i+ib-1,1:n-k+i-1) from the left in
                   two steps - implementing the lookahead techniques.
                   This is the main update from the lookahead techniques. */
                rows = m - k + old_i + old_ib;
                cols = n - k + old_i - old_ib;
                magma_clarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaBackward, MagmaColumnwise,
                                  rows, cols, old_ib,
                                  dA(0, cols+old_ib), ldda, dwork(0),      lddwork,
                                  dA(0, 0          ), ldda, dwork(old_ib), lddwork);
            }

            magma_queue_sync( queues[1] );
            /* Compute the QL factorization of the current block
               A(1:m-k+i+ib-1,n-k+i:n-k+i+ib-1) */
            rows = m - k + i + ib;
            cols = n - k + i;
            lapackf77_cgeqlf( &rows, &ib, A(0,cols), &lda, tau+i, work, &lwork, &iinfo );

            if (cols > 0) {
                /* Form the triangular factor of the block reflector
                   H = H(i+ib-1) . . . H(i+1) H(i) */
                lapackf77_clarft( MagmaBackwardStr, MagmaColumnwiseStr,
                                  &rows, &ib,
                                  A(0, cols), &lda, tau + i, work, &ib);

                cpanel_to_q( MagmaLower, ib, A(rows-ib,cols), lda, work+ib*ib);
                magma_csetmatrix( rows, ib,
                                  A(0,cols),  lda,
                                  dA(0,cols), ldda );
                cq_to_panel( MagmaLower, ib, A(rows-ib,cols), lda, work+ib*ib);

                // Send the triangular part on the GPU
                magma_csetmatrix( ib, ib, work, ib, dwork(0), lddwork );

                /* Apply H' to A(1:m-k+i+ib-1,1:n-k+i-1) from the left in
                   two steps - implementing the lookahead techniques.
                   This is the update of first ib columns.                 */
                if (i-ib >= k -kk)
                    magma_clarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaBackward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(0, cols),   ldda, dwork(0),  lddwork,
                                      dA(0,cols-ib), ldda, dwork(ib), lddwork);
                else {
                    magma_clarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaBackward, MagmaColumnwise,
                                      rows, cols, ib,
                                      dA(0, cols), ldda, dwork(0),  lddwork,
                                      dA(0, 0   ), ldda, dwork(ib), lddwork);
                }

                old_i  = i;
                old_ib = ib;
            }
        }
        mu = m - k + i + nb;
        nu = n - k + i + nb;

        magma_cgetmatrix( m, nu, dA(0,0), ldda, A(0,0), lda );
    } else {
        mu = m;
        nu = n;
    }

    /* Use unblocked code to factor the last or only block */
    if (mu > 0 && nu > 0)
        lapackf77_cgeqlf(&mu, &nu, A(0,0), &lda, tau, work, &lwork, &iinfo);

    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    magma_free( dA );
    
    return *info;
} /* magma_cgeqlf */

#undef  A
#undef dA
