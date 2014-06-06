/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Raffaele Solca

       @generated s Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_sormql(const char side, const char trans,
             magma_int_t m, magma_int_t n, magma_int_t k,
             float *a, magma_int_t lda,
             float *tau,
             float *c, magma_int_t ldc,
             float *work, magma_int_t lwork,
             magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    SORMQL overwrites the general real M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'C':      Q**T * C       C * Q**T

    where Q is a real unitary matrix defined as the product of k
    elementary reflectors

          Q = H(k) . . . H(2) H(1)

    as returned by SGEQLF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========
    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**T from the Left;
            = 'R': apply Q or Q**T from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'C':  Transpose, apply Q**T.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) REAL array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            SGEQLF in the last k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEQLF.

    C       (input/output) REAL array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance LWORK >= N*NB if SIDE = 'L', and
            LWORK >= M*NB if SIDE = 'R', where NB is the optimal
            blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================    */
    
    char side_[2] = {side, 0};
    char trans_[2] = {trans, 0};

    magma_int_t i__4, i__;
    float *T;
    magma_int_t i1, i2, i3, ib, nb, mi, ni, nq, nw;
    magma_int_t iinfo, ldwork, lwkopt=0;
    int lquery, left, notran;

    *info  = 0;
    left   = lapackf77_lsame(side_, "L");
    notran = lapackf77_lsame(trans_, "N");
    lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = max(1,n);
    } else {
        nq = n;
        nw = max(1,m);
    }
    if (! left && ! lapackf77_lsame(side_, "R")) {
        *info = -1;
    } else if (! notran && ! lapackf77_lsame(trans_, "C")) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (lda < max(1,nq)) {
        *info = -7;
    } else if (ldc < max(1,m)) {
        *info = -10;
    }

    if (*info == 0) {
        if (m == 0 || n == 0) {
            lwkopt = 1;
        } else {
            /* Determine the block size.  NB may be at most NBMAX, where
               NBMAX is used to define the local array T.                 */
            nb = 64;
            lwkopt = nw * nb;
        }
        work[0] = MAGMA_S_MAKE( lwkopt, 0 );
        
        if (lwork < nw && ! lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return *info;
    }

    /* Allocate work space on the GPU */
    float *dwork, *dc;
    magma_smalloc( &dc, (m)*(n) );
    magma_smalloc( &dwork, 2*(m + 64)*64 );

    /* Copy matrix C from the CPU to the GPU */
    magma_ssetmatrix( m, n, c, ldc, dc, m );

    /* work space on CPU */
    if (  MAGMA_SUCCESS != magma_smalloc_pinned( &T, 2*nb*nb ) ) {
        magma_free( dc );
        magma_free( dwork );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    ldwork = nw;

    if ( nb >= k ) {
        /* Use CPU code */
        lapackf77_sormql(side_, trans_, &m, &n, &k, a, &lda, tau,
                         c, &ldc, work, &lwork, &iinfo);
    }
    else {
        /* Use hybrid CPU-GPU code */
        if ((left && notran) || (! left && ! notran)) {
            i1 = 1;
            i2 = k;
            i3 = nb;
        } else {
            i1 = (k - 1) / nb * nb + 1;
            i2 = 1;
            i3 = -nb;
        }

        // silence "uninitialized" warnings
        mi = 0;
        ni = 0;
        
        if (left) {
            ni = n;
        } else {
            mi = m;
        }

        for (i__ = i1; (i3 < 0 ? i__ >= i2 : i__ <= i2); i__ += i3) {
            ib = min(nb, k - i__ + 1);
            
            /* Form the triangular factor of the block reflector
               H = H(i+ib-1) . . . H(i+1) H(i) */
            i__4 = nq - k + i__ + ib - 1;
            lapackf77_slarft("Backward", "Columnwise", &i__4, &ib,
                             &a[(i__-1) * lda], &lda, &tau[i__-1], T, &ib);
            
            /* 1) Put 0s in the lower triangular part of A;
               2) copy the panel from A to the GPU, and
               3) restore A                                      */
            spanel_to_q('L', ib, &a[i__-1 + (i__-1) * lda], lda, T+ib*ib);
            magma_ssetmatrix( i__4, ib, &a[(i__-1) * lda], lda, dwork, i__4 );
            sq_to_panel('L', ib, &a[i__-1 + (i__-1) * lda], lda, T+ib*ib);
            
            if (left) {
                /* H or H' is applied to C(1:m-k+i+ib-1,1:n) */
                mi = m - k + i__ + ib - 1;
            }
            else {
                /* H or H' is applied to C(1:m,1:n-k+i+ib-1) */
                ni = n - k + i__ + ib - 1;
            }
            
            /* Apply H or H'; First copy T to the GPU */
            magma_ssetmatrix( ib, ib, T, ib, dwork+i__4*ib, ib );
            magma_slarfb_gpu(side, trans, MagmaBackward, MagmaColumnwise,
                             mi, ni, ib,
                             dwork, i__4, dwork+i__4*ib, ib,
                             dc, m,
                             dwork+i__4*ib + ib*ib, ldwork);
        }

        magma_sgetmatrix( m, n, dc, m, c, ldc );
    }
    work[0] = MAGMA_S_MAKE( lwkopt, 0 );

    magma_free( dc );
    magma_free( dwork );

    magma_free_pinned( T);  

    return *info;
} /* magma_sormql */
