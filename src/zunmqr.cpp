/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Stan Tomov

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zunmqr(const char side, const char trans,
             magma_int_t m, magma_int_t n, magma_int_t k,
             magmaDoubleComplex *A,    magma_int_t lda,
             magmaDoubleComplex *tau,
             magmaDoubleComplex *C,    magma_int_t ldc,
             magmaDoubleComplex *work, magma_int_t lwork,
             magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    ZUNMQR overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**H * C       C * Q**H

    where Q is a complex orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by ZGEQRF. Q is of order M if SIDE = 'L' and of order N
    if SIDE = 'R'.

    Arguments
    =========
    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**H from the Left;
            = 'R': apply Q or Q**H from the Right.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'T':  Transpose, apply Q**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = 'L', M >= K >= 0;
            if SIDE = 'R', N >= K >= 0.

    A       (input) COMPLEX_16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument A.
            A is modified by the routine but restored on exit.

    LDA     (input) INTEGER
            The leading dimension of the array A.
            If SIDE = 'L', LDA >= max(1,M);
            if SIDE = 'R', LDA >= max(1,N).

    TAU     (input) COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    C       (input/output) COMPLEX_16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H * C or C * Q**H or C*Q.

    LDC     (input) INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(0) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If SIDE = 'L', LWORK >= max(1,N);
            if SIDE = 'R', LWORK >= max(1,M).
            For optimum performance
            LWORK >= N*NB if SIDE = 'L', and
            LWORK >= M*NB if SIDE = 'R',
            where NB is the optimal blocksize.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */
    
    #define  A(a_1,a_2) ( A + (a_1) + (a_2)*lda)
    #define dC(a_1,a_2) (dC + (a_1) + (a_2)*lddc)
    
    magma_int_t nb = magma_get_zgeqrf_nb( min( m, n ));
    
    magmaDoubleComplex c_one = MAGMA_Z_ONE;

    char side_[2]  = {side,  0};
    char trans_[2] = {trans, 0};

    magma_int_t nq_i, lddwork;
    magma_int_t i;
    magmaDoubleComplex *T;
    magma_int_t i1, i2, step, ib, ic, jc, mi, ni, nq, nw;
    int left, notran, lquery;
    magma_int_t iinfo, lwkopt;

    *info = 0;
    left   = lapackf77_lsame(side_,  "L");
    notran = lapackf77_lsame(trans_, "N");
    lquery = (lwork == -1);

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }
    lwkopt = max(1,nw) * nb;
    work[0] = MAGMA_Z_MAKE( lwkopt, 0 );
    
    if (! left && ! lapackf77_lsame(side_, "R")) {
        *info = -1;
    } else if (! notran && ! lapackf77_lsame(trans_, MagmaConjTransStr)) {
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
    } else if (lwork < max(1,nw) && ! lquery) {
        *info = -12;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        work[0] = c_one;
        return *info;
    }

    /* Allocate work space on the GPU */
    magma_int_t lddc = m;
    magmaDoubleComplex *dwork, *dC;
    magma_zmalloc( &dC, lddc*n );
    magma_zmalloc( &dwork, (m + n + nb)*nb );
    if ( dC == NULL || dwork == NULL ) {
        magma_free( dC );
        magma_free( dwork );
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    
    /* work space on CPU */
    T = (magmaDoubleComplex*) malloc( 2*nb*nb * sizeof(magmaDoubleComplex) );
    if ( T == NULL ) {
        magma_free( dC );
        magma_free( dwork );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    /* Copy matrix C from the CPU to the GPU */
    magma_zsetmatrix( m, n, C, ldc, dC, lddc );
    
    if (nb >= k) {
        /* Use CPU code */
        lapackf77_zunmqr(side_, trans_, &m, &n, &k, A, &lda, tau,
                         C, &ldc, work, &lwork, &iinfo);
    }
    else {
        /* Use hybrid CPU-GPU code */
        if ( (left && (! notran)) ||  ((! left) && notran) ) {
            i1 = 0;
            i2 = k;
            step = nb;
        } else {
            i1 = ((k - 1) / nb) * nb;
            i2 = 0;
            step = -nb;
        }

        // silence "uninitialized" warnings
        mi = 0;
        ni = 0;
        
        if (left) {
            ni = n;
            jc = 0;
        } else {
            mi = m;
            ic = 0;
        }
        
        for( i=i1; (step<0 ? i>=i2 : i<i2); i += step ) {
            ib = min(nb, k - i);

            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            nq_i = nq - i;
            lapackf77_zlarft("F", "C", &nq_i, &ib, A(i,i), &lda,
                             &tau[i], T, &ib);

            /* 1) Put 0s in the upper triangular part of A;
               2) copy the panel from A to the GPU, and
               3) restore A                                      */
            zpanel_to_q('U', ib, A(i,i), lda, T+ib*ib);
            magma_zsetmatrix( nq_i, ib, A(i,i), lda, dwork, nq_i );
            zq_to_panel('U', ib, A(i,i), lda, T+ib*ib);

            if (left) {
                /* H or H' is applied to C(i:m,1:n) */
                mi = m - i;
                ic = i;
            }
            else {
                /* H or H' is applied to C(1:m,i:n) */
                ni = n - i;
                jc = i;
            }
            
            if (left)
                lddwork = ni;
            else
                lddwork = mi;

            /* Apply H or H'; First copy T to the GPU */
            magma_zsetmatrix( ib, ib, T, ib, dwork+nq_i*ib, ib );
            magma_zlarfb_gpu( side, trans, MagmaForward, MagmaColumnwise,
                              mi, ni, ib,
                              dwork, nq_i, dwork+nq_i*ib, ib,
                              dC(ic,jc), lddc,
                              dwork+nq_i*ib + ib*ib, lddwork);
        }
        magma_zgetmatrix( m, n, dC, lddc, C, ldc );
    }
    work[0] = MAGMA_Z_MAKE( lwkopt, 0 );

    magma_free( dC );
    magma_free( dwork );
    free( T );

    return *info;
} /* magma_zunmqr */
