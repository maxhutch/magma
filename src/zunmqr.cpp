/*
    -- MAGMA (version 1.5.0-beta1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date April 2014

       @author Stan Tomov

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/**
    Purpose
    -------
    ZUNMQR overwrites the general complex M-by-N matrix C with

    @verbatim
                              SIDE = MagmaLeft   SIDE = MagmaRight
    TRANS = MagmaNoTrans:     Q * C              C * Q
    TRANS = MagmaConjTrans:   Q**H * C           C * Q**H
    @endverbatim

    where Q is a complex orthogonal matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by ZGEQRF. Q is of order M if SIDE = MagmaLeft and of order N
    if SIDE = MagmaRight.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:      apply Q or Q**H from the Left;
      -     = MagmaRight:     apply Q or Q**H from the Right.

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    No transpose, apply Q;
      -     = MagmaConjTrans:  Conjugate transpose, apply Q**H.

    @param[in]
    m       INTEGER
            The number of rows of the matrix C. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix C. N >= 0.

    @param[in]
    k       INTEGER
            The number of elementary reflectors whose product defines
            the matrix Q.
            If SIDE = MagmaLeft,  M >= K >= 0;
            if SIDE = MagmaRight, N >= K >= 0.

    @param[in]
    A       COMPLEX_16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument A.
            A is modified by the routine but restored on exit.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.
            If SIDE = MagmaLeft,  LDA >= max(1,M);
            if SIDE = MagmaRight, LDA >= max(1,N).

    @param[in]
    tau     COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    @param[in,out]
    C       COMPLEX_16 array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H * C or C * Q**H or C*Q.

    @param[in]
    ldc     INTEGER
            The leading dimension of the array C. LDC >= max(1,M).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(0) returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If SIDE = MagmaLeft,  LWORK >= max(1,N);
            if SIDE = MagmaRight, LWORK >= max(1,M).
            For optimum performance
            LWORK >= N*NB if SIDE = MagmaLeft, and
            LWORK >= M*NB if SIDE = MagmaRight,
            where NB is the optimal blocksize.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_zgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zunmqr(magma_side_t side, magma_trans_t trans,
             magma_int_t m, magma_int_t n, magma_int_t k,
             magmaDoubleComplex *A,    magma_int_t lda,
             magmaDoubleComplex *tau,
             magmaDoubleComplex *C,    magma_int_t ldc,
             magmaDoubleComplex *work, magma_int_t lwork,
             magma_int_t *info)
{
    #define  A(a_1,a_2) ( A + (a_1) + (a_2)*lda)
    #define dC(a_1,a_2) (dC + (a_1) + (a_2)*lddc)
    
    magma_int_t nb = magma_get_zgeqrf_nb( min( m, n ));
    
    magmaDoubleComplex c_one = MAGMA_Z_ONE;

    const char* side_  = lapack_side_const( side  );
    const char* trans_ = lapack_trans_const( trans );

    magma_int_t nq_i, lddwork;
    magma_int_t i;
    magmaDoubleComplex *T;
    magma_int_t i1, i2, step, ib, ic, jc, mi, ni, nq, nw;
    int left, notran, lquery;
    magma_int_t iinfo, lwkopt;

    *info = 0;
    left   = (side == MagmaLeft);
    notran = (trans == MagmaNoTrans);
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
    
    if (! left && side != MagmaRight) {
        *info = -1;
    } else if (! notran && trans != MagmaConjTrans) {
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
    magma_zmalloc_cpu( &T, 2*nb*nb );
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
        
        for( i=i1; (step < 0 ? i >= i2 : i < i2); i += step ) {
            ib = min(nb, k - i);

            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            nq_i = nq - i;
            lapackf77_zlarft("F", "C", &nq_i, &ib, A(i,i), &lda,
                             &tau[i], T, &ib);

            /* 1) Put 0s in the upper triangular part of A;
               2) copy the panel from A to the GPU, and
               3) restore A                                      */
            zpanel_to_q( MagmaUpper, ib, A(i,i), lda, T+ib*ib);
            magma_zsetmatrix( nq_i, ib, A(i,i), lda, dwork, nq_i );
            zq_to_panel( MagmaUpper, ib, A(i,i), lda, T+ib*ib);

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
    magma_free_cpu( T );

    return *info;
} /* magma_zunmqr */
