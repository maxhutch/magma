/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Raffaele Solca
       @author Stan Tomov

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern"C"{
  void magmablas_zsetdiag1subdiag0(char uplo, int k, int nb, magmaDoubleComplex *A, int lda);
}

extern "C" magma_int_t
magma_zunmqr2_gpu(const char side, const char trans,
                  magma_int_t m, magma_int_t n, magma_int_t k,
                  magmaDoubleComplex *da,   magma_int_t ldda,
                  magmaDoubleComplex *tau,
                  magmaDoubleComplex *dc,    magma_int_t lddc,
                  magmaDoubleComplex *wa,    magma_int_t ldwa,
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
    TRANS = 'T':      Q**H * C     C * Q**H

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

    DA      (input) COMPLEX_16 array, dimension (LDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            ZGEQRF in the first k columns of its array argument A.
            The diagonal and the upper part
            are destroyed, the reflectors are not modified.

    LDDA    (input) INTEGER
            The leading dimension of the array DA.
            LDDA >= max(1,M) if SIDE = 'L'; LDDA >= max(1,N) if SIDE = 'R'.

    TAU     (input) COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF.

    DC      (device input/output) COMPLEX_16 array, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by (Q*C) or (Q**H * C) or (C * Q**H) or (C*Q).

    LDDC    (input) INTEGER
            The leading dimension of the array C. LDDC >= max(1,M).

    WA      (input/workspace) COMPLEX_16 array, dimension
                                 (LDWA,M) if SIDE = 'L'
                                 (LDWA,N) if SIDE = 'R'
            The vectors which define the elementary reflectors, as
            returned by ZHETRD_GPU.

    LDWA    (input) INTEGER
            The leading dimension of the array A.
            LDWA >= max(1,M) if SIDE = 'L'; LDWA >= max(1,N) if SIDE = 'R'.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */

    char side_[2]  = {side,  0};
    char trans_[2] = {trans, 0};

    /* Allocate work space on the GPU */
    magmaDoubleComplex *dwork;

    magma_int_t wa_offset, dc_offset, i__4, lddwork;
    magma_int_t i;
    magmaDoubleComplex t[2*4160]        /* was [65][64] */;
    magma_int_t i1, i2, step, ib, ic, jc, nb, mi, ni, nq, nw;
    int left, notran;

    wa_offset = 1 + ldwa;
    wa -= wa_offset;
    --tau;
    dc_offset = 1 + lddc;
    dc -= dc_offset;

    *info = 0;
    left   = lapackf77_lsame(side_,  "L");
    notran = lapackf77_lsame(trans_, "N");

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
        magma_zmalloc( &dwork, (n + 64)*64 );
    } else {
        nq = n;
        nw = m;
        magma_zmalloc( &dwork, (m + 64)*64 );
    }
    if (! left && ! lapackf77_lsame(side_, "R")) {
        *info = -1;
    } else if (! notran && ! lapackf77_lsame(trans_, "T")) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (k < 0 || k > nq) {
        *info = -5;
    } else if (ldda < max(1,nq)) {
        *info = -7;
    } else if (lddc < max(1,m)) {
        *info = -10;
    } else if (ldwa < max(1,nq)) {
        *info = -12;
    }

    // size of the block
    nb = 64;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return *info;
    }

    /* Use hybrid CPU-GPU code */
    if ( ( left && (! notran) ) ||  ( (! left) && notran ) ) {
        i1 = 1;
        i2 = k;
        step = nb;
    } else {
        i1 = ((k - 1)/nb)*nb + 1;
        i2 = 1;
        step = -nb;
    }

    // silence "uninitialized" warnings
    mi = 0;
    ni = 0;
    
    if (left) {
        ni = n;
        jc = 1;
    } else {
        mi = m;
        ic = 1;
    }

    magmablas_zsetdiag1subdiag0('L', k, nb, da, ldda);

    // for i=i1 to i2 by step
    for (i = i1; (step < 0 ? i >= i2 : i <= i2); i += step) {
        ib = min(nb, k - i + 1);

        /* Form the triangular factor of the block reflector
           H = H(i) H(i+1) . . . H(i+ib-1) */
        i__4 = nq - i + 1;
        lapackf77_zlarft("F", "C", &i__4, &ib, &wa[i + i*ldwa], &ldwa,
                         &tau[i], t, &ib);


        if (left) {
            /* H or H' is applied to C(i:m,1:n) */
            mi = m - i + 1;
            ic = i;
        }
        else {
            /* H or H' is applied to C(1:m,i:n) */
            ni = n - i + 1;
            jc = i;
        }

        if (left)
            lddwork = ni;
        else
            lddwork = mi;

        /* Apply H or H'; First copy T to the GPU */
        magma_zsetmatrix( ib, ib, t, ib, dwork, ib );
        magma_zlarfb_gpu( side, trans, MagmaForward, MagmaColumnwise,
                          mi, ni, ib,
                          da + (i - 1) + (i - 1)*ldda , ldda, dwork, ib,
                          &dc[ic + jc*lddc], lddc,
                          dwork + ib*ib, lddwork);
    }

    magma_free( dwork );

    return *info;
} /* magma_zunmqr */


