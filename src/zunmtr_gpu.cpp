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

extern "C" magma_int_t
magma_zunmtr_gpu(char side, char uplo, char trans,
                 magma_int_t m, magma_int_t n,
                 magmaDoubleComplex *da,    magma_int_t ldda,
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
    ZUNMTR overwrites the general complex M-by-N matrix C with

                    SIDE = 'L'     SIDE = 'R'
    TRANS = 'N':      Q * C          C * Q
    TRANS = 'T':      Q**H * C       C * Q**H

    where Q is a complex orthogonal matrix of order nq, with nq = m if
    SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
    nq-1 elementary reflectors, as returned by SSYTRD:

    if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);

    if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).

    Arguments
    =========
    SIDE    (input) CHARACTER*1
            = 'L': apply Q or Q**H from the Left;
            = 'R': apply Q or Q**H from the Right.

    UPLO    (input) CHARACTER*1
            = 'U': Upper triangle of A contains elementary reflectors
                   from SSYTRD;
            = 'L': Lower triangle of A contains elementary reflectors
                   from SSYTRD.

    TRANS   (input) CHARACTER*1
            = 'N':  No transpose, apply Q;
            = 'T':  Transpose, apply Q**H.

    M       (input) INTEGER
            The number of rows of the matrix C. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix C. N >= 0.

    DA      (device input) COMPLEX_16 array, dimension
                                 (LDDA,M) if SIDE = 'L'
                                 (LDDA,N) if SIDE = 'R'
            The vectors which define the elementary reflectors, as
            returned by ZHETRD_GPU. On output the diagonal, the subdiagonal and the
            upper part (UPLO='L') or lower part (UPLO='U') are destroyed.

    LDDA    (input) INTEGER
            The leading dimension of the array DA.
            LDDA >= max(1,M) if SIDE = 'L'; LDDA >= max(1,N) if SIDE = 'R'.

    TAU     (input) COMPLEX_16 array, dimension
                                 (M-1) if SIDE = 'L'
                                 (N-1) if SIDE = 'R'
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SSYTRD.

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

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
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
            message related to LWORK is issued.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================    */

    char side_[2]  = {side, 0};
    char uplo_[2]  = {uplo, 0};
    char trans_[2] = {trans, 0};
    magma_int_t i1, i2, mi, ni, nq, nw;
    int left, upper;
    magma_int_t iinfo;

    *info = 0;
    left   = lapackf77_lsame(side_, "L");
    upper  = lapackf77_lsame(uplo_, "U");

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }
    if (! left && ! lapackf77_lsame(side_, "R")) {
        *info = -1;
    } else if (! upper && ! lapackf77_lsame(uplo_, "L")) {
        *info = -2;
    } else if (! lapackf77_lsame(trans_, "N") &&
               ! lapackf77_lsame(trans_, "C")) {
        *info = -3;
    } else if (m < 0) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (ldda < max(1,nq)) {
        *info = -7;
    } else if (lddc < max(1,m)) {
        *info = -10;
    } else if (ldwa < max(1,nq)) {
        *info = -12;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || nq == 1) {
        return *info;
    }

    if (left) {
        mi = m - 1;
        ni = n;
    } else {
        mi = m;
        ni = n - 1;
    }

    if (upper) {
        magma_zunmql2_gpu(side, trans, mi, ni, nq-1, &da[ldda], ldda, tau,
                          dc, lddc, &wa[ldwa], ldwa, &iinfo);
    }
    else {
        /* Q was determined by a call to SSYTRD with UPLO = 'L' */
        if (left) {
            i1 = 1;
            i2 = 0;
        } else {
            i1 = 0;
            i2 = 1;
        }
        magma_zunmqr2_gpu(side, trans, mi, ni, nq-1, &da[1], ldda, tau,
                          &dc[i1 + i2*lddc], lddc, &wa[1], ldwa, &iinfo);
    }

    return *info;
} /* zunmtr */
