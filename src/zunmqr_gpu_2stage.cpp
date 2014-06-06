/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @author Azzam Haidar
       @author Stan Tomov
       @author Raffaele Solca

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zunmqr_gpu_2stages(const char side, const char trans,
                         magma_int_t m, magma_int_t n, magma_int_t k,
                         magmaDoubleComplex *da,   magma_int_t ldda,
                         magmaDoubleComplex *dc,    magma_int_t lddc,
                         magmaDoubleComplex *dT,    magma_int_t nb,
                         magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    ZUNMQR_GPU overwrites the general complex M-by-N matrix C with

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

    DA      (input) COMPLEX_16 array on the GPU, dimension (LDDA,K)
    The i-th column must contain the vector which defines the
    elementary reflector H(i), for i = 1,2,...,k, as returned by
    ZGEQRF in the first k columns of its array argument DA.
    DA is modified by the routine but restored on exit.

    LDDA    (input) INTEGER
    The leading dimension of the array DA.
    If SIDE = 'L', LDDA >= max(1,M);
    if SIDE = 'R', LDDA >= max(1,N).

    DC      (input/output) COMPLEX_16 array on the GPU, dimension (LDDC,N)
    On entry, the M-by-N matrix C.
    On exit, C is overwritten by Q*C or Q**H * C or C * Q**H or C*Q.

    LDDC     (input) INTEGER
    The leading dimension of the array DC. LDDC >= max(1,M).

    DT      (input) COMPLEX_16 array on the GPU that is the output
    (the 9th argument) of magma_zgeqrf_gpu.

    NB      (input) INTEGER
    This is the blocking size that was used in pre-computing DT, e.g.,
    the blocking size used in magma_zgeqrf_gpu.

    INFO    (output) INTEGER
    = 0:  successful exit
    < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================   */

    char side_[2] = {side, 0};
    char trans_[2] = {trans, 0};

    magmaDoubleComplex *dwork;

    magma_int_t i1, i2, i3, ib, ic, jc, mi, ni, nq, nw, ret;
    int left, notran;
    //magma_int_t lwkopt;

    *info = 0;
    left   = lapackf77_lsame(side_, "L");
    notran = lapackf77_lsame(trans_, "N");

    /* NQ is the order of Q and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    } else {
        nq = n;
        nw = m;
    }
    if ( (!left) && (!lapackf77_lsame(side_, "R")) ) {
        *info = -1;
    } else if ( (!notran) && (!lapackf77_lsame(trans_, MagmaConjTransStr)) ) {
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
    }

    if(MAGMA_SUCCESS != magma_zmalloc( &dwork, n*nb )) {
        printf ("!!!! zungqr_2stage magma_alloc failed for: dwork\n" );
        exit(-1);
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        return *info;
    }

    if ( (left && (! notran)) || ( (!left) && notran ) ) {
        i1 = 0;
        i2 = k;
        i3 = nb;
    } else {
        i1 = (k - 1) / nb * nb;
        i2 = 0;
        i3 = -nb;
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

    for (magma_int_t i=i1; (i3<0 ? i>=i2 : i<i2); i+=i3)
    {
        ib = min(nb, k - i);
        if (left){
            mi = m - i;
            ic = i;
        }
        else {
            ni = n - i;
            jc = i;
        }
        ret = magma_zlarfb_gpu( MagmaLeft, trans, MagmaForward, MagmaColumnwise,
                               mi, ni, ib, da+i+i*ldda, ldda, dT+i*nb, nb,
                               dc+ic+jc*lddc, lddc, dwork, nw);

        if ( ret != MAGMA_SUCCESS ){
            magma_free(dwork);
            return ret;
        }
    }

    return MAGMA_SUCCESS;
}   /* End of MAGMA_ZUNMQR_GPU_2stages */
