/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Stan Tomov
       @author Mark Gates

       @generated from zunmqr_gpu.cpp normal z -> s, Fri Jan 30 19:00:16 2015

*/
#include "common_magma.h"

/**
    Purpose
    -------
    SORMQR_GPU overwrites the general real M-by-N matrix C with

    @verbatim
                               SIDE = MagmaLeft    SIDE = MagmaRight
    TRANS = MagmaNoTrans:      Q * C               C * Q
    TRANS = MagmaTrans:   Q**H * C            C * Q**H
    @endverbatim

    where Q is a real unitary matrix defined as the product of k
    elementary reflectors

          Q = H(1) H(2) . . . H(k)

    as returned by SGEQRF. Q is of order M if SIDE = MagmaLeft and of order N
    if SIDE = MagmaRight.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:   apply Q or Q**H from the Left;
      -     = MagmaRight:  apply Q or Q**H from the Right.

    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    No transpose, apply Q;
      -     = MagmaTrans: Conjugate transpose, apply Q**H.

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
    dA      REAL array on the GPU, dimension (LDDA,K)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,k, as returned by
            SGEQRF in the first k columns of its array argument DA.
            DA is modified by the routine but restored on exit.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array DA.
            If SIDE = MagmaLeft,  LDDA >= max(1,M);
            if SIDE = MagmaRight, LDDA >= max(1,N).

    @param[in]
    tau     REAL array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by SGEQRF.

    @param[in,out]
    dC      REAL array on the GPU, dimension (LDDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H * C or C * Q**H or C*Q.

    @param[in]
    lddc    INTEGER
            The leading dimension of the array DC. LDDC >= max(1,M).

    @param[out]
    hwork   (workspace) REAL array, dimension (MAX(1,LWORK))
    \n
            Currently, sgetrs_gpu assumes that on exit, hwork contains the last
            block of A and C. This will change and *should not be relied on*!

    @param[in]
    lwork   INTEGER
            The dimension of the array HWORK.
            LWORK >= (M-K+NB)*(N+NB) + N*NB if SIDE = MagmaLeft, and
            LWORK >= (N-K+NB)*(M+NB) + M*NB if SIDE = MagmaRight,
            where NB is the given blocksize.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the HWORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[in]
    dT      REAL array on the GPU that is the output
            (the 9th argument) of magma_sgeqrf_gpu.

    @param[in]
    nb      INTEGER
            This is the blocking size that was used in pre-computing DT, e.g.,
            the blocking size used in magma_sgeqrf_gpu.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_sgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_sormqr_gpu(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloat_ptr dA,    magma_int_t ldda,
    float    *tau,
    magmaFloat_ptr dC,    magma_int_t lddc,
    float    *hwork, magma_int_t lwork,
    magmaFloat_ptr dT,    magma_int_t nb,
    magma_int_t *info)
{
    #define dA(a_1,a_2) (dA + (a_1) + (a_2)*ldda)
    #define dC(a_1,a_2) (dC + (a_1) + (a_2)*lddc)
    #define dT(a_1)     (dT + (a_1)*nb)

    float c_one = MAGMA_S_ONE;

    const char* side_  = lapack_side_const( side  );
    const char* trans_ = lapack_trans_const( trans );

    magmaFloat_ptr dwork;
    magma_int_t i, lddwork;
    magma_int_t i1, i2, step, ib, ic, jc, ma, mi, ni, nq, nw;
    int left, notran, lquery;
    magma_int_t lwkopt;

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
    lwkopt = (nq - k + nb)*(nw + nb) + nw*nb;
    hwork[0] = MAGMA_S_MAKE( lwkopt, 0 );
    
    if ( ! left && side != MagmaRight ) {
        *info = -1;
    } else if ( ! notran && trans != MagmaTrans ) {
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
    } else if (lwork < lwkopt && ! lquery) {
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
        hwork[0] = c_one;
        return *info;
    }

    lddwork = k;
    dwork = dT(2*lddwork);

    if ( (left && (! notran)) || ((! left) && notran) ) {
        // left  trans:    Q^T C
        // right notrans:  C Q
        // multiply from first block, i = 0, to next-to-last block, i < k-nb
        i1 = 0;
        i2 = k-nb;
        step = nb;
    } else {
        // left  notrans:  Q C
        // right trans:    C Q^T
        // multiply from next-to-last block, i = floor((k-1-nb)/nb)*nb, to first block, i = 0
        i1 = ((k - 1 - nb) / nb) * nb;
        i2 = 0;
        step = -nb;
    }

    if (left) {
        ni = n;
        jc = 0;
    } else {
        mi = m;
        ic = 0;
    }
    
    /* Use unblocked code to multiply last or only block (cases Q*C or C*Q^T). */
    // workspace left:  A(mi*nb) + C(mi*ni) + work(ni*nb_la) = (m-k-nb)*nb + (m-k-nb)*n + n*nb
    // workspace right: A(ni*nb) + C(mi*ni) + work(mi*nb_la) = (n-k-nb)*nb + m*(n-k-nb) + m*nb
    if ( step < 0 ) {
        // i is beginning of last block
        i = i1 - step;
        if ( i >= k ) {
            i = i1;
        }
        ib = k - i;
        if (left) {
            // ni=n, jc=0, H or H^T is applied to C(i:m-1,0:n-1)
            mi = m - i;
            ma = mi;
            ic = i;
        }
        else {
            // mi=m, ic=0, H or H^T is applied to C(0:m-1,i:n-1)
            ni = n - i;
            ma = ni;
            jc = i;
        }
        
        float* hA = hwork;
        float* hC = hwork + ma*ib;
        float* hW = hwork + ma*ib + mi*ni;
        magma_int_t lhwork = lwork - (ma*ib + mi*ni);
        
        magma_sgetmatrix( ma, ib, dA(i,  i ), ldda, hA, ma );
        magma_sgetmatrix( mi, ni, dC(ic, jc), lddc, hC, mi );

        lapackf77_sormqr( side_, trans_,
                          &mi, &ni, &ib,
                          hA, &ma, tau+i,
                          hC, &mi,
                          hW, &lhwork, info );

        // send the updated part of C back to the GPU
        magma_ssetmatrix( mi, ni, hC, mi, dC(ic, jc), lddc );
    }

    /* Use blocked code to multiply blocks */
    if (nb < k) {
        for( i=i1; (step < 0 ? i >= i2 : i < i2); i += step ) {
            ib = min(nb, k - i);
            if (left) {
                // ni=n, jc=0, H or H^T is applied to C(i:m-1,0:n-1)
                mi = m - i;
                ic = i;
            }
            else {
                // mi=m, ic=0, H or H^T is applied to C(0:m-1,i:n-1)
                ni = n - i;
                jc = i;
            }
            
            magma_slarfb_gpu( side, trans, MagmaForward, MagmaColumnwise,
                              mi, ni, ib,
                              dA(i,  i ), ldda, dT(i), nb,
                              dC(ic, jc), lddc, dwork, nw );
        }
    }
    else {
        i = i1;
    }

    /* Use unblocked code to multiply the last or only block (cases Q^T*C or C*Q). */
    if ( step > 0 ) {
        ib = k-i;
        if (left) {
            // ni=n, jc=0, H or H^T is applied to C(i:m-1,0:n-1)
            mi = m - i;
            ma = mi;
            ic = i;
        }
        else {
            // mi=m, ic=0, H or H^T is applied to C(0:m-1,i:n-1)
            ni = n - i;
            ma = ni;
            jc = i;
        }
        
        float* hA = hwork;
        float* hC = hwork + ma*ib;
        float* hW = hwork + ma*ib + mi*ni;
        magma_int_t lhwork = lwork - (ma*ib + mi*ni);
        
        magma_sgetmatrix( ma, ib, dA(i,  i ), ldda, hA, ma );
        magma_sgetmatrix( mi, ni, dC(ic, jc), lddc, hC, mi );

        lapackf77_sormqr( side_, trans_,
                          &mi, &ni, &ib,
                          hA, &ma, tau+i,
                          hC, &mi,
                          hW, &lhwork, info );
        
        // send the updated part of C back to the GPU
        magma_ssetmatrix( mi, ni, hC, mi, dC(ic, jc), lddc );
    }
    
    // TODO sync. For cases Q*C and C*Q^T, last call is magma_slarfb_gpu,
    // which is async magma_gemm calls, so sormqr can be unfinished.

    // TODO: sgeqrs_gpu ASSUMES that hwork contains the last block of A and C.
    // That needs to be fixed, but until then, don't modify hwork[0] here.
    // In LAPACK: On exit, if INFO = 0, HWORK[0] returns the optimal LWORK.
    //hwork[0] = MAGMA_S_MAKE( lwkopt, 0 );
    return *info;
} /* magma_sormqr_gpu */
