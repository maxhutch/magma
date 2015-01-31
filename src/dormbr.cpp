/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates

       @generated from zunmbr.cpp normal z -> d, Fri Jan 30 19:00:19 2015

*/
#include "common_magma.h"

/*
 * Version 1 - LAPACK
 * Version 2 - MAGMA
 */
#define VERSION 2

/**
    Purpose
    -------
    If VECT = MagmaQ, DORMBR overwrites the general real M-by-N matrix C with
                                 SIDE = MagmaLeft     SIDE = MagmaRight
    TRANS = MagmaNoTrans:        Q*C                  C*Q
    TRANS = MagmaTrans:     Q**H*C               C*Q**H
    
    If VECT = MagmaP, DORMBR overwrites the general real M-by-N matrix C with
                                 SIDE = MagmaLeft     SIDE = MagmaRight
    TRANS = MagmaNoTrans:        P*C                  C*P
    TRANS = MagmaTrans:     P**H*C               C*P**H
    
    Here Q and P**H are the unitary matrices determined by DGEBRD when
    reducing A real matrix A to bidiagonal form: A = Q*B * P**H. Q
    and P**H are defined as products of elementary reflectors H(i) and
    G(i) respectively.
    
    Let nq = m if SIDE = MagmaLeft and nq = n if SIDE = MagmaRight. Thus nq is the
    order of the unitary matrix Q or P**H that is applied.
    
    If VECT = MagmaQ, A is assumed to have been an NQ-by-K matrix:
    if nq >= k, Q = H(1) H(2) . . . H(k);
    if nq <  k, Q = H(1) H(2) . . . H(nq-1).
    
    If VECT = MagmaP, A is assumed to have been A K-by-NQ matrix:
    if k <  nq, P = G(1) G(2) . . . G(k);
    if k >= nq, P = G(1) G(2) . . . G(nq-1).
    
    Arguments
    ---------
    @param[in]
    vect    magma_vect_t
      -     = MagmaQ: apply Q or Q**H;
      -     = MagmaP: apply P or P**H.
    
    @param[in]
    side    magma_side_t
      -     = MagmaLeft:  apply Q, Q**H, P or P**H from the Left;
      -     = MagmaRight: apply Q, Q**H, P or P**H from the Right.
    
    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:    No transpose, apply Q or P;
      -     = MagmaTrans: Conjugate transpose, apply Q**H or P**H.
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix C. M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix C. N >= 0.
    
    @param[in]
    k       INTEGER
            If VECT = MagmaQ, the number of columns in the original
            matrix reduced by DGEBRD.
            If VECT = MagmaP, the number of rows in the original
            matrix reduced by DGEBRD.
            K >= 0.
    
    @param[in]
    A       DOUBLE_PRECISION array, dimension
                                  (LDA,min(nq,K)) if VECT = MagmaQ
                                  (LDA,nq)        if VECT = MagmaP
            The vectors which define the elementary reflectors H(i) and
            G(i), whose products determine the matrices Q and P, as
            returned by DGEBRD.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.
            If VECT = MagmaQ, LDA >= max(1,nq);
            if VECT = MagmaP, LDA >= max(1,min(nq,K)).
    
    @param[in]
    tau     DOUBLE_PRECISION array, dimension (min(nq,K))
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i) or G(i) which determines Q or P, as returned
            by DGEBRD in the array argument TAUQ or TAUP.
    
    @param[in,out]
    C       DOUBLE_PRECISION array, dimension (LDC,N)
            On entry, the M-by-N matrix C.
            On exit, C is overwritten by Q*C or Q**H*C or C*Q**H or C*Q
            or P*C or P**H*C or C*P or C*P**H.
    
    @param[in]
    ldc     INTEGER
            The leading dimension of the array C. LDC >= max(1,M).
    
    @param[out]
    work    (workspace) DOUBLE_PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.
    
    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If SIDE = MagmaLeft,  LWORK >= max(1,N);
            if SIDE = MagmaRight, LWORK >= max(1,M);
            if N = 0 or M = 0, LWORK >= 1.
            For optimum performance
            if SIDE = MagmaLeft,  LWORK >= max(1,N*NB);
            if SIDE = MagmaRight, LWORK >= max(1,M*NB),
            where NB is the optimal blocksize. (NB = 0 if M = 0 or N = 0.)
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
    

    @ingroup magma_dgesvd_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dormbr(
    magma_vect_t vect, magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t k,
    double *A, magma_int_t lda,
    double *tau,
    double *C, magma_int_t ldc,
    double *work, magma_int_t lwork,
    magma_int_t *info)
{
    #define A(i,j)  (A + (i) + (j)*lda)
    #define C(i,j)  (C + (i) + (j)*ldc)
            
    magma_int_t i1, i2, nb, mi, ni, nq, nq_1, nw, iinfo, lwkopt;
    magma_int_t left, notran, applyq, lquery;
    magma_trans_t transt;
    
    MAGMA_UNUSED( nq_1 );  // used only in version 1

    *info = 0;
    applyq = (vect  == MagmaQ);
    left   = (side  == MagmaLeft);
    notran = (trans == MagmaNoTrans);
    lquery = (lwork == -1);

    /* NQ is the order of Q or P and NW is the minimum dimension of WORK */
    if (left) {
        nq = m;
        nw = n;
    }
    else {
        nq = n;
        nw = m;
    }
    if (m == 0 || n == 0) {
        nw = 0;
    }
    
    /* check arguments */
    if (! applyq && vect != MagmaP) {
        *info = -1;
    }
    else if (! left && side != MagmaRight) {
        *info = -2;
    }
    else if (! notran && trans != MagmaTrans) {
        *info = -3;
    }
    else if (m < 0) {
        *info = -4;
    }
    else if (n < 0) {
        *info = -5;
    }
    else if (k < 0) {
        *info = -6;
    }
    else if ( (   applyq && lda < max(1,nq)        ) ||
              ( ! applyq && lda < max(1,min(nq,k)) ) ) {
        *info = -8;
    }
    else if (ldc < max(1,m)) {
        *info = -11;
    }
    else if (lwork < max(1,nw) && ! lquery) {
        *info = -13;
    }

    if (*info == 0) {
        if (nw > 0) {
            // TODO have get_dormqr_nb and get_dormlq_nb routines? see original LAPACK dormbr.
            // TODO make them dependent on m, n, and k?
            nb = magma_get_dgebrd_nb( min( m, n ));
            lwkopt = max(1, nw*nb);
        }
        else {
            lwkopt = 1;
        }
        work[0] = MAGMA_D_MAKE( lwkopt, 0 );
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

    if (applyq) {
        /* Apply Q */
        if (nq >= k) {
            /* Q was determined by a call to DGEBRD with nq >= k */
            #if VERSION == 1
            lapackf77_dormqr( lapack_side_const(side), lapack_trans_const(trans),
                              &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &iinfo);
            #else
            magma_dormqr( side, trans,
                          m, n, k, A, lda, tau, C, ldc, work, lwork, &iinfo);
            #endif
        }
        else if (nq > 1) {
            /* Q was determined by a call to DGEBRD with nq < k */
            if (left) {
                mi = m - 1;
                ni = n;
                i1 = 1;
                i2 = 0;
            }
            else {
                mi = m;
                ni = n - 1;
                i1 = 0;
                i2 = 1;
            }
            #if VERSION == 1
            nq_1 = nq - 1;
            lapackf77_dormqr( lapack_side_const(side), lapack_trans_const(trans),
                              &mi, &ni, &nq_1, A(1,0), &lda, tau, C(i1,i2), &ldc, work, &lwork, &iinfo);
            #else
            magma_dormqr( side, trans,
                          mi, ni, nq-1, A(1,0), lda, tau, C(i1,i2), ldc, work, lwork, &iinfo);
            #endif
        }
    }
    else {
        /* Apply P */
        if (notran) {
            transt = MagmaTrans;
        }
        else {
            transt = MagmaNoTrans;
        }
        if (nq > k) {
            /* P was determined by a call to DGEBRD with nq > k */
            #if VERSION == 1
            lapackf77_dormlq( lapack_side_const(side), lapack_trans_const(transt),
                              &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &iinfo);
            #else
            magma_dormlq( side, transt,
                          m, n, k, A, lda, tau, C, ldc, work, lwork, &iinfo);
            #endif
        }
        else if (nq > 1) {
            /* P was determined by a call to DGEBRD with nq <= k */
            if (left) {
                mi = m - 1;
                ni = n;
                i1 = 1;
                i2 = 0;
            }
            else {
                mi = m;
                ni = n - 1;
                i1 = 0;
                i2 = 1;
            }
            #if VERSION == 1
            nq_1 = nq - 1;
            lapackf77_dormlq( lapack_side_const(side), lapack_trans_const(transt),
                              &mi, &ni, &nq_1, A(0,1), &lda, tau, C(i1,i2), &ldc, work, &lwork, &iinfo);
            #else
            magma_dormlq( side, transt,
                          mi, ni, nq-1, A(0,1), lda, tau, C(i1,i2), ldc, work, lwork, &iinfo);
            #endif
        }
    }
    work[0] = MAGMA_D_MAKE( lwkopt, 0 );
    return *info;
} /* magma_dormbr */
