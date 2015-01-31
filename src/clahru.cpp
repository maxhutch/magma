/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlahru.cpp normal z -> c, Fri Jan 30 19:00:18 2015
       @author Stan Tomov
       @author Mark Gates
*/
#include "common_magma.h"

/**
    Purpose
    -------
    CLAHRU is an auxiliary MAGMA routine that is used in CGEHRD to update
    the trailing sub-matrices after the reductions of the corresponding
    panels.
    See further details below.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    ihi     INTEGER
            Last row to update. Same as IHI in cgehrd.

    @param[in]
    k       INTEGER
            Number of rows of the matrix Am (see details below)

    @param[in]
    nb      INTEGER
            Block size

    @param[out]
    A       COMPLEX array, dimension (LDA,N-K)
            On entry, the N-by-(N-K) general matrix to be updated. The
            computation is done on the GPU. After Am is updated on the GPU
            only Am(1:NB) is transferred to the CPU - to update the
            corresponding Am matrix. See Further Details below.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[in,out]
    dA      COMPLEX array on the GPU, dimension (LDDA,N-K).
            On entry, the N-by-(N-K) general matrix to be updated.
            On exit, the 1st K rows (matrix Am) of A are updated by
            applying an orthogonal transformation from the right
            Am = Am (I-V T V'), and sub-matrix Ag is updated by
            Ag = (I - V T V') Ag (I - V T V(NB+1:)' )
            where Q = I - V T V' represent the orthogonal matrix
            (as a product of elementary reflectors V) used to reduce
            the current panel of A to upper Hessenberg form. After Am
            is updated Am(:,1:NB) is sent to the CPU.
            See Further Details below.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).

    @param[in,out]
    dY      (workspace) COMPLEX array on the GPU, dimension (LDDY, NB).
            On entry the (N-K)-by-NB Y = A V. It is used internally
            as workspace, so its value is changed on exit.
    
    @param[in]
    lddy    INTEGER
            The leading dimension of the array dY.  LDDY >= max(1,N).

    @param[in,out]
    dV      (workspace) COMPLEX array on the GPU, dimension (LDDV, NB).
            On entry the (N-K)-by-NB matrix V of elementary reflectors
            used to reduce the current panel of A to upper Hessenberg form.
            The rest K-by-NB part is used as workspace. V is unchanged on
            exit.
    
    @param[in]
    lddv    INTEGER
            The leading dimension of the array dV.  LDDV >= max(1,N).

    @param[in]
    dT      COMPLEX array on the GPU, dimension (NB, NB).
            On entry the NB-by-NB upper trinagular matrix defining the
            orthogonal Hessenberg reduction transformation matrix for
            the current panel. The lower triangular part are 0s.

    @param
    dwork   (workspace) COMPLEX array on the GPU, dimension N*NB.

    Further Details
    ---------------
    This implementation follows the algorithm and notations described in:

    S. Tomov and J. Dongarra, "Accelerating the reduction to upper Hessenberg
    form through hybrid GPU-based computing," University of Tennessee Computer
    Science Technical Report, UT-CS-09-642 (also LAPACK Working Note 219),
    May 24, 2009.

    The difference is that here Am is computed on the GPU.
    M is renamed Am, G is renamed Ag.

    @ingroup magma_cgeev_aux
    ********************************************************************/
extern "C" magma_int_t
magma_clahru(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    magmaFloatComplex *A,  magma_int_t lda,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dY, magma_int_t lddy,
    magmaFloatComplex_ptr dV, magma_int_t lddv,
    magmaFloatComplex_ptr dT,
    magmaFloatComplex_ptr dwork )
{
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    
    magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;

    magmaFloatComplex *dYm = dV + ihi - k;

    magma_int_t info = 0;
    if (n < 0) {
        info = -1;
    } else if (ihi < 0 || ihi > n) {
        info = -2;
    } else if (k < 0 || k > n) {
        info = -3;
    } else if (nb < 1 || nb > n) {
        info = -4;
    } else if (lda < max(1,n)) {
        info = -6;
    } else if (ldda < max(1,n)) {
        info = -8;
    } else if (lddy < max(1,n)) {
        info = -10;
    } else if (lddv < max(1,n)) {
        info = -12;
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    // top part of Y, above panel, hasn't been computed yet, so do that now
    // Ym = Am V = A(0:k-1, 0:ihi-k-1) * V(0:ihi-k-1, 0:nb-1)
    magma_cgemm( MagmaNoTrans, MagmaNoTrans, k, nb, ihi-k,
                 c_one,  dA,  ldda,
                         dV,  lddv,
                 c_zero, dYm, ldda );

    // -----
    // on right, A := A Q = A - A V T V'
    // Update Am = Am - Am V T V' = Am - Ym W', with W = V T'
    // W = V T' = V(0:ihi-k-1, 0:nb-1) * T(0:nb-1, 0:nb-1)'
    magma_cgemm( MagmaNoTrans, MagmaConjTrans, ihi-k, nb, nb,
                 c_one,  dV,    lddv,
                         dT,    nb,
                 c_zero, dwork, ldda );

    // Am = Am - Ym W' = A(0:k-1, 0:ihi-k-1) - Ym(0:k-1, 0:nb-1) * W(0:ihi-k-1, 0:nb-1)'
    magma_cgemm( MagmaNoTrans, MagmaConjTrans, k, ihi-k, nb,
                 c_neg_one, dYm,   ldda,
                            dwork, ldda,
                 c_one,     dA,    ldda );
    
    // copy first nb columns of Am, A(0:k-1, 0:nb-1), to host
    magma_cgetmatrix( k, nb, dA, ldda, A, lda );

    // -----
    // on right, A := A Q = A - A V T V'
    // Update Ag = Ag - Ag V T V' = Ag - Y W'
    // Ag = Ag - Y W' = A(k:ihi-1, nb:ihi-k-1) - Y(0:ihi-k-1, 0:nb-1) * W(nb:ihi-k-1, 0:nb-1)'
    magma_cgemm( MagmaNoTrans, MagmaConjTrans, ihi-k, ihi-k-nb, nb,
                 c_neg_one, dY,         ldda,
                            dwork + nb, ldda,
                 c_one,     dA(k,nb),   ldda );

    // -----
    // on left, A := Q' A = A - V T' V' A
    // Ag2 = Ag2 - V T' V' Ag2 = W Yg, with W = V T' and Yg = V' Ag2
    // Note that Ag is A(k:ihi, nb+1:ihi-k)
    // while    Ag2 is A(k:ihi, nb+1: n -k)
    
    // Z = V(0:ihi-k-1, 0:nb-1)' * A(k:ihi-1, nb:n-k-1);  Z is stored over Y
    magma_cgemm( MagmaConjTrans, MagmaNoTrans, nb, n-k-nb, ihi-k,
                 c_one,  dV,       lddv,
                         dA(k,nb), ldda,
                 c_zero, dY,       nb );
    
    // Ag2 = Ag2 - W Z = A(k:ihi-1, nb:n-k-1) - W(nb:n-k-1, 0:nb-1) * Z(0:nb-1, nb:n-k-1)
    magma_cgemm( MagmaNoTrans, MagmaNoTrans, ihi-k, n-k-nb, nb,
                 c_neg_one, dwork,    ldda,
                            dY,       nb,
                 c_one,     dA(k,nb), ldda );
    
    return info;
}
