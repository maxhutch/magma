/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:36 2013
       @author Mark Gates
*/
#include "common_magma.h"

#define PRECISION_s

extern "C" magma_int_t
magma_slahru_m(
    magma_int_t n, magma_int_t ihi, magma_int_t k, magma_int_t nb,
    float *A, magma_int_t lda,
    struct sgehrd_data* data )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    SLAHRU is an auxiliary MAGMA routine that is used in SGEHRD to update
    the trailing sub-matrices after the reductions of the corresponding
    panels.
    See further details below.

    Arguments
    =========
    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    IHI     (input) INTEGER
            Last row to update. Same as IHI in sgehrd.

    K       (input) INTEGER
            Number of rows of the matrix Am (see details below)

    NB      (input) INTEGER
            Block size

    A       (output) REAL array, dimension (LDA,N-K)
            On entry, the N-by-(N-K) general matrix to be updated. The
            computation is done on the GPU. After Am is updated on the GPU
            only Am(1:NB) is transferred to the CPU - to update the
            corresponding Am matrix. See Further Details below.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    DA      (input/output) REAL array on the GPU, dimension
            (N,N-K). On entry, the N-by-(N-K) general matrix to be updated.
            On exit, the 1st K rows (matrix Am) of A are updated by
            applying an orthogonal transformation from the right
            Am = Am (I-V T V'), and sub-matrix Ag is updated by
            Ag = (I - V T V') Ag (I - V T V(NB+1:)' )
            where Q = I - V T V' represent the orthogonal matrix
            (as a product of elementary reflectors V) used to reduce
            the current panel of A to upper Hessenberg form. After Am
            is updated Am(:,1:NB) is sent to the CPU.
            See Further Details below.

    DY      (input/workspace) REAL array on the GPU, dimension
            (N, NB). On entry the (N-K)-by-NB Y = A V. It is used internally
            as workspace, so its value is changed on exit.

    DV      (input/workspace) REAL array on the GPU, dimension
            (N, NB). On entry the (N-K)-by-NB matrix V of elementary reflectors
            used to reduce the current panel of A to upper Hessenberg form.
            The rest K-by-NB part is used as workspace. V is unchanged on
            exit.

    DT      (input) REAL array on the GPU, dimension (NB, NB).
            On entry the NB-by-NB upper trinagular matrix defining the
            orthogonal Hessenberg reduction transformation matrix for
            the current panel. The lower triangular part are 0s.

    DWORK   (workspace) REAL array on the GPU, dimension N*NB.

    Further Details
    ===============
    This implementation follows the algorithm and notations described in:

    S. Tomov and J. Dongarra, "Accelerating the reduction to upper Hessenberg
    form through hybrid GPU-based computing," University of Tennessee Computer
    Science Technical Report, UT-CS-09-642 (also LAPACK Working Note 219),
    May 24, 2009.

    The difference is that here Am is computed on the GPU.
    M is renamed Am, G is renamed Ag.
    =====================================================================    */

    #define dA(  d, i, j ) (data->A [d] + (i) + (j)*ldda)
    #define dTi( d       ) (data->Ti[d])
    #define dV(  d, i, j ) (data->V [d] + (i) + (j)*ldv )
    #define dVd( d, i, j ) (data->Vd[d] + (i) + (j)*ldvd)
    #define dW(  d, i, j ) (data->W [d] + (i) + (j)*ldda)
    #define dY(  d, i, j ) (data->Y [d] + (i) + (j)*ldda)
    
    float c_zero    = MAGMA_S_ZERO;
    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;

    magma_int_t ngpu = data->ngpu;
    magma_int_t ldda = data->ldda;
    magma_int_t ldv  = data->ldv;
    magma_int_t ldvd = data->ldvd;
    
    magma_int_t d;
    magma_int_t dk, dkhi, dknb, dn;
    
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
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    for( d = 0; d < ngpu; ++d ) {
        magma_setdevice( d );
        magmablasSetKernelStream( data->streams[d] );
        
        // convert global indices (k) to local indices (dk)
        magma_indices_1D_bcyclic( nb, ngpu, d, k,    ihi, &dk,   &dkhi );
        magma_indices_1D_bcyclic( nb, ngpu, d, k+nb, n,   &dknb, &dn   );
        
        // -----
        // on right, A := A Q = A - A V T V'
        // Update Am = Am - Am V T Vd' = Am - Ym Wd', with Wd = Vd T'
        // Wd = Vd T' = V(k:ihi-1, 0:nb-1) * T(0:nb-1, 0:nb-1)'
        // Vd and Wd are the portions corresponding to the block cyclic dkstribution
        magma_sgemm( MagmaNoTrans, MagmaTrans, dkhi-dk, nb, nb,
                     c_one,  dVd(d, dk, 0), ldvd,
                             dTi(d),        nb,
                     c_zero, dW (d, dk, 0), ldda );
        
        // Am = Am - Ym Wd' = A(0:k-1, k:ihi-1) - Ym(0:k-1, 0:nb-1) * W(k:ihi-1, 0:nb-1)'
        magma_sgemm( MagmaNoTrans, MagmaTrans, k, dkhi-dk, nb,
                     c_neg_one, dY(d, 0,  0),  ldda,
                                dW(d, dk, 0),  ldda,
                     c_one,     dA(d, 0,  dk), ldda );

        // -----
        // on right, A := A Q = A - A V T V'
        // Update Ag = Ag - Ag V T V' = Ag - Yg Wd'
        // Ag = Ag - Yg Wd' = A(k:ihi-1, nb:ihi-k-1) - Y(k:ihi-1, 0:nb-1) * W(k+nb:ihi-1, 0:nb-1)'
        magma_sgemm( MagmaNoTrans, MagmaTrans, ihi-k, dkhi-dknb, nb,
                     c_neg_one, dY(d, k,    0),    ldda,
                                dW(d, dknb, 0),    ldda,
                     c_one,     dA(d, k,    dknb), ldda );
        
        // -----
        // on left, A := Q' A = A - V T' V' A
        // Ag2 = Ag2 - V T' V' Ag2 = W Yg, with W = V T' and Yg = V' Ag2
        // Note that Ag is A(k:ihi, nb+1:ihi-k)
        // while    Ag2 is A(k:ihi, nb+1: n -k)
        
        // here V and W are the whole matrices, not just block cyclic portion
        // W = V T' = V(k:ihi-1, 0:nb-1) * T(0:nb-1, 0:nb-1)'
        // TODO would it be cheaper to compute the whole matrix and
        // copy the block cyclic portions to another workspace?
        magma_sgemm( MagmaNoTrans, MagmaTrans, ihi-k, nb, nb,
                     c_one,  dV (d, k, 0), ldv,
                             dTi(d),       nb,
                     c_zero, dW (d, k, 0), ldda );
        
        // Z = V(k:ihi-1, 0:nb-1)' * A(k:ihi-1, nb:n-k-1);  Z is stored over Y
        magma_sgemm( MagmaTrans, MagmaNoTrans, nb, dn-dknb, ihi-k,
                     c_one,  dV(d, k, 0),    ldv,
                             dA(d, k, dknb), ldda,
                     c_zero, dY(d, 0, 0),    nb );
        
        // Ag2 = Ag2 - W Z = A(k:ihi-1, k+nb:n-1) - W(k+nb:n-1, 0:nb-1) * Z(0:nb-1, k+nb:n-1)
        magma_sgemm( MagmaNoTrans, MagmaNoTrans, ihi-k, dn-dknb, nb,
                     c_neg_one, dW(d, k, 0),    ldda,
                                dY(d, 0, 0),    nb,
                     c_one,     dA(d, k, dknb), ldda );
    }
        
    return 0;
}
