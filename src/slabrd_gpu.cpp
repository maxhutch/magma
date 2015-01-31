/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Stan Tomov
       @generated from zlabrd_gpu.cpp normal z -> s, Fri Jan 30 19:00:19 2015

*/
#include "common_magma.h"

#define PRECISION_s

/**
    Purpose
    -------
    SLABRD reduces the first NB rows and columns of a real general
    m by n matrix A to upper or lower bidiagonal form by an orthogonal
    transformation Q' * A * P, and returns the matrices X and Y which
    are needed to apply the transformation to the unreduced part of A.

    If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
    bidiagonal form.

    This is an auxiliary routine called by SGEBRD.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows in the matrix A.

    @param[in]
    n       INTEGER
            The number of columns in the matrix A.

    @param[in]
    nb      INTEGER
            The number of leading rows and columns of A to be reduced.

    @param[in,out]
    A       REAL array, dimension (LDA,N)
            On entry, the m by n general matrix to be reduced.
            On exit, the first NB rows and columns of the matrix are
            overwritten; the rest of the array is unchanged.
            If m >= n, elements on and below the diagonal in the first NB
              columns, with the array TAUQ, represent the orthogonal
              matrix Q as a product of elementary reflectors; and
              elements above the diagonal in the first NB rows, with the
              array TAUP, represent the orthogonal matrix P as a product
              of elementary reflectors.
    \n
            If m < n, elements below the diagonal in the first NB
              columns, with the array TAUQ, represent the orthogonal
              matrix Q as a product of elementary reflectors, and
              elements on and above the diagonal in the first NB rows,
              with the array TAUP, represent the orthogonal matrix P as
              a product of elementary reflectors.
            See Further Details.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[in,out]
    dA      REAL array, dimension (LDDA,N)
            Copy of A on GPU.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).

    @param[out]
    d       REAL array, dimension (NB)
            The diagonal elements of the first NB rows and columns of
            the reduced matrix.  D(i) = A(i,i).

    @param[out]
    e       REAL array, dimension (NB)
            The off-diagonal elements of the first NB rows and columns of
            the reduced matrix.

    @param[out]
    tauq    REAL array dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix Q. See Further Details.

    @param[out]
    taup    REAL array, dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix P. See Further Details.

    @param[out]
    X       REAL array, dimension (LDX,NB)
            The m-by-nb matrix X required to update the unreduced part
            of A.

    @param[in]
    ldx     INTEGER
            The leading dimension of the array X. LDX >= M.

    @param[out]
    dX      REAL array, dimension (LDDX,NB)
            Copy of X on GPU.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX. LDDX >= M.

    @param[out]
    Y       REAL array, dimension (LDY,NB)
            The n-by-nb matrix Y required to update the unreduced part
            of A.

    @param[in]
    ldy     INTEGER
            The leading dimension of the array Y. LDY >= N.

    @param[out]
    dY      REAL array, dimension (LDDY,NB)
            Copy of Y on GPU.

    @param[in]
    lddy    INTEGER
            The leading dimension of the array dY. LDDY >= N.

    Further Details
    ---------------
    The matrices Q and P are represented as products of elementary
    reflectors:

       Q = H(1) H(2) . . . H(nb)  and  P = G(1) G(2) . . . G(nb)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are real scalars, and v and u are real vectors.

    If m >= n, v(1:i-1) = 0, v(i) = 1, and v(i:m) is stored on exit in
    A(i:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+1:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n, v(1:i) = 0, v(i+1) = 1, and v(i+1:m) is stored on exit in
    A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    The elements of the vectors v and u together form the m-by-nb matrix
    V and the nb-by-n matrix U' which are needed, with X and Y, to apply
    the transformation to the unreduced part of the matrix, using a block
    update of the form:  A := A - V*Y' - X*U'.

    The contents of A on exit are illustrated by the following examples
    with nb = 2:

    @verbatim
    m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):

      (  1   1   u1  u1  u1 )           (  1   u1  u1  u1  u1  u1 )
      (  v1  1   1   u2  u2 )           (  1   1   u2  u2  u2  u2 )
      (  v1  v2  a   a   a  )           (  v1  1   a   a   a   a  )
      (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
      (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
      (  v1  v2  a   a   a  )
    @endverbatim

    where a denotes an element of the original matrix which is unchanged,
    vi denotes an element of the vector defining H(i), and ui an element
    of the vector defining G(i).

    @ingroup magma_sgesvd_aux
    ********************************************************************/
extern "C" magma_int_t
magma_slabrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    float     *A, magma_int_t lda,
    magmaFloat_ptr dA, magma_int_t ldda,
    float *d, float *e, float *tauq, float *taup,
    float     *X, magma_int_t ldx,
    magmaFloat_ptr dX, magma_int_t lddx,
    float     *Y, magma_int_t ldy,
    magmaFloat_ptr dY, magma_int_t lddy)
{
    #define A(i_,j_) (A + (i_) + (j_)*lda)
    #define X(i_,j_) (X + (i_) + (j_)*ldx)
    #define Y(i_,j_) (Y + (i_) + (j_)*ldy)
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    #define dY(i_,j_) (dY + (i_) + (j_)*lddy)
    #define dX(i_,j_) (dX + (i_) + (j_)*lddx)
    
    float c_neg_one = MAGMA_S_NEG_ONE;
    float c_one     = MAGMA_S_ONE;
    float c_zero    = MAGMA_S_ZERO;
    magma_int_t ione = 1;
    
    magma_int_t i__2, i__3;
    magma_int_t i;
    float alpha;

    A  -= 1 + lda;
    X  -= 1 + ldx;
    dX -= 1 + lddx;
    Y  -= 1 + ldy;
    dY -= 1 + lddy;
    --d;
    --e;
    --tauq;
    --taup;

    /* Quick return if possible */
    magma_int_t info = 0;
    if (m <= 0 || n <= 0) {
        return info;
    }

    float *f;
    magma_queue_t stream;
    magma_queue_create( &stream );
    magma_smalloc_cpu( &f, max(n,m) );
    if ( f == NULL ) {
        info = MAGMA_ERR_HOST_ALLOC;
        return info;
    }
    
    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i = 1; i <= nb; ++i) {
            /*  Update A(i:m,i) */
            i__2 = m - i + 1;
            i__3 = i - 1;
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_slacgv( &i__3, Y(i,1), &ldy );
            #endif
            blasf77_sgemv( "No transpose", &i__2, &i__3, &c_neg_one,
                           A(i,1), &lda,
                           Y(i,1), &ldy, &c_one,
                           A(i,i), &ione );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_slacgv( &i__3, Y(i,1), &ldy );
            #endif
            blasf77_sgemv( "No transpose", &i__2, &i__3, &c_neg_one,
                           X(i,1), &ldx,
                           A(1,i), &ione, &c_one,
                           A(i,i), &ione );
            
            /* Generate reflection Q(i) to annihilate A(i+1:m,i) */
            alpha = *A(i,i);
            i__2 = m - i + 1;
            i__3 = i + 1;
            lapackf77_slarfg( &i__2, &alpha, A(min(i__3,m),i), &ione, &tauq[i] );
            d[i] = MAGMA_S_REAL( alpha );
            if (i < n) {
                *A(i,i) = c_one;

                /* Compute Y(i+1:n,i) */
                i__2 = m - i + 1;
                i__3 = n - i;

                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_ssetvector( i__2,
                                  A(i,i), 1,
                                  dA(i-1,i-1), 1 );
                // 2. Multiply ---------------------------------------------
                magma_sgemv( MagmaConjTrans, i__2, i__3, c_one,
                             dA(i-1,i),   ldda,
                             dA(i-1,i-1), ione, c_zero,
                             dY(i+1,i),   ione );
                
                // 3. Put the result back ----------------------------------
                magma_sgetmatrix_async( i__3, 1,
                                        dY(i+1,i), lddy,
                                        Y(i+1,i),  ldy, stream );
                i__2 = m - i + 1;
                i__3 = i - 1;
                blasf77_sgemv( MagmaConjTransStr, &i__2, &i__3, &c_one,
                               A(i,1), &lda,
                               A(i,i), &ione, &c_zero,
                               Y(1,i), &ione );

                i__2 = n - i;
                i__3 = i - 1;
                blasf77_sgemv( "N", &i__2, &i__3, &c_neg_one,
                               Y(i+1,1), &ldy,
                               Y(1,i),   &ione, &c_zero,
                               f,        &ione );
                i__2 = m - i + 1;
                i__3 = i - 1;
                blasf77_sgemv( MagmaConjTransStr, &i__2, &i__3, &c_one,
                               X(i,1), &ldx,
                               A(i,i), &ione, &c_zero,
                               Y(1,i), &ione );
                
                // 4. Sync to make sure the result is back ----------------
                magma_queue_sync( stream );

                if (i__3 != 0) {
                    i__2 = n - i;
                    blasf77_saxpy( &i__2, &c_one, f, &ione, Y(i+1,i), &ione );
                }

                i__2 = i - 1;
                i__3 = n - i;
                blasf77_sgemv( MagmaConjTransStr, &i__2, &i__3, &c_neg_one,
                               A(1,i+1), &lda,
                               Y(1,i),   &ione, &c_one,
                               Y(i+1,i), &ione );
                i__2 = n - i;
                blasf77_sscal( &i__2, &tauq[i], Y(i+1,i), &ione );

                /* Update A(i,i+1:n) */
                i__2 = n - i;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_slacgv( &i__2, A(i,i+1), &lda );
                lapackf77_slacgv( &i,  A(i,1), &lda );
                #endif
                blasf77_sgemv( "No transpose", &i__2, &i, &c_neg_one,
                               Y(i+1,1), &ldy,
                               A(i,1),   &lda, &c_one,
                               A(i,i+1), &lda );
                i__2 = i - 1;
                i__3 = n - i;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_slacgv( &i,  A(i,1), &lda );
                lapackf77_slacgv( &i__2, X(i,1), &ldx );
                #endif
                blasf77_sgemv( MagmaConjTransStr, &i__2, &i__3, &c_neg_one,
                               A(1,i+1), &lda,
                               X(i,1),   &ldx, &c_one,
                               A(i,i+1), &lda );
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_slacgv( &i__2, X(i,1), &ldx );
                #endif

                /* Generate reflection P(i) to annihilate A(i,i+2:n) */
                i__2 = n - i;
                i__3 = i + 2;
                alpha = *A(i,i+1);
                lapackf77_slarfg( &i__2, &alpha, A(i,min(i__3,n)), &lda, &taup[i] );
                e[i] = MAGMA_S_REAL( alpha );
                *A(i,i+1) = c_one;

                /* Compute X(i+1:m,i) */
                i__2 = m - i;
                i__3 = n - i;
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_ssetvector( i__3,
                                  A(i,i+1), lda,
                                  dA(i-1,i), ldda );
                // 2. Multiply ---------------------------------------------
                //magma_scopy( i__3, dA(i-1,i), ldda, dY(1,1), 1 );
                magma_sgemv( MagmaNoTrans, i__2, i__3, c_one,
                             dA(i,i), ldda,
                             dA(i-1,i), ldda,
                             //dY(1,1), 1,
                             c_zero,
                             dX(i+1,i), ione );

                // 3. Put the result back ----------------------------------
                magma_sgetmatrix_async( i__2, 1,
                                        dX(i+1,i), lddx,
                                        X(i+1,i),  ldx, stream );

                i__2 = n - i;
                blasf77_sgemv( MagmaConjTransStr, &i__2, &i, &c_one,
                               Y(i+1,1), &ldy,
                               A(i,i+1), &lda, &c_zero,
                               X(1,i),   &ione );

                i__2 = m - i;
                blasf77_sgemv( "N", &i__2, &i, &c_neg_one,
                               A(i+1,1), &lda,
                               X(1,i),   &ione, &c_zero,
                               f,        &ione );
                i__2 = i - 1;
                i__3 = n - i;
                blasf77_sgemv( "N", &i__2, &i__3, &c_one,
                               A(1,i+1), &lda,
                               A(i,i+1), &lda, &c_zero,
                               X(1,i),   &ione );

                // 4. Sync to make sure the result is back ----------------
                magma_queue_sync( stream );
                if (i != 0) {
                    i__2 = m - i;
                    blasf77_saxpy( &i__2, &c_one, f, &ione, X(i+1,i), &ione );
                }


                i__2 = m - i;
                i__3 = i - 1;
                blasf77_sgemv( "No transpose", &i__2, &i__3, &c_neg_one,
                               X(i+1,1), &ldx,
                               X(1,i),   &ione, &c_one,
                               X(i+1,i), &ione );
                i__2 = m - i;
                blasf77_sscal( &i__2, &taup[i], X(i+1,i), &ione );

                #if defined(PRECISION_z) || defined(PRECISION_c)
                i__2 = n - i;
                lapackf77_slacgv( &i__2,  A(i,i+1), &lda );
                // 4. Send the block reflector  A(i+1:m,i) to the GPU after SLACGV()
                magma_ssetvector( i__2,
                                  A(i,i+1),  lda,
                                  dA(i-1,i), ldda );
                #endif
            }
        }
    }
    else {
        /* Reduce to lower bidiagonal form */
        for (i = 1; i <= nb; ++i) {
        
            /* Update A(i,i:n) */
            i__2 = n - i + 1;
            i__3 = i - 1;
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_slacgv( &i__2, A(i,i), &lda );
            lapackf77_slacgv( &i__3, A(i,1), &lda );
            #endif
            blasf77_sgemv( "No transpose", &i__2, &i__3, &c_neg_one,
                           Y(i,1), &ldy,
                           A(i,1), &lda, &c_one,
                           A(i,i), &lda );
            i__2 = i - 1;
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_slacgv( &i__3, A(i,1), &lda );
            lapackf77_slacgv( &i__3, X(i,1), &ldx );
            #endif
            i__3 = n - i + 1;
            blasf77_sgemv( MagmaConjTransStr, &i__2, &i__3, &c_neg_one,
                           A(1,i), &lda,
                           X(i,1), &ldx, &c_one,
                           A(i,i), &lda );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_slacgv( &i__2, X(i,1), &ldx );
            #endif
            
            /* Generate reflection P(i) to annihilate A(i,i+1:n) */
            i__2 = n - i + 1;
            i__3 = i + 1;
            alpha = *A(i,i);
            lapackf77_slarfg( &i__2, &alpha, A(i,min(i__3,n)), &lda, &taup[i] );
            d[i] = MAGMA_S_REAL( alpha );
            if (i < m) {
                *A(i,i) = c_one;
                
                /* Compute X(i+1:m,i) */
                i__2 = m - i;
                i__3 = n - i + 1;
                
                // 1. Send the block reflector  A(i,i+1:n) to the GPU ------
                magma_ssetvector( i__3,
                                  A(i,i), lda,
                                  dA(i-1,i-1), ldda );
                
                // 2. Multiply ---------------------------------------------
                //magma_scopy( i__3, dA(i-1,i-1), ldda, dY(1,1), 1 );
                magma_sgemv( MagmaNoTrans, i__2, i__3, c_one,
                             dA(i,i-1), ldda,
                             dA(i-1,i-1), ldda,
                             //dY(1,1), 1,
                             c_zero,
                             dX(i+1,i), ione );
                
                // 3. Put the result back ----------------------------------
                magma_sgetmatrix_async( i__2, 1,
                                        dX(i+1,i), lddx,
                                        X(i+1,i),  ldx, stream );
                
                i__2 = n - i + 1;
                i__3 = i - 1;
                blasf77_sgemv( MagmaConjTransStr, &i__2, &i__3, &c_one,
                               Y(i,1), &ldy,
                               A(i,i), &lda, &c_zero,
                               X(1,i), &ione );
                i__2 = m - i;
                i__3 = i - 1;
                blasf77_sgemv( "No transpose", &i__2, &i__3, &c_neg_one,
                               A(i+1,1), &lda,
                               X(1,i),   &ione, &c_zero,
                               f,        &ione );
                
                i__2 = i - 1;
                i__3 = n - i + 1;
                blasf77_sgemv( "No transpose", &i__2, &i__3, &c_one,
                               A(1,i), &lda,
                               A(i,i), &lda, &c_zero,
                               X(1,i), &ione );
                
                // 4. Sync to make sure the result is back ----------------
                magma_queue_sync( stream );
                if (i__2 != 0) {
                    i__3 = m - i;
                    blasf77_saxpy( &i__3, &c_one, f, &ione, X(i+1,i), &ione );
                }
                
                i__2 = m - i;
                i__3 = i - 1;
                blasf77_sgemv( "No transpose", &i__2, &i__3, &c_neg_one,
                               X(i+1,1), &ldx,
                               X(1,i),   &ione, &c_one,
                               X(i+1,i), &ione );
                i__2 = m - i;
                blasf77_sscal( &i__2, &taup[i], X(i+1,i), &ione );
                i__2 = n - i + 1;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_slacgv( &i__2, A(i,i), &lda );
                magma_ssetvector( i__2,
                                  A(i,i), lda,
                                  dA(i-1,i-1), ldda );
                #endif
                
                /* Update A(i+1:m,i) */
                i__2 = m - i;
                i__3 = i - 1;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_slacgv( &i__3, Y(i,1), &ldy );
                #endif
                blasf77_sgemv( "No transpose", &i__2, &i__3, &c_neg_one,
                               A(i+1,1), &lda,
                               Y(i,1),   &ldy, &c_one,
                               A(i+1,i), &ione );
                i__2 = m - i;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_slacgv( &i__3, Y(i,1), &ldy );
                #endif
                blasf77_sgemv( "No transpose", &i__2, &i, &c_neg_one,
                               X(i+1,1), &ldx,
                               A(1,i),   &ione, &c_one,
                               A(i+1,i), &ione );
                
                /* Generate reflection Q(i) to annihilate A(i+2:m,i) */
                i__2 = m - i;
                i__3 = i + 2;
                alpha = *A(i+1,i);
                lapackf77_slarfg( &i__2, &alpha, A(min(i__3,m),i), &ione, &tauq[i] );
                e[i] = MAGMA_S_REAL( alpha );
                *A(i+1,i) = c_one;
                
                /* Compute Y(i+1:n,i) */
                i__2 = m - i;
                i__3 = n - i;
                
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_ssetvector( i__2,
                                  A(i+1,i), 1,
                                  dA(i,i-1), 1 );
                // 2. Multiply ---------------------------------------------
                magma_sgemv( MagmaConjTrans, i__2, i__3, c_one,
                             dA(i,i),   ldda,
                             dA(i,i-1), ione, c_zero,
                             dY(i+1,i), ione );
                
                // 3. Put the result back ----------------------------------
                magma_sgetmatrix_async( i__3, 1,
                                        dY(i+1,i), lddy,
                                        Y(i+1,i),  ldy, stream );
                
                i__2 = m - i;
                i__3 = i - 1;
                blasf77_sgemv( MagmaConjTransStr, &i__2, &i__3, &c_one,
                               A(i+1,1), &lda,
                               A(i+1,i), &ione, &c_zero,
                               Y(1,i),   &ione );
                i__2 = n - i;
                i__3 = i - 1;
                blasf77_sgemv( "No transpose", &i__2, &i__3, &c_neg_one,
                               Y(i+1,1), &ldy,
                               Y(1,i),   &ione, &c_zero,
                               f,        &ione );
                
                i__2 = m - i;
                blasf77_sgemv( MagmaConjTransStr, &i__2, &i, &c_one,
                               X(i+1,1), &ldx,
                               A(i+1,i), &ione, &c_zero,
                               Y(1,i),   &ione );
                
                // 4. Sync to make sure the result is back ----------------
                magma_queue_sync( stream );
                if (i__3 != 0) {
                    i__2 = n - i;
                    blasf77_saxpy( &i__2, &c_one, f, &ione, Y(i+1,i), &ione );
                }
                
                i__2 = n - i;
                blasf77_sgemv( MagmaConjTransStr, &i, &i__2, &c_neg_one,
                               A(1,i+1), &lda,
                               Y(1,i),   &ione, &c_one,
                               Y(i+1,i), &ione );
                i__2 = n - i;
                blasf77_sscal( &i__2, &tauq[i], Y(i+1,i), &ione );
            }
            #if defined(PRECISION_z) || defined(PRECISION_c)
            else {
                i__2 = n - i + 1;
                lapackf77_slacgv( &i__2, A(i,i), &lda );
                magma_ssetvector( i__2,
                                  A(i,i), lda,
                                  dA(i-1,i-1), ldda );
            }
            #endif
        }
    }
    
    magma_queue_destroy( stream );
    magma_free_cpu( f );
    
    return info;
} /* magma_slabrd_gpu */
