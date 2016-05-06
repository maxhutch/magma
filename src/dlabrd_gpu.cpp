/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Stan Tomov
       @author Mark Gates
       
       @generated from src/zlabrd_gpu.cpp normal z -> d, Mon May  2 23:30:24 2016

*/
#include "magma_internal.h"

#define REAL

/**
    Purpose
    -------
    DLABRD reduces the first NB rows and columns of a real general
    m by n matrix A to upper or lower bidiagonal form by an orthogonal
    transformation Q' * A * P, and returns the matrices X and Y which
    are needed to apply the transformation to the unreduced part of A.

    If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
    bidiagonal form.

    This is an auxiliary routine called by DGEBRD.

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
    A       DOUBLE PRECISION array, dimension (LDA,N)
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
    dA      DOUBLE PRECISION array, dimension (LDDA,N)
            Copy of A on GPU.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).

    @param[out]
    d       DOUBLE PRECISION array, dimension (NB)
            The diagonal elements of the first NB rows and columns of
            the reduced matrix.  D(i) = A(i,i).

    @param[out]
    e       DOUBLE PRECISION array, dimension (NB)
            The off-diagonal elements of the first NB rows and columns of
            the reduced matrix.

    @param[out]
    tauq    DOUBLE PRECISION array dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix Q. See Further Details.

    @param[out]
    taup    DOUBLE PRECISION array, dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix P. See Further Details.

    @param[out]
    X       DOUBLE PRECISION array, dimension (LDX,NB)
            The m-by-nb matrix X required to update the unreduced part
            of A.

    @param[in]
    ldx     INTEGER
            The leading dimension of the array X. LDX >= M.

    @param[out]
    dX      DOUBLE PRECISION array, dimension (LDDX,NB)
            Copy of X on GPU.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX. LDDX >= M.

    @param[out]
    Y       DOUBLE PRECISION array, dimension (LDY,NB)
            The n-by-nb matrix Y required to update the unreduced part
            of A.

    @param[in]
    ldy     INTEGER
            The leading dimension of the array Y. LDY >= N.

    @param[out]
    dY      DOUBLE PRECISION array, dimension (LDDY,NB)
            Copy of Y on GPU.

    @param[in]
    lddy    INTEGER
            The leading dimension of the array dY. LDDY >= N.

    @param
    work    DOUBLE PRECISION array, dimension (LWORK)
            Workspace.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK. LWORK >= max( M, N ).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

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

    @ingroup magma_dgesvd_aux
    ********************************************************************/
extern "C" magma_int_t
magma_dlabrd_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nb,
    double     *A, magma_int_t lda,
    magmaDouble_ptr dA, magma_int_t ldda,
    double *d, double *e, double *tauq, double *taup,
    double     *X, magma_int_t ldx,
    magmaDouble_ptr dX, magma_int_t lddx,
    double     *Y, magma_int_t ldy,
    magmaDouble_ptr dY, magma_int_t lddy,
    double  *work, magma_int_t lwork,
    magma_queue_t queue )
{
    #define  A(i_,j_) ( A + (i_) + (j_)*lda)
    #define  X(i_,j_) ( X + (i_) + (j_)*ldx)
    #define  Y(i_,j_) ( Y + (i_) + (j_)*ldy)
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    #define dY(i_,j_) (dY + (i_) + (j_)*lddy)
    #define dX(i_,j_) (dX + (i_) + (j_)*lddx)
    
    /* Constants */
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const double c_one     = MAGMA_D_ONE;
    const double c_zero    = MAGMA_D_ZERO;
    const magma_int_t ione = 1;
    
    /* Local variables */
    magma_int_t i, i1, m_i, m_i1, n_i, n_i1;
    double alpha;

    /* Quick return if possible */
    magma_int_t info = 0;
    if (m <= 0 || n <= 0) {
        return info;
    }

    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i=0; i < nb; ++i) {
            /* Update A(i:m,i) */
            i1   = i + 1;
            m_i  = m - i;
            m_i1 = m - (i+1);
            n_i1 = n - (i+1);
            #ifdef COMPLEX
            lapackf77_dlacgv( &i, Y(i,0), &ldy );
            #endif
            blasf77_dgemv( "No transpose", &m_i, &i, &c_neg_one,
                           A(i,0), &lda,
                           Y(i,0), &ldy, &c_one,
                           A(i,i), &ione );
            #ifdef COMPLEX
            lapackf77_dlacgv( &i, Y(i,0), &ldy );
            #endif
            blasf77_dgemv( "No transpose", &m_i, &i, &c_neg_one,
                           X(i,0), &ldx,
                           A(0,i), &ione, &c_one,
                           A(i,i), &ione );
            
            /* Generate reflection Q(i) to annihilate A(i+1:m,i) */
            alpha = *A(i,i);
            lapackf77_dlarfg( &m_i, &alpha, A(min(i+1,m-1),i), &ione, &tauq[i] );
            d[i] = MAGMA_D_REAL( alpha );
            if (i+1 < n) {
                *A(i,i) = c_one;

                /* Compute Y(i+1:n,i) */
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_dsetvector( m_i,
                                  A(i,i), 1,
                                  dA(i,i), 1, queue );
                // 2. Multiply ---------------------------------------------
                magma_dgemv( MagmaConjTrans, m_i, n_i1, c_one,
                             dA(i,i+1),   ldda,
                             dA(i,i), ione, c_zero,
                             dY(i+1,i),   ione, queue );
                
                // 3. Put the result back ----------------------------------
                magma_dgetmatrix_async( n_i1, 1,
                                        dY(i+1,i), lddy,
                                        Y(i+1,i),  ldy, queue );
                blasf77_dgemv( MagmaConjTransStr, &m_i, &i, &c_one,
                               A(i,0), &lda,
                               A(i,i), &ione, &c_zero,
                               Y(0,i), &ione );

                blasf77_dgemv( "N", &n_i1, &i, &c_neg_one,
                               Y(i+1,0), &ldy,
                               Y(0,i),   &ione, &c_zero,
                               work,     &ione );
                blasf77_dgemv( MagmaConjTransStr, &m_i, &i, &c_one,
                               X(i,0), &ldx,
                               A(i,i), &ione, &c_zero,
                               Y(0,i), &ione );
                
                // 4. Sync to make sure the result is back ----------------
                magma_queue_sync( queue );

                if (i != 0) {
                    blasf77_daxpy( &n_i1, &c_one, work, &ione, Y(i+1,i), &ione );
                }

                blasf77_dgemv( MagmaConjTransStr, &i, &n_i1, &c_neg_one,
                               A(0,i+1), &lda,
                               Y(0,i),   &ione, &c_one,
                               Y(i+1,i), &ione );
                blasf77_dscal( &n_i1, &tauq[i], Y(i+1,i), &ione );

                /* Update A(i,i+1:n) */
                #ifdef COMPLEX
                lapackf77_dlacgv( &n_i1, A(i,i+1), &lda );
                lapackf77_dlacgv( &i1,  A(i,0), &lda );
                #endif
                blasf77_dgemv( "No transpose", &n_i1, &i1, &c_neg_one,
                               Y(i+1,0), &ldy,
                               A(i,0),   &lda, &c_one,
                               A(i,i+1), &lda );
                #ifdef COMPLEX
                lapackf77_dlacgv( &i1,  A(i,0), &lda );
                lapackf77_dlacgv( &i, X(i,0), &ldx );
                #endif
                blasf77_dgemv( MagmaConjTransStr, &i, &n_i1, &c_neg_one,
                               A(0,i+1), &lda,
                               X(i,0),   &ldx, &c_one,
                               A(i,i+1), &lda );
                #ifdef COMPLEX
                lapackf77_dlacgv( &i, X(i,0), &ldx );
                #endif

                /* Generate reflection P(i) to annihilate A(i,i+2:n) */
                alpha = *A(i,i+1);
                lapackf77_dlarfg( &n_i1, &alpha, A(i,min(i+2,n-1)), &lda, &taup[i] );
                e[i] = MAGMA_D_REAL( alpha );
                *A(i,i+1) = c_one;

                /* Compute X(i+1:m,i) */
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_dsetvector( n_i1,
                                  A(i,i+1), lda,
                                  dA(i,i+1), ldda, queue );
                
                // 2. Multiply ---------------------------------------------
                magma_dgemv( MagmaNoTrans, m_i1, n_i1, c_one,
                             dA(i+1,i+1), ldda,
                             dA(i,i+1), ldda,
                             //dY(0,0), 1,
                             c_zero,
                             dX(i+1,i), ione, queue );

                // 3. Put the result back ----------------------------------
                magma_dgetmatrix_async( m_i1, 1,
                                        dX(i+1,i), lddx,
                                        X(i+1,i),  ldx, queue );

                blasf77_dgemv( MagmaConjTransStr, &n_i1, &i1, &c_one,
                               Y(i+1,0), &ldy,
                               A(i,i+1), &lda, &c_zero,
                               X(0,i),   &ione );

                blasf77_dgemv( "N", &m_i1, &i1, &c_neg_one,
                               A(i+1,0), &lda,
                               X(0,i),   &ione, &c_zero,
                               work,     &ione );
                blasf77_dgemv( "N", &i, &n_i1, &c_one,
                               A(0,i+1), &lda,
                               A(i,i+1), &lda, &c_zero,
                               X(0,i),   &ione );

                // 4. Sync to make sure the result is back ----------------
                magma_queue_sync( queue );
                if ((i+1) != 0) {
                    blasf77_daxpy( &m_i1, &c_one, work, &ione, X(i+1,i), &ione );
                }

                blasf77_dgemv( "No transpose", &m_i1, &i, &c_neg_one,
                               X(i+1,0), &ldx,
                               X(0,i),   &ione, &c_one,
                               X(i+1,i), &ione );
                blasf77_dscal( &m_i1, &taup[i], X(i+1,i), &ione );

                #ifdef COMPLEX
                lapackf77_dlacgv( &n_i1,  A(i,i+1), &lda );
                // 4. Send the block reflector  A(i+1:m,i) to the GPU after DLACGV()
                magma_dsetvector( n_i1,
                                  A(i,i+1),  lda,
                                  dA(i,i+1), ldda, queue );
                #endif
            }
        }
    }
    else {
        /* Reduce to lower bidiagonal form */
        for (i=0; i < nb; ++i) {
            /* Update A(i,i:n) */
            i1   = i + 1;
            m_i1 = m - (i+1);
            n_i  = n - i;
            n_i1 = n - (i+1);
            #ifdef COMPLEX
            lapackf77_dlacgv( &n_i, A(i,i), &lda );
            lapackf77_dlacgv( &i, A(i,0), &lda );
            #endif
            blasf77_dgemv( "No transpose", &n_i, &i, &c_neg_one,
                           Y(i,0), &ldy,
                           A(i,0), &lda, &c_one,
                           A(i,i), &lda );
            #ifdef COMPLEX
            lapackf77_dlacgv( &i, A(i,0), &lda );
            lapackf77_dlacgv( &i, X(i,0), &ldx );
            #endif
            blasf77_dgemv( MagmaConjTransStr, &i, &n_i, &c_neg_one,
                           A(0,i), &lda,
                           X(i,0), &ldx, &c_one,
                           A(i,i), &lda );
            #ifdef COMPLEX
            lapackf77_dlacgv( &i, X(i,0), &ldx );
            #endif
            
            /* Generate reflection P(i) to annihilate A(i,i+1:n) */
            alpha = *A(i,i);
            lapackf77_dlarfg( &n_i, &alpha, A(i,min(i+1,n-1)), &lda, &taup[i] );
            d[i] = MAGMA_D_REAL( alpha );
            if (i+1 < m) {
                *A(i,i) = c_one;
                
                /* Compute X(i+1:m,i) */
                // 1. Send the block reflector  A(i,i+1:n) to the GPU ------
                magma_dsetvector( n_i,
                                  A(i,i), lda,
                                  dA(i,i), ldda, queue );
                
                // 2. Multiply ---------------------------------------------
                magma_dgemv( MagmaNoTrans, m_i1, n_i, c_one,
                             dA(i+1,i), ldda,
                             dA(i,i), ldda,
                             //dY(0,0), 1,
                             c_zero,
                             dX(i+1,i), ione, queue );
                
                // 3. Put the result back ----------------------------------
                magma_dgetmatrix_async( m_i1, 1,
                                        dX(i+1,i), lddx,
                                        X(i+1,i),  ldx, queue );
                
                blasf77_dgemv( MagmaConjTransStr, &n_i, &i, &c_one,
                               Y(i,0), &ldy,
                               A(i,i), &lda, &c_zero,
                               X(0,i), &ione );
                
                blasf77_dgemv( "No transpose", &m_i1, &i, &c_neg_one,
                               A(i+1,0), &lda,
                               X(0,i),   &ione, &c_zero,
                               work,     &ione );
                
                blasf77_dgemv( "No transpose", &i, &n_i, &c_one,
                               A(0,i), &lda,
                               A(i,i), &lda, &c_zero,
                               X(0,i), &ione );
                
                // 4. Sync to make sure the result is back ----------------
                magma_queue_sync( queue );
                if (i != 0) {
                    blasf77_daxpy( &m_i1, &c_one, work, &ione, X(i+1,i), &ione );
                }
                
                blasf77_dgemv( "No transpose", &m_i1, &i, &c_neg_one,
                               X(i+1,0), &ldx,
                               X(0,i),   &ione, &c_one,
                               X(i+1,i), &ione );
                blasf77_dscal( &m_i1, &taup[i], X(i+1,i), &ione );
                #ifdef COMPLEX
                lapackf77_dlacgv( &n_i, A(i,i), &lda );
                magma_dsetvector( n_i,
                                  A(i,i), lda,
                                  dA(i,i), ldda, queue );
                #endif
                
                /* Update A(i+1:m,i) */
                #ifdef COMPLEX
                lapackf77_dlacgv( &i, Y(i,0), &ldy );
                #endif
                blasf77_dgemv( "No transpose", &m_i1, &i, &c_neg_one,
                               A(i+1,0), &lda,
                               Y(i,0),   &ldy, &c_one,
                               A(i+1,i), &ione );
                #ifdef COMPLEX
                lapackf77_dlacgv( &i, Y(i,0), &ldy );
                #endif
                blasf77_dgemv( "No transpose", &m_i1, &i1, &c_neg_one,
                               X(i+1,0), &ldx,
                               A(0,i),   &ione, &c_one,
                               A(i+1,i), &ione );
                
                /* Generate reflection Q(i) to annihilate A(i+2:m,i) */
                alpha = *A(i+1,i);
                lapackf77_dlarfg( &m_i1, &alpha, A(min(i+2,m-1),i), &ione, &tauq[i] );
                e[i] = MAGMA_D_REAL( alpha );
                *A(i+1,i) = c_one;
                
                /* Compute Y(i+1:n,i) */
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_dsetvector( m_i1,
                                  A(i+1,i), 1,
                                  dA(i+1,i), 1, queue );
                
                // 2. Multiply ---------------------------------------------
                magma_dgemv( MagmaConjTrans, m_i1, n_i1, c_one,
                             dA(i+1,i+1), ldda,
                             dA(i+1,i), ione, c_zero,
                             dY(i+1,i), ione, queue );
                
                // 3. Put the result back ----------------------------------
                magma_dgetmatrix_async( n_i1, 1,
                                        dY(i+1,i), lddy,
                                        Y(i+1,i),  ldy, queue );
                
                blasf77_dgemv( MagmaConjTransStr, &m_i1, &i, &c_one,
                               A(i+1,0), &lda,
                               A(i+1,i), &ione, &c_zero,
                               Y(0,i),   &ione );
                blasf77_dgemv( "No transpose", &n_i1, &i, &c_neg_one,
                               Y(i+1,0), &ldy,
                               Y(0,i),   &ione, &c_zero,
                               work,     &ione );
                
                blasf77_dgemv( MagmaConjTransStr, &m_i1, &i1, &c_one,
                               X(i+1,0), &ldx,
                               A(i+1,i), &ione, &c_zero,
                               Y(0,i),   &ione );
                
                // 4. Sync to make sure the result is back ----------------
                magma_queue_sync( queue );
                if (i != 0) {
                    blasf77_daxpy( &n_i1, &c_one, work, &ione, Y(i+1,i), &ione );
                }
                
                blasf77_dgemv( MagmaConjTransStr, &i1, &n_i1, &c_neg_one,
                               A(0,i+1), &lda,
                               Y(0,i),   &ione, &c_one,
                               Y(i+1,i), &ione );
                blasf77_dscal( &n_i1, &tauq[i], Y(i+1,i), &ione );
            }
            #ifdef COMPLEX
            else {
                lapackf77_dlacgv( &n_i, A(i,i), &lda );
                magma_dsetvector( n_i,
                                  A(i,i), lda,
                                  dA(i,i), ldda, queue );
            }
            #endif
        }
    }
    
    return info;
} /* magma_dlabrd_gpu */
