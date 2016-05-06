/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Raffaele Solca

       @generated from src/ztrsm_m.cpp normal z -> d, Mon May  2 23:30:21 2016
*/
#include "magma_internal.h"

extern "C" magma_int_t
magma_get_dtrsm_m_nb() { return 128; }

/**
    Purpose
    -------
    DTRSM solves one of the matrix equations
       op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,
    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op( A ) is one of

        op( A ) = A      or
        op( A ) = A**T   or
        op( A ) = A**H.

    The matrix X is overwritten on B.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    side    magma_side_t.
            On entry, SIDE specifies whether op( A ) appears on the left
            or right of X as follows:
         -     = MagmaLeft:        op( A )*X = alpha*B.
         -     = MagmaRight:       X*op( A ) = alpha*B.

    @param[in]
    uplo    magma_uplo_t.
            On entry, UPLO specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
         -     = MagmaUpper:   A is an upper triangular matrix.
         -     = MagmaLower:   A is a lower triangular matrix.

    @param[in]
    transa  magma_trans_t.
            On entry, TRANSA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
         -     = MagmaNoTrans:     op( A ) = A.
         -     = MagmaTrans:       op( A ) = A**T.
         -     = MagmaConjTrans:   op( A ) = A**H.

    @param[in]
    diag    magma_diag_t.
            On entry, DIAG specifies whether or not A is unit triangular
            as follows:
         -     = MagmaUnit:      A is assumed to be unit triangular.
         -     = MagmaNonUnit:   A is not assumed to be unit triangular.

    @param[in]
    m       INTEGER.
            On entry, M specifies the number of rows of B. M must be at
            least zero.

    @param[in]
    n       INTEGER.
            On entry, N specifies the number of columns of B. N must be
            at least zero.

    @param[in]
    alpha   DOUBLE PRECISION.
            On entry, ALPHA specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    A       DOUBLE PRECISION array of DIMENSION ( LDA, k ), where k is m
            when SIDE = MagmaLeft and is n when SIDE = MagmaRight.
            Before entry with UPLO = MagmaUpper, the leading k by k
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with UPLO = MagmaLower, the leading k by k
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when DIAG = MagmaUnit, the diagonal elements of
            A are not referenced either, but are assumed to be unity.

    @param[in]
    lda     INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program.
            When SIDE = MagmaLeft  then LDA >= max( 1, m ),
            when SIDE = MagmaRight then LDA >= max( 1, n ).

    @param[in,out]
    B       DOUBLE PRECISION array of DIMENSION ( LDB, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B, and on exit is
            overwritten by the solution matrix X.

    @param[in]
    ldb     INTEGER.
            On entry, LDB specifies the first dimension of B as declared
            in the calling (sub) program. LDB must be at least
            max( 1, m ).

    @ingroup magma_dblas3
    ********************************************************************/
extern "C" magma_int_t
magma_dtrsm_m(
    magma_int_t ngpu,
    magma_side_t side, magma_uplo_t uplo,
    magma_trans_t transa, magma_diag_t diag,
    magma_int_t m, magma_int_t n, double alpha,
    const double *A, magma_int_t lda,
    double       *B, magma_int_t ldb)
{
#define A(i, j) (A + (j)*nb*lda + (i)*nb)
#define B(i, j) (B + (j)*nb*ldb + (i)*nb)

#define dB(gpui, i, j) (dw[gpui] + (j)*nb*lddb + (i)*nb)

#define dA(gpui, i, j) (dw[gpui] + dimb*lddb + (i)*nb + (j)*nb*ldda)

    double  c_one     = MAGMA_D_ONE;
    double  c_neg_one = MAGMA_D_NEG_ONE;
    double  alpha_;
    double* dw[MagmaMaxGPUs];
    magma_queue_t queues[MagmaMaxGPUs][3];
    magma_int_t nrowa;
    magma_int_t nb = magma_get_dtrsm_m_nb();
    magma_int_t dev;
    magma_int_t info;
    magma_int_t k, j, kb, jb;
    magma_int_t ldda, dima, lddb, dimb;
    
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );

    bool lside = (side == MagmaLeft);
    if (lside) {
        nrowa = m;
    } else {
        nrowa = n;
    }
    bool upper   = (uplo == MagmaUpper);
    bool notrans = (transa == MagmaNoTrans);

    info = 0;
    if (! lside && side != MagmaRight) {
        info = 1;
    } else if (! upper && uplo != MagmaLower) {
        info = 2;
    } else if (! notrans && transa != MagmaTrans
               && transa != MagmaConjTrans) {
        info = 3;
    } else if (diag != MagmaUnit && diag != MagmaNonUnit) {
        info = 4;
    } else if (m < 0) {
        info = 5;
    } else if (n < 0) {
        info = 6;
    } else if (lda < max(1,nrowa)) {
        info = 9;
    } else if (ldb < max(1,m)) {
        info = 11;
    }

    // alpha = 0 case not done
    if (MAGMA_D_REAL(alpha) == 0. && MAGMA_D_IMAG(alpha) == 0.) {
        info = MAGMA_ERR_NOT_IMPLEMENTED;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    // Quick return if possible.
    if (n == 0) {
        return info;
    }

    magma_int_t nbl = magma_ceildiv( n, nb ); // number of blocks in a row
    magma_int_t mbl = magma_ceildiv( m, nb ); // number of blocks in a column

    if (lside) {
        lddb = m;
        dimb = magma_ceildiv( nbl, ngpu )*nb;
        if ( notrans ) {
            ldda = m;
            dima = 2 * nb;
        } else {
            ldda = 2 * nb;
            dima = m;
        }
    } else {
        lddb = magma_ceildiv( mbl, ngpu )*nb;
        dimb = n;
        if ( ! notrans ) {
            ldda = n;
            dima = 2 * nb;
        } else {
            ldda = 2 * nb;
            dima = n;
        }
    }

    for (dev = 0; dev < ngpu; ++dev) {
        magma_setdevice( dev );
        if (MAGMA_SUCCESS != magma_dmalloc( &dw[dev], (dimb*lddb + dima*ldda) )) {
            info = MAGMA_ERR_DEVICE_ALLOC;
            return info;
        }
        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queues[dev][0] );
        magma_queue_create( cdev, &queues[dev][1] );
        magma_queue_create( cdev, &queues[dev][2] );
    }

    if (lside) {
        if (notrans) {
            // Form  B := alpha*inv( A )*B
            if (upper) {
                // left upper notranspose
                magma_int_t nloc[MagmaMaxGPUs];
                for (dev = 0; dev < ngpu; ++dev)
                    nloc[dev] = 0;

                // copy B to mgpus
                for (k = 0; k < nbl; ++k) {
                    dev = k%ngpu;
                    magma_setdevice( dev );
                    kb = min( nb, n-k*nb );
                    nloc[dev] += kb;
                    magma_dsetmatrix_async( m, kb,
                                            B(0, k),            ldb,
                                            dB(dev, 0, k/ngpu), lddb, queues[dev][(mbl+1)%2] );
                }
                jb = min( nb, m-(mbl-1)*nb );
                for (dev = 0; dev < ngpu; ++dev) {
                    magma_setdevice( dev );
                    magma_dsetmatrix_async( m, jb,
                                            A(0, mbl-1),           lda,
                                            dA(dev, 0, (mbl-1)%2), ldda, queues[dev][(mbl+1)%2] );
                }
                for (j = mbl-1; j >= 0; --j) {
                    if (j > 0) {
                        jb = nb;
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dsetmatrix_async( j*nb, jb,
                                                    A(0, j-1),           lda,
                                                    dA(dev, 0, (j+1)%2), ldda, queues[dev][(j+1)%2] );
                        }
                    }
                    if (j == mbl-1)
                        alpha_ = alpha;
                    else
                        alpha_ = c_one;

                    jb = min( nb, m-j*nb );

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_setdevice( dev );
                        magma_dtrsm( side, uplo, transa, diag, jb, nloc[dev],
                                     alpha_, dA(dev, j, j%2), ldda,
                                             dB(dev, j, 0),   lddb, queues[dev][j%2] );
                    }

                    if (j > 0) {
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dgemm( transa, MagmaNoTrans, j*nb, nloc[dev], jb,
                                         c_neg_one, dA(dev, 0, j%2), ldda,
                                                    dB(dev, j, 0),   lddb,
                                         alpha_,    dB(dev, 0, 0),   lddb, queues[dev][j%2] );
                        }
                    }

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_queue_sync( queues[dev][j%2] );
                    }

                    for (k = 0; k < nbl; ++k) {
                        dev = k%ngpu;
                        magma_setdevice( dev );
                        kb = min( nb, n-k*nb );
                        magma_dgetmatrix_async( jb, kb,
                                                dB(dev, j, k/ngpu), lddb,
                                                B(j, k),            ldb, queues[dev][2] );
                    }
                }
            }
            else {
                // left lower notranspose
                magma_int_t nloc[MagmaMaxGPUs];
                for (dev = 0; dev < ngpu; ++dev)
                    nloc[dev] = 0;

                // copy B to mgpus
                for (k = 0; k < nbl; ++k) {
                    dev = k%ngpu;
                    magma_setdevice( dev );
                    kb = min( nb, n-k*nb );
                    nloc[dev] += kb;
                    magma_dsetmatrix_async( m, kb,
                                            B(0, k),            ldb,
                                            dB(dev, 0, k/ngpu), lddb, queues[dev][0] );
                }
                jb = min( nb, m );
                for (dev = 0; dev < ngpu; ++dev) {
                    magma_setdevice( dev );
                    magma_dsetmatrix_async( m, jb,
                                            A(0, 0),       lda,
                                            dA(dev, 0, 0), ldda, queues[dev][0] );
                }
                for (j = 0; j < mbl; ++j) {
                    if ((j+1)*nb < m) {
                        jb = min( nb, m-(j+1)*nb );
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dsetmatrix_async( (m-(j+1)*nb), jb,
                                                    A(j+1, j+1),           lda,
                                                    dA(dev, j+1, (j+1)%2), ldda, queues[dev][(j+1)%2] );
                        }
                    }
                    jb = min( nb, m-j*nb );

                    if (j == 0)
                        alpha_ = alpha;
                    else
                        alpha_ = c_one;

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_setdevice( dev );
                        magma_dtrsm( side, uplo, transa, diag, jb, nloc[dev],
                                     alpha_, dA(dev, j, j%2), ldda,
                                             dB(dev, j, 0),   lddb, queues[dev][j%2] );
                    }

                    if ( j < mbl-1 ) {
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dgemm( transa, MagmaNoTrans, m-(j+1)*nb, nloc[dev], nb,
                                         c_neg_one, dA(dev, j+1, j%2), ldda,
                                                    dB(dev, j, 0),     lddb,
                                         alpha_,    dB(dev, j+1, 0),   lddb, queues[dev][j%2] );
                        }
                    }

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_queue_sync( queues[dev][j%2] );
                    }

                    for (k = 0; k < nbl; ++k) {
                        dev = k%ngpu;
                        magma_setdevice( dev );
                        kb = min( nb, n-k*nb );
                        magma_dgetmatrix_async( jb, kb,
                                                dB(dev, j, k/ngpu), lddb,
                                                B(j, k),            ldb, queues[dev][2] );
                    }
                }
            }
        }
        else {
            // Form  B := alpha*inv( A**T )*B
            if (upper) {
                // left upper transpose or conj transpose
                magma_int_t nloc[MagmaMaxGPUs];
                for (dev = 0; dev < ngpu; ++dev)
                    nloc[dev] = 0;

                // copy B to mgpus
                for (k = 0; k < nbl; ++k) {
                    dev = k%ngpu;
                    magma_setdevice( dev );
                    kb = min( nb, n-k*nb );
                    nloc[dev] += kb;
                    magma_dsetmatrix_async( m, kb,
                                            B(0, k),            ldb,
                                            dB(dev, 0, k/ngpu), lddb, queues[dev][0] );
                }
                jb = min( nb, m );
                for (dev = 0; dev < ngpu; ++dev) {
                    magma_setdevice( dev );
                    magma_dsetmatrix_async( jb, m,
                                            A(0, 0),       lda,
                                            dA(dev, 0, 0), ldda, queues[dev][0] );
                }
                for (j = 0; j < mbl; ++j) {
                    if ((j+1)*nb < m) {
                        jb = min( nb, m-(j+1)*nb );
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dsetmatrix_async( jb, m-(j+1)*nb,
                                                    A(j+1, j+1),           lda,
                                                    dA(dev, (j+1)%2, j+1), ldda, queues[dev][(j+1)%2] );
                        }
                    }
                    jb = min( nb, m-j*nb );

                    if (j == 0)
                        alpha_ = alpha;
                    else
                        alpha_ = c_one;

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_setdevice( dev );
                        magma_dtrsm( side, uplo, transa, diag, jb, nloc[dev],
                                     alpha_, dA(dev, j%2, j), ldda,
                                             dB(dev, j, 0),   lddb, queues[dev][j%2] );
                    }

                    if ( j < mbl-1 ) {
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dgemm( transa, MagmaNoTrans, m-(j+1)*nb, nloc[dev], nb,
                                         c_neg_one, dA(dev, j%2, j+1), ldda,
                                                    dB(dev, j, 0),     lddb,
                                         alpha_,    dB(dev, j+1, 0),   lddb, queues[dev][j%2] );
                        }
                    }

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_queue_sync( queues[dev][j%2] );
                    }

                    for (k = 0; k < nbl; ++k) {
                        dev = k%ngpu;
                        magma_setdevice( dev );
                        kb = min( nb, n-k*nb );
                        magma_dgetmatrix_async( jb, kb,
                                                dB(dev, j, k/ngpu), lddb,
                                                B(j, k),            ldb, queues[dev][2] );
                    }
                }
            }
            else {
                // left lower transpose or conj transpose
                magma_int_t nloc[MagmaMaxGPUs];
                for (dev = 0; dev < ngpu; ++dev)
                    nloc[dev] = 0;

                // copy B to mgpus
                for (k = 0; k < nbl; ++k) {
                    dev = k%ngpu;
                    magma_setdevice( dev );
                    kb = min( nb, n-k*nb );
                    nloc[dev] += kb;
                    magma_dsetmatrix_async( m, kb,
                                            B(0, k),            ldb,
                                            dB(dev, 0, k/ngpu), lddb, queues[dev][(mbl+1)%2] );
                }
                jb = min( nb, m-(mbl-1)*nb );
                for (dev = 0; dev < ngpu; ++dev) {
                    magma_setdevice( dev );
                    magma_dsetmatrix_async( jb, m,
                                            A(mbl-1, 0),           lda,
                                            dA(dev, (mbl-1)%2, 0), ldda, queues[dev][(mbl+1)%2] );
                }
                for (j = mbl-1; j >= 0; --j) {
                    if (j > 0) {
                        jb = nb;
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dsetmatrix_async( jb, j*nb,
                                                    A(j-1, 0),           lda,
                                                    dA(dev, (j+1)%2, 0), ldda, queues[dev][(j+1)%2] );
                        }
                    }
                    if (j == mbl-1)
                        alpha_ = alpha;
                    else
                        alpha_ = c_one;

                    jb = min( nb, m-j*nb );

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_setdevice( dev );
                        magma_dtrsm( side, uplo, transa, diag, jb, nloc[dev],
                                     alpha_, dA(dev, j%2, j), ldda,
                                             dB(dev, j, 0),   lddb, queues[dev][j%2] );
                    }

                    if (j > 0) {
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dgemm( transa, MagmaNoTrans, j*nb, nloc[dev], jb,
                                         c_neg_one, dA(dev, j%2, 0), ldda,
                                                    dB(dev, j, 0),   lddb,
                                         alpha_,    dB(dev, 0, 0),   lddb, queues[dev][j%2] );
                        }
                    }

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_queue_sync( queues[dev][j%2] );
                    }

                    for (k = 0; k < nbl; ++k) {
                        dev = k%ngpu;
                        magma_setdevice( dev );
                        kb = min( nb, n-k*nb );
                        magma_dgetmatrix_async( jb, kb,
                                                dB(dev, j, k/ngpu), lddb,
                                                B(j, k),            ldb, queues[dev][2] );
                    }
                }
            }
        }
    }
    else {  // right
        if (notrans) {
            // Form  B := alpha*B*inv( A ).
            if (upper) {
                // right upper notranspose
                magma_int_t mloc[MagmaMaxGPUs];
                for (dev = 0; dev < ngpu; ++dev)
                    mloc[dev] = 0;

                // copy B to mgpus
                for (j = 0; j < mbl; ++j) {
                    dev = j%ngpu;
                    magma_setdevice( dev );
                    jb = min( nb, m-j*nb );
                    mloc[dev] += jb;
                    magma_dsetmatrix_async( jb, n,
                                            B(j, 0),            ldb,
                                            dB(dev, j/ngpu, 0), lddb, queues[dev][0] );
                }
                kb = min( nb, n );
                for (dev = 0; dev < ngpu; ++dev) {
                    magma_setdevice( dev );
                    magma_dsetmatrix_async( kb, n,
                                            A(0, 0),       lda,
                                            dA(dev, 0, 0), ldda, queues[dev][0] );
                }
                for (k = 0; k < nbl; ++k) {
                    if ((k+1)*nb < n) {
                        kb = min( nb, n-(k+1)*nb );
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dsetmatrix_async( kb, n-(k+1)*nb,
                                                    A(k+1, k+1),           lda,
                                                    dA(dev, (k+1)%2, k+1), ldda, queues[dev][(k+1)%2] );
                        }
                    }
                    kb = min( nb, n-k*nb );

                    if (k == 0)
                        alpha_ = alpha;
                    else
                        alpha_ = c_one;

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_setdevice( dev );
                        magma_dtrsm( side, uplo, transa, diag, mloc[dev], kb,
                                     alpha_, dA(dev, k%2, k), ldda,
                                             dB(dev, 0, k),   lddb, queues[dev][k%2] );
                    }

                    if ( k < nbl-1 ) {
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dgemm( MagmaNoTrans, transa, mloc[dev], n-(k+1)*nb, nb,
                                         c_neg_one, dB(dev, 0, k),     lddb,
                                                    dA(dev, k%2, k+1), ldda,
                                         alpha_,    dB(dev, 0, k+1),   lddb, queues[dev][k%2] );
                        }
                    }

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_queue_sync( queues[dev][k%2] );
                    }

                    for (j = 0; j < mbl; ++j) {
                        dev = j%ngpu;
                        magma_setdevice( dev );
                        jb = min( nb, m-j*nb );
                        magma_dgetmatrix_async( jb, kb,
                                                dB(dev, j/ngpu, k), lddb,
                                                B(j, k),            ldb, queues[dev][2] );
                    }
                }
            }
            else {
                // right lower notranspose
                magma_int_t mloc[MagmaMaxGPUs];
                for (dev = 0; dev < ngpu; ++dev)
                    mloc[dev] = 0;

                // copy B to mgpus
                for (j = 0; j < mbl; ++j) {
                    dev = j%ngpu;
                    magma_setdevice( dev );
                    jb = min( nb, m-j*nb );
                    mloc[dev] += jb;
                    magma_dsetmatrix_async( jb, n,
                                            B(j, 0),            ldb,
                                            dB(dev, j/ngpu, 0), lddb, queues[dev][(nbl+1)%2] );
                }
                kb = min( nb, n-(nbl-1)*nb );
                for (dev = 0; dev < ngpu; ++dev) {
                    magma_setdevice( dev );
                    magma_dsetmatrix_async( kb, n,
                                            A(nbl-1, 0),           lda,
                                            dA(dev, (nbl-1)%2, 0), ldda, queues[dev][(nbl+1)%2] );
                }
                for (k = nbl-1; k >= 0; --k) {
                    if (k > 0) {
                        kb = nb;
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dsetmatrix_async( kb, k*nb,
                                                    A(k-1, 0),           lda,
                                                    dA(dev, (k+1)%2, 0), ldda, queues[dev][(k+1)%2] );
                        }
                    }
                    if (k == nbl-1)
                        alpha_ = alpha;
                    else
                        alpha_ = c_one;

                    kb = min( nb, n-k*nb );

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_setdevice( dev );
                        magma_dtrsm( side, uplo, transa, diag, mloc[dev], kb,
                                     alpha_, dA(dev, k%2, k), ldda,
                                             dB(dev, 0, k),   lddb, queues[dev][k%2] );
                    }

                    if (k > 0) {
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dgemm( MagmaNoTrans, transa, mloc[dev], k*nb, kb,
                                         c_neg_one, dB(dev, 0, k),   lddb,
                                                    dA(dev, k%2, 0), ldda,
                                         alpha_,    dB(dev, 0, 0),   lddb, queues[dev][k%2] );
                        }
                    }

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_queue_sync( queues[dev][k%2] );
                    }

                    for (j = 0; j < mbl; ++j) {
                        dev = j%ngpu;
                        magma_setdevice( dev );
                        jb = min( nb, m-j*nb );
                        magma_dgetmatrix_async( jb, kb,
                                                dB(dev, j/ngpu, k), lddb,
                                                B(j, k),            ldb, queues[dev][2] );
                    }
                }
            }
        }
        else {
            // Form  B := alpha*B*inv( A**T ).
            if (upper) {
                // right upper transpose or conj transpose
                magma_int_t mloc[MagmaMaxGPUs];
                for (dev = 0; dev < ngpu; ++dev)
                    mloc[dev] = 0;

                // copy B to mgpus
                for (j = 0; j < mbl; ++j) {
                    dev = j%ngpu;
                    magma_setdevice( dev );
                    jb = min( nb, m-j*nb );
                    mloc[dev] += jb;
                    magma_dsetmatrix_async( jb, n,
                                            B(j, 0),            ldb,
                                            dB(dev, j/ngpu, 0), lddb, queues[dev][(nbl+1)%2] );
                }
                kb = min( nb, n-(nbl-1)*nb );
                for (dev = 0; dev < ngpu; ++dev) {
                    magma_setdevice( dev );
                    magma_dsetmatrix_async( n, kb,
                                            A(0, nbl-1),           lda,
                                            dA(dev, 0, (nbl-1)%2), ldda, queues[dev][(nbl+1)%2] );
                }
                for (k = nbl-1; k >= 0; --k) {
                    if (k > 0) {
                        kb = nb;
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dsetmatrix_async( k*nb, kb,
                                                    A(0, k-1),           lda,
                                                    dA(dev, 0, (k+1)%2), ldda, queues[dev][(k+1)%2] );
                        }
                    }
                    if (k == nbl-1)
                        alpha_ = alpha;
                    else
                        alpha_ = c_one;

                    kb = min( nb, n-k*nb );

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_setdevice( dev );
                        magma_dtrsm( side, uplo, transa, diag, mloc[dev], kb,
                                     alpha_, dA(dev, k, k%2), ldda,
                                             dB(dev, 0, k),   lddb, queues[dev][k%2] );
                    }

                    if (k > 0) {
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dgemm( MagmaNoTrans, transa, mloc[dev], k*nb, kb,
                                         c_neg_one, dB(dev, 0, k),   lddb,
                                                    dA(dev, 0, k%2), ldda,
                                         alpha_,    dB(dev, 0, 0),   lddb, queues[dev][k%2] );
                        }
                    }

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_queue_sync( queues[dev][k%2] );
                    }

                    for (j = 0; j < mbl; ++j) {
                        dev = j%ngpu;
                        magma_setdevice( dev );
                        jb = min( nb, m-j*nb );
                        magma_dgetmatrix_async( jb, kb,
                                                dB(dev, j/ngpu, k), lddb,
                                                B(j, k),            ldb, queues[dev][2] );
                    }
                }
            }
            else {
                // right lower transpose or conj transpose
                magma_int_t mloc[MagmaMaxGPUs];
                for (dev = 0; dev < ngpu; ++dev)
                    mloc[dev] = 0;

                // copy B to mgpus
                for (j = 0; j < mbl; ++j) {
                    dev = j%ngpu;
                    magma_setdevice( dev );
                    jb = min( nb, m-j*nb );
                    mloc[dev] += jb;
                    magma_dsetmatrix_async( jb, n,
                                            B(j, 0),            ldb,
                                            dB(dev, j/ngpu, 0), lddb, queues[dev][0] );
                }
                kb = min( nb, n );
                for (dev = 0; dev < ngpu; ++dev) {
                    magma_setdevice( dev );
                    magma_dsetmatrix_async( n, kb,
                                            A(0, 0),       lda,
                                            dA(dev, 0, 0), ldda, queues[dev][0] );
                }
                for (k = 0; k < nbl; ++k) {
                    if ((k+1)*nb < n) {
                        kb = min( nb, n-(k+1)*nb );
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dsetmatrix_async( (n-(k+1)*nb), kb,
                                                    A(k+1, k+1),           lda,
                                                    dA(dev, k+1, (k+1)%2), ldda, queues[dev][(k+1)%2] );
                        }
                    }
                    kb = min( nb, n-k*nb );

                    if (k == 0)
                        alpha_ = alpha;
                    else
                        alpha_ = c_one;

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_setdevice( dev );
                        magma_dtrsm( side, uplo, transa, diag, mloc[dev], kb,
                                     alpha_, dA(dev, k, k%2), ldda,
                                             dB(dev, 0, k),   lddb, queues[dev][k%2] );
                    }

                    if ( k < nbl-1 ) {
                        for (dev = 0; dev < ngpu; ++dev) {
                            magma_setdevice( dev );
                            magma_dgemm( MagmaNoTrans, transa, mloc[dev], n-(k+1)*nb, nb,
                                         c_neg_one, dB(dev, 0, k),     lddb,
                                                    dA(dev, k+1, k%2), ldda,
                                         alpha_,    dB(dev, 0, k+1),   lddb, queues[dev][k%2] );
                        }
                    }

                    for (dev = 0; dev < ngpu; ++dev) {
                        magma_queue_sync( queues[dev][k%2] );
                    }

                    for (j = 0; j < mbl; ++j) {
                        dev = j%ngpu;
                        magma_setdevice( dev );
                        jb = min( nb, m-j*nb );
                        magma_dgetmatrix_async( jb, kb,
                                                dB(dev, j/ngpu, k), lddb,
                                                B(j, k),            ldb, queues[dev][2] );
                    }
                }
            }
        }
    }

    for (dev = 0; dev < ngpu; ++dev) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev][2] );
        magma_queue_destroy( queues[dev][0] );
        magma_queue_destroy( queues[dev][1] );
        magma_queue_destroy( queues[dev][2] );
        magma_free( dw[dev] );
    }
    magma_setdevice( orig_dev );

    return info;
} /* magma_dtrsm_m */
