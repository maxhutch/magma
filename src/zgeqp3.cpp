/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
  
       @precisions normal z -> c d s

       @author Mitch Horton
*/

#include "common_magma.h"

#define PRECISION_z
#define COMPLEX

/**
    Purpose
    -------
    ZGEQP3 computes a QR factorization with column pivoting of a
    matrix A:  A*P = Q*R  using Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the upper triangle of the array contains the
            min(M,N)-by-N upper trapezoidal matrix R; the elements below
            the diagonal, together with the array TAU, represent the
            unitary matrix Q as a product of min(M,N) elementary
            reflectors.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    @param[in,out]
    jpvt    INTEGER array, dimension (N)
            On entry, if JPVT(J).ne.0, the J-th column of A is permuted
            to the front of A*P (a leading column); if JPVT(J)=0,
            the J-th column of A is a free column.
            On exit, if JPVT(J)=K, then the J-th column of A*P was the
            the K-th column of A.

    @param[out]
    tau     COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors.

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO=0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            For [sd]geqp3, LWORK >= (N+1)*NB + 2*N;
            for [cz]geqp3, LWORK >= (N+1)*NB,
            where NB is the optimal blocksize.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param
    rwork   (workspace, for [cz]geqp3 only) DOUBLE PRECISION array, dimension (2*N)

    @param[out]
    info    INTEGER
      -     = 0: successful exit.
      -     < 0: if INFO = -i, the i-th argument had an illegal value.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

      Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

      H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector
    with v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in
    A(i+1:m,i), and tau in TAU(i).

    @ingroup magma_zgeqp3_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zgeqp3(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info )
{
#define  A(i, j) (A     + (i) + (j)*(lda ))
#define dA(i, j) (dwork + (i) + (j)*(ldda))

    magmaDoubleComplex   *dwork, *df;

    magma_int_t ione = 1;

    magma_int_t n_j, ldda, ldwork;
    magma_int_t j, jb, na, nb, sm, sn, fjb, nfxd, minmn;
    magma_int_t topbmn, sminmn, lwkopt=0, lquery;
    
    *info = 0;
    lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    }
    
    nb = magma_get_zgeqp3_nb(min(m, n));
    minmn = min(m,n);
    if (*info == 0) {
        if (minmn == 0) {
            lwkopt = 1;
        } else {
            lwkopt = (n + 1)*nb;
            #ifdef REAL
            lwkopt += 2*n;
            #endif
        }
        work[0] = MAGMA_Z_MAKE( lwkopt, 0. );

        if (lwork < lwkopt && ! lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    } else if (lquery) {
        return *info;
    }

    if (minmn == 0)
        return *info;

    #ifdef REAL
    double *rwork = work + (n + 1)*nb;
    #endif

    ldda = ((m+31)/32)*32;
    ldwork = n*ldda + (n+1)*nb;
    if (MAGMA_SUCCESS != magma_zmalloc( &dwork, ldwork )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    df = dwork + n*ldda;
    // dwork used for dA

    magma_queue_t stream;
    magma_queue_create( &stream );

    /* Move initial columns up front.
     * Note jpvt uses 1-based indices for historical compatibility. */
    nfxd = 0;
    for (j = 0; j < n; ++j) {
        if (jpvt[j] != 0) {
            if (j != nfxd) {
                blasf77_zswap(&m, A(0, j), &ione, A(0, nfxd), &ione);
                jpvt[j]    = jpvt[nfxd];
                jpvt[nfxd] = j + 1;
            }
            else {
                jpvt[j] = j + 1;
            }
            ++nfxd;
        }
        else {
            jpvt[j] = j + 1;
        }
    }

    /*     Factorize fixed columns
           =======================
           Compute the QR factorization of fixed columns and update
           remaining columns. */
    if (nfxd > 0) {
        na = min(m,nfxd);
        lapackf77_zgeqrf(&m, &na, A, &lda, tau, work, &lwork, info);
        if (na < n) {
            n_j = n - na;
            lapackf77_zunmqr( MagmaLeftStr, MagmaConjTransStr, &m, &n_j, &na,
                              A, &lda, tau, A(0, na), &lda,
                              work, &lwork, info );
        }
    }
    
    /*  Factorize free columns */
    if (nfxd < minmn) {
        sm = m - nfxd;
        sn = n - nfxd;
        sminmn = minmn - nfxd;
        
        if (nb < sminmn) {
            j = nfxd;
            
            // Set the original matrix to the GPU
            magma_zsetmatrix_async( m, sn,
                                    A (0,j), lda,
                                    dA(0,j), ldda, stream );
        }

        /* Initialize partial column norms. */
        for (j = nfxd; j < n; ++j) {
            rwork[j] = magma_cblas_dznrm2( sm, A(nfxd,j), ione );
            rwork[n + j] = rwork[j];
        }
        
        j = nfxd;
        if (nb < sminmn) {
            /* Use blocked code initially. */
            magma_queue_sync( stream );
            
            /* Compute factorization: while loop. */
            topbmn = minmn - nb;
            while(j < topbmn) {
                jb = min(nb, topbmn - j);
                
                /* Factorize JB columns among columns J:N. */
                n_j = n - j;
                
                if (j > nfxd) {
                    // Get panel to the CPU
                    magma_zgetmatrix( m-j, jb,
                                      dA(j,j), ldda,
                                      A (j,j), lda );
                    
                    // Get the rows
                    magma_zgetmatrix( jb, n_j - jb,
                                      dA(j,j + jb), ldda,
                                      A (j,j + jb), lda );
                }

                magma_zlaqps( m, n_j, j, jb, &fjb,
                              A (0, j), lda,
                              dA(0, j), ldda,
                              &jpvt[j], &tau[j], &rwork[j], &rwork[n + j],
                              work,
                              &work[jb], n_j,
                              &df[jb],   n_j );
                
                j += fjb;  /* fjb is actual number of columns factored */
            }
        }
        
        /* Use unblocked code to factor the last or only block. */
        if (j < minmn) {
            n_j = n - j;
            if (j > nfxd) {
                magma_zgetmatrix( m-j, n_j,
                                  dA(j,j), ldda,
                                  A (j,j), lda );
            }
            lapackf77_zlaqp2(&m, &n_j, &j, A(0, j), &lda, &jpvt[j],
                             &tau[j], &rwork[j], &rwork[n+j], work );
        }
    }

    work[0] = MAGMA_Z_MAKE( lwkopt, 0. );
    magma_free( dwork );

    magma_queue_destroy( stream );

    return *info;
} /* magma_zgeqp3 */
