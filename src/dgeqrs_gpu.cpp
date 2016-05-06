/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zgeqrs_gpu.cpp normal z -> d, Mon May  2 23:30:07 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    DGEQRS solves the least squares problem
           min || A*X - C ||
    using the QR factorization A = Q*R computed by DGEQRF_GPU.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. M >= N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of columns of the matrix C. NRHS >= 0.

    @param[in]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,n, as returned by
            DGEQRF_GPU in the first n columns of its array argument A.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A, LDDA >= M.

    @param[in]
    tau     DOUBLE PRECISION array, dimension (N)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by MAGMA_DGEQRF_GPU.

    @param[in,out]
    dB      DOUBLE PRECISION array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    @param[in,out]
    dT      DOUBLE PRECISION array that is the output (the 6th argument)
            of magma_dgeqrf_gpu of size
            2*MIN(M, N)*NB + ceil(N/32)*32 )* MAX(NB, NRHS).
            The array starts with a block of size MIN(M,N)*NB that stores
            the triangular T matrices used in the QR factorization,
            followed by MIN(M,N)*NB block storing the diagonal block
            inverses for the R matrix, followed by work space of size
            (ceil(N/32)*32)* MAX(NB, NRHS).

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB. LDDB >= M.

    @param[out]
    hwork   (workspace) DOUBLE PRECISION array, dimension (LWORK)
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK,
            LWORK >= (M - N + NB)*(NRHS + NB) + NRHS*NB,
            where NB is the blocksize given by magma_get_dgeqrf_nb( M, N ).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the WORK array.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_dgels_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dgeqrs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDouble_const_ptr dA,    magma_int_t ldda,
    double const *tau,
    magmaDouble_ptr dT,
    magmaDouble_ptr dB, magma_int_t lddb,
    double *hwork, magma_int_t lwork,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    #define dT(i_)    (dT + (lddwork + (i_))*nb)

    /* Constants */
    const double c_zero    = MAGMA_D_ZERO;
    const double c_one     = MAGMA_D_ONE;
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const magma_int_t ione = 1;
    
    /* Local variables */
    magmaDouble_ptr dwork;
    magma_int_t i, min_mn, lddwork, rows, ib;

    magma_int_t nb     = magma_get_dgeqrf_nb( m, n );
    magma_int_t lwkopt = (m - n + nb)*(nrhs + nb) + nrhs*nb;
    bool lquery = (lwork == -1);

    hwork[0] = magma_dmake_lwork( lwkopt );

    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0 || m < n)
        *info = -2;
    else if (nrhs < 0)
        *info = -3;
    else if (ldda < max(1,m))
        *info = -5;
    else if (lddb < max(1,m))
        *info = -9;
    else if (lwork < lwkopt && ! lquery)
        *info = -11;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    min_mn = min(m,n);
    if (min_mn == 0) {
        hwork[0] = c_one;
        return *info;
    }

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    
    /* B := Q^H * B */
    magma_dormqr_gpu( MagmaLeft, MagmaTrans,
                      m, nrhs, n,
                      dA(0,0), ldda, tau,
                      dB, lddb, hwork, lwork, dT, nb, info );
    if ( *info != 0 ) {
        magma_queue_destroy( queue );
        return *info;
    }

    /* Solve R*X = B(1:n,:) */
    lddwork= min_mn;
    if (nb < min_mn)
        dwork = dT+2*lddwork*nb;
    else
        dwork = dT;
    // To do: Why did we have this line originally; seems to be a bug (Stan)?
    // dwork = dT;

    i    = (min_mn - 1)/nb * nb;
    ib   = n-i;
    rows = m-i;

    // TODO: this assumes that, on exit from magma_dormqr_gpu, hwork contains
    // the last block of A and B (i.e., C in dormqr). This should be fixed.
    // Seems this data should already be on the GPU, so could switch to
    // magma_dtrsm and drop the dsetmatrix.
    if ( nrhs == 1 ) {
        blasf77_dtrsv( MagmaUpperStr, MagmaNoTransStr, MagmaNonUnitStr,
                       &ib, hwork,         &rows,
                            hwork+rows*ib, &ione);
    } else {
        blasf77_dtrsm( MagmaLeftStr, MagmaUpperStr, MagmaNoTransStr, MagmaNonUnitStr,
                       &ib, &nrhs,
                       &c_one, hwork,         &rows,
                               hwork+rows*ib, &rows);
    }
    
    // update the solution vector
    magma_dsetmatrix( ib, nrhs,
                      hwork+rows*ib, rows,
                      dwork+i,       lddwork, queue );

    // update c
    if (nrhs == 1) {
        magma_dgemv( MagmaNoTrans, i, ib,
                     c_neg_one, dA(0, i), ldda,
                                dwork + i,   1,
                     c_one,     dB,           1, queue );
    }
    else {
        magma_dgemm( MagmaNoTrans, MagmaNoTrans, i, nrhs, ib,
                     c_neg_one, dA(0, i),  ldda,
                                dwork + i, lddwork,
                     c_one,     dB,        lddb, queue );
    }

    magma_int_t start = i-nb;
    if (nb < min_mn) {
        for (i = start; i >= 0; i -= nb) {
            ib = min(min_mn - i, nb);
            rows = m - i;

            if (i + ib < n) {
                if (nrhs == 1) {
                    magma_dgemv( MagmaNoTrans, ib, ib,
                                 c_one,  dT(i), ib,
                                         dB+i,      1,
                                 c_zero, dwork+i,  1, queue );
                    magma_dgemv( MagmaNoTrans, i, ib,
                                 c_neg_one, dA(0, i), ldda,
                                            dwork + i,   1,
                                 c_one,     dB,           1, queue );
                }
                else {
                    magma_dgemm( MagmaNoTrans, MagmaNoTrans, ib, nrhs, ib,
                                 c_one,  dT(i),   ib,
                                         dB+i,    lddb,
                                 c_zero, dwork+i, lddwork, queue );
                    magma_dgemm( MagmaNoTrans, MagmaNoTrans, i, nrhs, ib,
                                 c_neg_one, dA(0, i),  ldda,
                                            dwork + i, lddwork,
                                 c_one,     dB,        lddb, queue );
                }
            }
        }
    }

    magma_dcopymatrix( n, nrhs,
                       dwork, lddwork,
                       dB,    lddb, queue );
    
    magma_queue_destroy( queue );
    return *info;
}
