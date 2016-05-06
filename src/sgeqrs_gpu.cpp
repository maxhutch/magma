/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zgeqrs_gpu.cpp normal z -> s, Mon May  2 23:30:07 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    SGEQRS solves the least squares problem
           min || A*X - C ||
    using the QR factorization A = Q*R computed by SGEQRF_GPU.

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
    dA      REAL array on the GPU, dimension (LDDA,N)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,n, as returned by
            SGEQRF_GPU in the first n columns of its array argument A.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A, LDDA >= M.

    @param[in]
    tau     REAL array, dimension (N)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by MAGMA_SGEQRF_GPU.

    @param[in,out]
    dB      REAL array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    @param[in,out]
    dT      REAL array that is the output (the 6th argument)
            of magma_sgeqrf_gpu of size
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
    hwork   (workspace) REAL array, dimension (LWORK)
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK,
            LWORK >= (M - N + NB)*(NRHS + NB) + NRHS*NB,
            where NB is the blocksize given by magma_get_sgeqrf_nb( M, N ).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the WORK array.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_sgels_comp
    ********************************************************************/
extern "C" magma_int_t
magma_sgeqrs_gpu(
    magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloat_const_ptr dA,    magma_int_t ldda,
    float const *tau,
    magmaFloat_ptr dT,
    magmaFloat_ptr dB, magma_int_t lddb,
    float *hwork, magma_int_t lwork,
    magma_int_t *info)
{
    #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
    #define dT(i_)    (dT + (lddwork + (i_))*nb)

    /* Constants */
    const float c_zero    = MAGMA_S_ZERO;
    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const magma_int_t ione = 1;
    
    /* Local variables */
    magmaFloat_ptr dwork;
    magma_int_t i, min_mn, lddwork, rows, ib;

    magma_int_t nb     = magma_get_sgeqrf_nb( m, n );
    magma_int_t lwkopt = (m - n + nb)*(nrhs + nb) + nrhs*nb;
    bool lquery = (lwork == -1);

    hwork[0] = magma_smake_lwork( lwkopt );

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
    magma_sormqr_gpu( MagmaLeft, MagmaTrans,
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

    // TODO: this assumes that, on exit from magma_sormqr_gpu, hwork contains
    // the last block of A and B (i.e., C in sormqr). This should be fixed.
    // Seems this data should already be on the GPU, so could switch to
    // magma_strsm and drop the ssetmatrix.
    if ( nrhs == 1 ) {
        blasf77_strsv( MagmaUpperStr, MagmaNoTransStr, MagmaNonUnitStr,
                       &ib, hwork,         &rows,
                            hwork+rows*ib, &ione);
    } else {
        blasf77_strsm( MagmaLeftStr, MagmaUpperStr, MagmaNoTransStr, MagmaNonUnitStr,
                       &ib, &nrhs,
                       &c_one, hwork,         &rows,
                               hwork+rows*ib, &rows);
    }
    
    // update the solution vector
    magma_ssetmatrix( ib, nrhs,
                      hwork+rows*ib, rows,
                      dwork+i,       lddwork, queue );

    // update c
    if (nrhs == 1) {
        magma_sgemv( MagmaNoTrans, i, ib,
                     c_neg_one, dA(0, i), ldda,
                                dwork + i,   1,
                     c_one,     dB,           1, queue );
    }
    else {
        magma_sgemm( MagmaNoTrans, MagmaNoTrans, i, nrhs, ib,
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
                    magma_sgemv( MagmaNoTrans, ib, ib,
                                 c_one,  dT(i), ib,
                                         dB+i,      1,
                                 c_zero, dwork+i,  1, queue );
                    magma_sgemv( MagmaNoTrans, i, ib,
                                 c_neg_one, dA(0, i), ldda,
                                            dwork + i,   1,
                                 c_one,     dB,           1, queue );
                }
                else {
                    magma_sgemm( MagmaNoTrans, MagmaNoTrans, ib, nrhs, ib,
                                 c_one,  dT(i),   ib,
                                         dB+i,    lddb,
                                 c_zero, dwork+i, lddwork, queue );
                    magma_sgemm( MagmaNoTrans, MagmaNoTrans, i, nrhs, ib,
                                 c_neg_one, dA(0, i),  ldda,
                                            dwork + i, lddwork,
                                 c_one,     dB,        lddb, queue );
                }
            }
        }
    }

    magma_scopymatrix( n, nrhs,
                       dwork, lddwork,
                       dB,    lddb, queue );
    
    magma_queue_destroy( queue );
    return *info;
}
