/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_dgeqrs_gpu(magma_int_t m, magma_int_t n, magma_int_t nrhs,
                 double *dA,    magma_int_t ldda,
                 double *tau,   double *dT,
                 double *dB,    magma_int_t lddb,
                 double *hwork, magma_int_t lwork,
                 magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    Solves the least squares problem
           min || A*X - C ||
    using the QR factorization A = Q*R computed by DGEQRF_GPU.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A. M >= N >= 0.

    NRHS    (input) INTEGER
            The number of columns of the matrix C. NRHS >= 0.

    A       (input) DOUBLE_PRECISION array on the GPU, dimension (LDDA,N)
            The i-th column must contain the vector which defines the
            elementary reflector H(i), for i = 1,2,...,n, as returned by
            DGEQRF_GPU in the first n columns of its array argument A.

    LDDA    (input) INTEGER
            The leading dimension of the array A, LDDA >= M.

    TAU     (input) DOUBLE_PRECISION array, dimension (N)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by MAGMA_DGEQRF_GPU.

    DB      (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    DT      (input) DOUBLE_PRECISION array that is the output (the 6th argument)
            of magma_dgeqrf_gpu of size
            2*MIN(M, N)*NB + ((N+31)/32*32 )* MAX(NB, NRHS).
            The array starts with a block of size MIN(M,N)*NB that stores
            the triangular T matrices used in the QR factorization,
            followed by MIN(M,N)*NB block storing the diagonal block
            inverses for the R matrix, followed by work space of size
            ((N+31)/32*32 )* MAX(NB, NRHS).

    LDDB    (input) INTEGER
            The leading dimension of the array DB. LDDB >= M.

    HWORK   (workspace/output) DOUBLE_PRECISION array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK,
            LWORK >= (M - N + NB)*(NRHS + NB) + NRHS*NB,
            where NB is the blocksize given by magma_get_dgeqrf_nb( M ).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the WORK array.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================    */

    #define a_ref(a_1,a_2) (dA+(a_2)*(ldda) + (a_1))
    #define d_ref(a_1)     (dT+(lddwork+(a_1))*nb)

    double c_zero    = MAGMA_D_ZERO;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double *dwork;
    magma_int_t i, k, lddwork, rows, ib;
    magma_int_t ione = 1;

    magma_int_t nb     = magma_get_dgeqrf_nb(m);
    magma_int_t lwkopt = (m - n + nb)*(nrhs + nb) + nrhs*nb;
    int lquery = (lwork == -1);

    hwork[0] = MAGMA_D_MAKE( (double)lwkopt, 0. );

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

    k = min(m,n);
    if (k == 0) {
        hwork[0] = c_one;
        return *info;
    }

    /* B := Q' * B */
    magma_dormqr_gpu( MagmaLeft, MagmaTrans,
                      m, nrhs, n,
                      a_ref(0,0), ldda, tau,
                      dB, lddb, hwork, lwork, dT, nb, info );
    if ( *info != 0 ) {
        return *info;
    }

    /* Solve R*X = B(1:n,:) */
    lddwork= k;
    if (nb < k)
        dwork = dT+2*lddwork*nb;
    else
        dwork = dT;
    // To do: Why did we have this line originally; seems to be a bug (Stan)?
    // dwork = dT;

    i    = (k-1)/nb * nb;
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
    magma_dsetmatrix( ib, nrhs, hwork+rows*ib, rows, dwork+i, lddwork );

    // update c
    if (nrhs == 1)
        magma_dgemv( MagmaNoTrans, i, ib,
                     c_neg_one, a_ref(0, i), ldda,
                                dwork + i,   1,
                     c_one,     dB,           1);
    else
        magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                     i, nrhs, ib,
                     c_neg_one, a_ref(0, i), ldda,
                                dwork + i,   lddwork,
                     c_one,     dB,           lddb);

    int start = i-nb;
    if (nb < k) {
        for (i = start; i >=0; i -= nb) {
            ib = min(k-i, nb);
            rows = m -i;

            if (i + ib < n) {
                if (nrhs == 1) {
                    magma_dgemv( MagmaNoTrans, ib, ib,
                                 c_one,  d_ref(i), ib,
                                         dB+i,      1,
                                 c_zero, dwork+i,  1);
                    magma_dgemv( MagmaNoTrans, i, ib,
                                 c_neg_one, a_ref(0, i), ldda,
                                            dwork + i,   1,
                                 c_one,     dB,           1);
                }
                else {
                    magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                                 ib, nrhs, ib,
                                 c_one,  d_ref(i), ib,
                                         dB+i,      lddb,
                                 c_zero, dwork+i,  lddwork);
                    magma_dgemm( MagmaNoTrans, MagmaNoTrans,
                                 i, nrhs, ib,
                                 c_neg_one, a_ref(0, i), ldda,
                                            dwork + i,   lddwork,
                                 c_one,     dB,          lddb);
                }
            }
        }
    }

    magma_dcopymatrix( (n), nrhs,
                       dwork, lddwork,
                       dB,    lddb );
    
    return *info;
}

#undef a_ref
#undef d_ref
