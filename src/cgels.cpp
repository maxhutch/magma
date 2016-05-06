/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zgels.cpp normal z -> c, Mon May  2 23:30:09 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    CGELS solves the overdetermined, least squares problem
           min || A*X - C ||
    using the QR factorization A.
    The underdetermined problem (m < n) is not currently handled.


    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:   the linear system involves A.
            Only TRANS=MagmaNoTrans is currently handled.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. M >= N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of columns of the matrix C. NRHS >= 0.

    @param[in,out]
    A       COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, A is overwritten by details of its QR
            factorization as returned by CGEQRF.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A, LDA >= M.

    @param[in,out]
    B       COMPLEX array, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    @param[in]
    ldb     INTEGER
            The leading dimension of the array B. LDB >= M.

    @param[out]
    hwork   (workspace) COMPLEX array, dimension MAX(1,LWORK).
            On exit, if INFO = 0, HWORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array HWORK,
            LWORK >= max( N*NB, 2*NB*NB ),
            where NB is the blocksize given by magma_get_cgeqrf_nb( M, N ).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the HWORK array.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_cgels_driver
    ********************************************************************/
extern "C" magma_int_t
magma_cgels(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr A, magma_int_t lda,
    magmaFloatComplex_ptr B, magma_int_t ldb,
    magmaFloatComplex *hwork, magma_int_t lwork,
    magma_int_t *info)
{
    /* Constants */
    const magmaFloatComplex c_one = MAGMA_C_ONE;
    
    /* Local variables */
    magmaFloatComplex *tau;
    magma_int_t min_mn;
    magma_int_t nb     = magma_get_cgeqrf_nb( m, n );
    magma_int_t lwkopt = max( n*nb, 2*nb*nb ); // (m - n + nb)*(nrhs + nb) + nrhs*nb;
    bool lquery = (lwork == -1);

    hwork[0] = magma_cmake_lwork( lwkopt );

    *info = 0;
    /* For now, N is the only case working */
    if ( trans != MagmaNoTrans )
        *info = -1;
    else if (m < 0)
        *info = -2;
    else if (n < 0 || m < n) /* LQ is not handle for now */
        *info = -3;
    else if (nrhs < 0)
        *info = -4;
    else if (lda < max(1,m))
        *info = -6;
    else if (ldb < max(1,m))
        *info = -8;
    else if (lwork < lwkopt && ! lquery)
        *info = -10;

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

    magma_cmalloc_cpu( &tau, min_mn );
    if ( tau == NULL ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    magma_cgeqrf( m, n, A, lda, tau, hwork, lwork, info );

    if ( *info == 0 ) {
        // B := Q' * B
        lapackf77_cunmqr( MagmaLeftStr, Magma_ConjTransStr, &m, &nrhs, &n, 
                          A, &lda, tau, B, &ldb, hwork, &lwork, info );
 
        // Solve R*X = B(1:n,:)
        blasf77_ctrsm( MagmaLeftStr, MagmaUpperStr, MagmaNoTransStr, MagmaNonUnitStr, 
                       &n, &nrhs, &c_one, A, &lda, B, &ldb );
    }
    
    magma_free_cpu( tau );
    return *info;
}
