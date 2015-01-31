/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
    
       @author Raffaele Solca
    
       @generated from zhegvr.cpp normal z -> c, Fri Jan 30 19:00:18 2015

*/
#include "common_magma.h"

/**
    Purpose
    -------
    CHEGVR computes all the eigenvalues, and optionally, the eigenvectors
    of a complex generalized Hermitian-definite eigenproblem, of the form
    A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
    B are assumed to be Hermitian and B is also positive definite.

    Whenever possible, CHEEVR calls CSTEGR to compute the
    eigenspectrum using Relatively Robust Representations.  CSTEGR
    computes eigenvalues by the dqds algorithm, while orthogonal
    eigenvectors are computed from various "good" L D L^T representations
    (also known as Relatively Robust Representations). Gram-Schmidt
    orthogonalization is avoided as far as possible. More specifically,
    the various steps of the algorithm are as follows. For the i-th
    unreduced block of T,
       1.  Compute T - sigma_i = L_i D_i L_i^T, such that L_i D_i L_i^T
            is a relatively robust representation,
       2.  Compute the eigenvalues, lambda_j, of L_i D_i L_i^T to high
           relative accuracy by the dqds algorithm,
       3.  If there is a cluster of close eigenvalues, "choose" sigma_i
           close to the cluster, and go to step (a),
       4.  Given the approximate eigenvalue lambda_j of L_i D_i L_i^T,
           compute the corresponding eigenvector by forming a
           rank-revealing twisted factorization.
    The desired accuracy of the output can be specified by the input
    parameter ABSTOL.

    For more details, see "A new O(n^2) algorithm for the symmetric
    tridiagonal eigenvalue/eigenvector problem", by Inderjit Dhillon,
    Computer Science Division Technical Report No. UCB//CSD-97-971,
    UC Berkeley, May 1997.


    Note 1 : CHEEVR calls CSTEGR when the full spectrum is requested
    on machines which conform to the ieee-754 floating point standard.
    CHEEVR calls SSTEBZ and CSTEIN on non-ieee machines and
    when partial spectrum requests are made.

    Normal execution of CSTEGR may create NaNs and infinities and
    hence may abort due to a floating point exception in environments
    which do not handle NaNs and infinities in the ieee standard default
    manner.

    Arguments
    ---------
    @param[in]
    itype   INTEGER
            Specifies the problem type to be solved:
            = 1:  A*x = (lambda)*B*x
            = 2:  A*B*x = (lambda)*x
            = 3:  B*A*x = (lambda)*x

    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                   will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.

    @param[in]
    jobz    magma_vec_t
      -     = MagmaNoVec:  Compute eigenvalues only;
      -     = MagmaVec:    Compute eigenvalues and eigenvectors.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangles of A and B are stored;
      -     = MagmaLower:  Lower triangles of A and B are stored.

    @param[in]
    n       INTEGER
            The order of the matrices A and B.  N >= 0.

    @param[in,out]
    A       COMPLEX array, dimension (LDA, N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = MagmaLower,
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
    \n
            On exit, if JOBZ = MagmaVec, then if INFO = 0, A contains the
            matrix Z of eigenvectors.  The eigenvectors are normalized
            as follows:
            if ITYPE = 1 or 2, Z**H*B*Z = I;
            if ITYPE = 3, Z**H*inv(B)*Z = I.
            If JOBZ = MagmaNoVec, then on exit the upper triangle (if UPLO=MagmaUpper)
            or the lower triangle (if UPLO=MagmaLower) of A, including the
            diagonal, is destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[in,out]
    B       COMPLEX array, dimension (LDB, N)
            On entry, the Hermitian matrix B.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of B contains the
            upper triangular part of the matrix B.  If UPLO = MagmaLower,
            the leading N-by-N lower triangular part of B contains
            the lower triangular part of the matrix B.
    \n
            On exit, if INFO <= N, the part of B containing the matrix is
            overwritten by the triangular factor U or L from the Cholesky
            factorization B = U**H*U or B = L*L**H.

    @param[in]
    ldb     INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    @param[in]
    vl      REAL
    @param[in]
    vu      REAL
            If RANGE=MagmaRangeV, the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeI.

    @param[in]
    il      INTEGER
    @param[in]
    iu      INTEGER
            If RANGE=MagmaRangeI, the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeV.

    @param[in]
    abstol  REAL
            The absolute error tolerance for the eigenvalues.
            An approximate eigenvalue is accepted as converged
            when it is determined to lie in an interval [a,b]
            of width less than or equal to

                    ABSTOL + EPS * max( |a|,|b| ),
    \n
            where EPS is the machine precision.  If ABSTOL is less than
            or equal to zero, then  EPS*|T|  will be used in its place,
            where |T| is the 1-norm of the tridiagonal matrix obtained
            by reducing A to tridiagonal form.
    \n
            See "Computing Small Singular Values of Bidiagonal Matrices
            with Guaranteed High Relative Accuracy," by Demmel and
            Kahan, LAPACK Working Note #3.
    \n
            If high relative accuracy is important, set ABSTOL to
            SLAMCH( 'Safe minimum' ).  Doing so will guarantee that
            eigenvalues are computed to high relative accuracy when
            possible in future releases.  The current code does not
            make any guarantees about high relative accuracy, but
            furutre releases will. See J. Barlow and J. Demmel,
            "Computing Accurate Eigensystems of Scaled Diagonally
            Dominant Matrices", LAPACK Working Note #7, for a discussion
            of which matrices define their eigenvalues to high relative
            accuracy.

    @param[out]
    m       INTEGER
            The total number of eigenvalues found.  0 <= M <= N.
            If RANGE = MagmaRangeAll, M = N, and if RANGE = MagmaRangeI, M = IU-IL+1.

    @param[out]
    w       REAL array, dimension (N)
            If INFO = 0, the eigenvalues in ascending order.

    @param[out]
    Z       COMPLEX array, dimension (LDZ, max(1,M))
            If JOBZ = MagmaVec, then if INFO = 0, the first M columns of Z
            contain the orthonormal eigenvectors of the matrix A
            corresponding to the selected eigenvalues, with the i-th
            column of Z holding the eigenvector associated with W(i).
            If JOBZ = MagmaNoVec, then Z is not referenced.
            Note: the user must ensure that at least max(1,M) columns are
            supplied in the array Z; if RANGE = MagmaRangeV, the exact value of M
            is not known in advance and an upper bound must be used.

    @param[in]
    ldz     INTEGER
            The leading dimension of the array Z.  LDZ >= 1, and if
            JOBZ = MagmaVec, LDZ >= max(1,N).

    @param[out]
    isuppz  INTEGER ARRAY, dimension ( 2*max(1,M) )
            The support of the eigenvectors in Z, i.e., the indices
            indicating the nonzero elements in Z. The i-th eigenvector
            is nonzero only in elements ISUPPZ( 2*i-1 ) through
            ISUPPZ( 2*i ).
            __Implemented only for__ RANGE = MagmaRangeAll or MagmaRangeI and IU - IL = N - 1

    @param[out]
    work    (workspace) COMPLEX array, dimension (LWORK)
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK.  LWORK >= max(1,2*N).
            For optimal efficiency, LWORK >= (NB+1)*N,
            where NB is the max of the blocksize for CHETRD and for
            CUNMTR as returned by ILAENV.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    rwork   (workspace) REAL array, dimension (LRWORK)
            On exit, if INFO = 0, RWORK[0] returns the optimal
            (and minimal) LRWORK.

    @param[in]
    lrwork  INTEGER
            The length of the array RWORK.  LRWORK >= max(1,24*N).
    \n
            If LRWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the RWORK array, returns
            this value as the first entry of the RWORK array, and no error
            message related to LRWORK is issued by XERBLA.

    @param[out]
    iwork   (workspace) INTEGER array, dimension (LIWORK)
            On exit, if INFO = 0, IWORK[0] returns the optimal
            (and minimal) LIWORK.

    @param[in]
    liwork  INTEGER
            The dimension of the array IWORK.  LIWORK >= max(1,10*N).
    \n
            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal size of the IWORK array,
            returns this value as the first entry of the IWORK array, and
            no error message related to LIWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  Internal error

    Further Details
    ---------------
    Based on contributions by
       Inderjit Dhillon, IBM Almaden, USA
       Osni Marques, LBNL/NERSC, USA
       Ken Stanley, Computer Science Division, University of
         California at Berkeley, USA

    @ingroup magma_chegv_driver
    ********************************************************************/
extern "C" magma_int_t
magma_chegvr(
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *B, magma_int_t ldb,
    float vl, float vu, magma_int_t il, magma_int_t iu, float abstol,
    magma_int_t *m, float *w,
    magmaFloatComplex *Z, magma_int_t ldz,
    magma_int_t *isuppz,
    magmaFloatComplex *work, magma_int_t lwork,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info)
{
    magmaFloatComplex c_one = MAGMA_C_ONE;
    
    magmaFloatComplex *dA;
    magmaFloatComplex *dB;
    magmaFloatComplex *dZ;
    magma_int_t ldda = n;
    magma_int_t lddb = n;
    magma_int_t lddz = n;
    
    magma_int_t lower;
    magma_trans_t trans;
    magma_int_t wantz;
    magma_int_t lquery;
    magma_int_t alleig, valeig, indeig;
    
    magma_int_t lwmin, lrwmin, liwmin;
    
    magma_queue_t stream;
    magma_queue_create( &stream );
    
    wantz  = (jobz  == MagmaVec);
    lower  = (uplo  == MagmaLower);
    alleig = (range == MagmaRangeAll);
    valeig = (range == MagmaRangeV);
    indeig = (range == MagmaRangeI);
    lquery = (lwork == -1);
    
    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (! (alleig || valeig || indeig)) {
        *info = -2;
    } else if (! (wantz || (jobz == MagmaNoVec))) {
        *info = -3;
    } else if (! (lower || (uplo == MagmaUpper))) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < max(1,n)) {
        *info = -7;
    } else if (ldb < max(1,n)) {
        *info = -9;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -18;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -11;
            }
        } else if (indeig) {
            if (il < 1 || il > max(1,n)) {
                *info = -12;
            } else if (iu < min(n,il) || iu > n) {
                *info = -13;
            }
        }
    }
    
    magma_int_t nb = magma_get_chetrd_nb(n);
    
    lwmin =  n * (nb + 1);
    lrwmin = 24 * n;
    liwmin = 10 * n;
    
    work[0] = MAGMA_C_MAKE( lwmin, 0 );
    rwork[0] = lrwmin;
    iwork[0] = liwmin;
    
    if (lwork < lwmin && ! lquery) {
        *info = -21;
    } else if ((lrwork < lrwmin) && ! lquery) {
        *info = -23;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -25;
    }
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info));
        return *info;
    } else if (lquery) {
        return *info;
    }
    
    /* Quick return if possible */
    if (n == 0) {
        return *info;
    }
    
    if (MAGMA_SUCCESS != magma_cmalloc( &dA, n*ldda ) ||
        MAGMA_SUCCESS != magma_cmalloc( &dB, n*lddb ) ||
        MAGMA_SUCCESS != magma_cmalloc( &dZ, n*lddz )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    
    /* Form a Cholesky factorization of B. */
    magma_csetmatrix( n, n, B, ldb, dB, lddb );
    
    magma_csetmatrix_async( n, n,
                            A,  lda,
                            dA, ldda, stream );
    
    magma_cpotrf_gpu(uplo, n, dB, lddb, info);
    if (*info != 0) {
        *info = n + *info;
        return *info;
    }
    
    magma_queue_sync( stream );
    
    magma_cgetmatrix_async( n, n,
                            dB, lddb,
                            B,  ldb, stream );
    
    /* Transform problem to standard eigenvalue problem and solve. */
    magma_chegst_gpu(itype, uplo, n, dA, ldda, dB, lddb, info);
    
    magma_cheevr_gpu(jobz, range, uplo, n, dA, ldda, vl, vu, il, iu, abstol,
                     m, w, dZ, lddz, isuppz, A, lda, Z, ldz, work, lwork,
                     rwork, lrwork, iwork, liwork, info);
    
    if (wantz && *info == 0) {
        /* Backtransform eigenvectors to the original problem. */
    
        if (itype == 1 || itype == 2) {
            /* For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
               backtransform eigenvectors: x = inv(L)'*y or inv(U)*y */
            if (lower) {
                trans = MagmaConjTrans;
            } else {
                trans = MagmaNoTrans;
            }
            
            magma_ctrsm(MagmaLeft, uplo, trans, MagmaNonUnit, n, *m, c_one,
                          dB, lddb, dZ, lddz);
        }
        else if (itype == 3) {
            /* For B*A*x=(lambda)*x;
               backtransform eigenvectors: x = L*y or U'*y */
            if (lower) {
                trans = MagmaNoTrans;
            } else {
                trans = MagmaConjTrans;
            }
            
            magma_ctrmm(MagmaLeft, uplo, trans, MagmaNonUnit, n, *m, c_one,
                          dB, lddb, dZ, lddz);
        }
        
        magma_cgetmatrix( n, *m, dZ, lddz, Z, ldz );
        
    }
    
    magma_queue_sync( stream );
    
    magma_queue_destroy( stream );
    
    magma_free( dA );
    magma_free( dB );
    magma_free( dZ );
    
    return *info;
} /* magma_chegvr */
