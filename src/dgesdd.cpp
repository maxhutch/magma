/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
       @precisions normal d -> s

*/
#include "common_magma.h"

#define PRECISION_d
#define REAL

// Version 1 - LAPACK
// Version 2 - MAGMA
#define VERSION 2

/**
    Purpose
    -------
    DGESDD computes the singular value decomposition (SVD) of a real
    M-by-N matrix A, optionally computing the left and right singular
    vectors, by using divide-and-conquer method. The SVD is written

        A = U * SIGMA * transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note that the routine returns VT = V**T, not V.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    ---------
    @param[in]
    jobz    magma_vec_t
            Specifies options for computing all or part of the matrix U:
      -     = MagmaAllVec:  all M columns of U and all N rows of V**T are
                            returned in the arrays U and VT;
      -     = MagmaSomeVec: the first min(M,N) columns of U and
                            the first min(M,N) rows of V**T are
                            returned in the arrays U and VT;
      -     = MagmaOverwriteVec:
                    If M >= N, the first N columns of U are overwritten
                    on the array A and all rows of V**T are returned in
                    the array VT;
                    otherwise, all columns of U are returned in the
                    array U and the first M rows of V**T are overwritten
                    on the array A;
      -     = MagmaNoVec:   no columns of U or rows of V**T are computed.

    @param[in]
    m       INTEGER
            The number of rows of the input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the input matrix A.  N >= 0.

    @param[in,out]
    A       DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
      -     if JOBZ = MagmaOverwriteVec,
                if M >= N, A is overwritten with the first N columns
                of U (the left singular vectors, stored columnwise);
                otherwise, A is overwritten with the first M rows
                of V**T (the right singular vectors, stored owwise).
      -     if JOBZ != MagmaOverwriteVec, the contents of A are destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    s       DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i + 1).

    @param[out]
    U       DOUBLE PRECISION array, dimension (LDU,UCOL)
            UCOL = M if JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N;
            UCOL = min(M,N) if JOBZ = MagmaSomeVec.
      -     If JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N,
            U contains the M-by-M orthogonal matrix U;
      -     if JOBZ = MagmaSomeVec, U contains the first min(M,N) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBZ = MagmaOverwriteVec and M >= N, or JOBZ = MagmaNoVec, U is not referenced.

    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBZ = MagmaSomeVec or MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N, LDU >= M.

    @param[out]
    VT      DOUBLE PRECISION array, dimension (LDVT,N)
      -     If JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M >= N,
            VT contains the N-by-N orthogonal matrix V**T;
      -     if JOBZ = MagmaSomeVec, VT contains the first min(M,N) rows of
            V**T (the right singular vectors, stored rowwise);
      -     if JOBZ = MagmaOverwriteVec and M < N, or JOBZ = MagmaNoVec, VT is not referenced.

    @param[in]
    ldvt    INTEGER
            The leading dimension of the array VT.  LDVT >= 1; if
            JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M >= N, LDVT >= N;
            if JOBZ = MagmaSomeVec, LDVT >= min(M,N).

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            Let x = max(M,N) and y = min(M,N). The optimal block size
            nb can be obtained through magma_get_dgesvd_nb(N).
            The threshold for x >> y currently is x >= y*11/6.
            *Required size different than in LAPACK.* In most cases, these
            sizes should give optimal performance for both MAGMA and LAPACK.
      -     If JOBZ = MagmaNoVec,
                if x >> y, LWORK >=       3*y + max( (2*y)*nb, 7*y );
                otherwise, LWORK >=       3*y + max( (x+y)*nb, 7*y ).
      -     If JOBZ = MagmaOverwriteVec,
                if x >> y, LWORK >= y*y + 3*y + max( (2*y)*nb, 4*y*y + 4*y ),
                   prefer  LWORK >= y*y + 3*y + max( (2*y)*nb, 4*y*y + 4*y, y*y + y*nb );
                otherwise, LWORK >=       3*y + max( (x+y)*nb, 4*y*y + 4*y ).
      -     If JOBZ = MagmaSomeVec,
                if x >> y, LWORK >= y*y + 3*y + max( (2*y)*nb, 3*y*y + 4*y );
                otherwise, LWORK >=       3*y + max( (x+y)*nb, 3*y*y + 4*y ).
      -     If JOBZ = MagmaAllVec,
                if x >> y, LWORK >= y*y + max( 3*y + max( (2*y)*nb, 3*y*y + 4*y ), y + x    ),
                   prefer  LWORK >= y*y + max( 3*y + max( (2*y)*nb, 3*y*y + 4*y ), y + x*nb );
                otherwise, LWORK >=            3*y + max( (x+y)*nb, 3*y*y + 4*y ).
    \n
            If LWORK = -1 but other input arguments are legal, WORK[0]
            returns the optimal LWORK.

    @param
    iwork   (workspace) INTEGER array, dimension (8*min(M,N))

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  DBDSDC did not converge, updating process failed.

    Further Details
    ---------------
    Based on contributions by
    Ming Gu and Huan Ren, Computer Science Division, University of
    California at Berkeley, USA

    @ingroup magma_dgesvd_driver
    ********************************************************************/
extern "C" magma_int_t
magma_dgesdd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    double *s,
    double *U, magma_int_t ldu,
    double *VT, magma_int_t ldvt,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *iwork,
    magma_int_t *info)
{
    #define A(i_,j_)  (A  + (i_) + (j_)*lda)
    #define U(i_,j_)  (U  + (i_) + (j_)*ldu)
    #define VT(i_,j_) (VT + (i_) + (j_)*ldvt)

    /* Constants */
    const double c_zero = MAGMA_D_ZERO;
    const double c_one  = MAGMA_D_ONE;
    const magma_int_t izero = 0;
    const magma_int_t ione  = 1;

    /* Local variables */
    magma_int_t lnwork, i__1;
    magma_int_t i, ie, il=0, ir=0, iu, blk, nb;
    double dum[1], eps;
    magma_int_t ivt, iscl;
    double anrm;
    magma_int_t idum[1], ierr, itau;
    magma_int_t chunk=0, wrkbl, itaup, itauq, mnthr=0;
    magma_int_t nwork;
    magma_int_t bdspac=0;
    double bignum;
    magma_int_t ldwrkl, ldwrkr, minwrk, ldwrku, maxwrk, ldwkvt;
    double smlnum;
    
    /* Parameter adjustments */
    A  -= 1 + lda;
    --work;

    /* Function Body */
    *info = 0;
    const magma_int_t m_1 = m - 1;
    const magma_int_t n_1 = n - 1;
    const magma_int_t minmn   = min(m,n);
    const magma_int_t wantqa  = (jobz == MagmaAllVec);
    const magma_int_t wantqs  = (jobz == MagmaSomeVec);
    const magma_int_t wantqas = (wantqa || wantqs);
    const magma_int_t wantqo  = (jobz == MagmaOverwriteVec);
    const magma_int_t wantqn  = (jobz == MagmaNoVec);
    const magma_int_t lquery  = (lwork == -1);
    minwrk = 1;
    maxwrk = 1;

    /* Test the input arguments */
    if (! (wantqa || wantqs || wantqo || wantqn)) {
        *info = -1;
    }
    else if (m < 0) {
        *info = -2;
    }
    else if (n < 0) {
        *info = -3;
    }
    else if (lda < max(1,m)) {
        *info = -5;
    }
    else if (ldu < 1 || (wantqas && ldu < m) || (wantqo && m < n && ldu < m)) {
        *info = -8;
    }
    else if (ldvt < 1 || (wantqa && ldvt < n) || (wantqs && ldvt < minmn)
                      || (wantqo && m >= n && ldvt < n)) {
        *info = -10;
    }
    
    nb = magma_get_dgesvd_nb(n);

    /* Compute workspace */
    /* (Note: Comments in the code beginning "Workspace:" describe the */
    /* minimal amount of workspace needed at that point in the code, */
    /* as well as the preferred amount for good performance. */
    /* NB refers to the optimal block size for the immediately */
    /* following subroutine, as returned by ILAENV.) */
    /* We assume MAGMA's nb >= LAPACK's nb for all routines, */
    /* because calling Fortran's ILAENV is not portable. */
    if (*info == 0) {
        if (m >= n && minmn > 0) {
            /* Compute space needed for DBDSDC */
            mnthr = (magma_int_t) (minmn*11. / 6.);
            if (wantqn) {
                bdspac = 7*n;          // dbdsdc claims only 4*n
            }
            else {
                bdspac = 3*n*n + 4*n;  // consistent with dbdsdc
            }
            if (m >= mnthr) {
                if (wantqn) {
                    /* Path 1 (M much larger than N, JOBZ='N') */
                    wrkbl  =              n +   n * nb;   //magma_ilaenv( 1, "DGEQRF", " ",   m, n, -1, -1 )
                    wrkbl  = max(wrkbl, 3*n + 2*n * nb ); // dgebrd
                    maxwrk = max(wrkbl,   n + bdspac);
                    minwrk = maxwrk;  // lapack was: bdspac + n
                }
                else if (wantqo) {
                    /* Path 2 (M much larger than N, JOBZ='O') */
                    wrkbl  =              n +   n * nb;   //magma_ilaenv( 1, "DGEQRF", " ",   m, n, -1, -1 )
                    wrkbl  = max(wrkbl,   n +   n * nb ); //magma_ilaenv( 1, "DORGQR", " ",   m, n,  n, -1 ))
                    wrkbl  = max(wrkbl, 3*n + 2*n * nb ); // dgebrd
                    wrkbl  = max(wrkbl, 3*n +   n * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", n, n,  n, -1 ))
                    wrkbl  = max(wrkbl, 3*n +   n * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", n, n,  n, -1 ))
                    wrkbl  = max(wrkbl, 3*n + n*n + bdspac);
                    maxwrk = wrkbl + n*n;
                    minwrk = maxwrk;  // lapack was: bdspac + 2*n*n + 3*n
                }
                else if (wantqs) {
                    /* Path 3 (M much larger than N, JOBZ='S') */
                    wrkbl  =              n +   n * nb;   //magma_ilaenv( 1, "DGEQRF", " ",   m, n, -1, -1 )
                    wrkbl  = max(wrkbl,   n +   n * nb ); //magma_ilaenv( 1, "DORGQR", " ",   m, n,  n, -1 ))
                    wrkbl  = max(wrkbl, 3*n + 2*n * nb ); // dgebrd
                    wrkbl  = max(wrkbl, 3*n +   n * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", n, n,  n, -1 ))
                    wrkbl  = max(wrkbl, 3*n +   n * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", n, n,  n, -1 ))
                    wrkbl  = max(wrkbl, 3*n + bdspac);
                    maxwrk = wrkbl + n*n;
                    minwrk = maxwrk;  // lapack was: bdspac + n*n + 3*n
                }
                else if (wantqa) {
                    /* Path 4 (M much larger than N, JOBZ='A') */
                    wrkbl  =              n +   n * nb;   //magma_ilaenv( 1, "DGEQRF", " ",   m, n, -1, -1 )
                    wrkbl  = max(wrkbl,   n +   m);       // min for dorgqr; preferred is below
                    wrkbl  = max(wrkbl, 3*n + 2*n * nb ); // dgebrd
                    wrkbl  = max(wrkbl, 3*n +   n * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", n, n,  n, -1 ))
                    wrkbl  = max(wrkbl, 3*n +   n * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", n, n,  n, -1 ))
                    wrkbl  = max(wrkbl, 3*n + bdspac);
                    minwrk = wrkbl + n*n;  // lapack was: bdspac + n*n + 2*n + m
                    // include preferred size for dorgqr
                    wrkbl  = max(wrkbl,   n +   m * nb ); //magma_ilaenv( 1, "DORGQR", " ",   m, m,  n, -1 ))
                    maxwrk = wrkbl + n*n;
                }
            }
            else {
                /* Path 5 (M at least N, but not much larger) */
                wrkbl  = 3*n + (m + n) * nb; // dgebrd
                if (wantqn) {
                    maxwrk = max(wrkbl, 3*n + bdspac);
                    minwrk = maxwrk;  // lapack was: 3*n + max(m, bdspac)
                }
                else if (wantqo) {
                    wrkbl  = max(wrkbl, 3*n + n * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", m, n, n, -1 ))
                    wrkbl  = max(wrkbl, 3*n + n * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", n, n, n, -1 ))
                    maxwrk = max(wrkbl, 3*n + m*n + bdspac);
                    minwrk = max(wrkbl, 3*n + n*n + bdspac);  // lapack was: 3*n + max(m, n*n + bdspac)
                }
                else if (wantqs) {
                    wrkbl  = max(wrkbl, 3*n + n * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", m, n, n, -1 ))
                    wrkbl  = max(wrkbl, 3*n + n * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", n, n, n, -1 ))
                    maxwrk = max(wrkbl, 3*n + bdspac);
                    minwrk = maxwrk;  // lapack was: 3*n + max(m, bdspac)
                }
                else if (wantqa) {
                    wrkbl  = max(wrkbl, 3*n + m * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", m, m, n, -1 ))
                    wrkbl  = max(wrkbl, 3*n + n * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", n, n, n, -1 ))
                    maxwrk = max(wrkbl, 3*n + bdspac);
                    minwrk = maxwrk;  // lapack was: 3*n + max(m, bdspac)
                }
            }
        }
        else if (minmn > 0) {
            /* Compute space needed for DBDSDC */
            mnthr = (magma_int_t) (minmn*11. / 6.);
            if (wantqn) {
                bdspac = 7*m;
            }
            else {
                bdspac = 3*m*m + 4*m;
            }
            if (n >= mnthr) {
                if (wantqn) {
                    /* Path 1t (N much larger than M, JOBZ='N') */
                    wrkbl  =              m +   m * nb;  //magma_ilaenv( 1, "DGELQF", " ",   m, n, -1, -1 )
                    wrkbl  = max(wrkbl, 3*m + 2*m * nb); // dgebrd
                    maxwrk = max(wrkbl,  m + bdspac);
                    minwrk = maxwrk;  // lapack was: bdspac + m
                }
                else if (wantqo) {
                    /* Path 2t (N much larger than M, JOBZ='O') */
                    wrkbl  =              m +   m * nb;   //magma_ilaenv( 1, "DGELQF", " ",   m, n, -1, -1 )
                    wrkbl  = max(wrkbl,   m +   m * nb ); //magma_ilaenv( 1, "DORGLQ", " ",   m, n,  m, -1 ))
                    wrkbl  = max(wrkbl, 3*m + 2*m * nb ); // dgebrd
                    wrkbl  = max(wrkbl, 3*m +   m * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", m, m,  m, -1 ))
                    wrkbl  = max(wrkbl, 3*m +   m * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", m, m,  m, -1 ))
                    wrkbl  = max(wrkbl, 3*m + m*m + bdspac);
                    maxwrk = wrkbl + m*m;
                    minwrk = maxwrk;  // lapack was: bdspac + 2*m*m + 3*m
                }
                else if (wantqs) {
                    /* Path 3t (N much larger than M, JOBZ='S') */
                    wrkbl  =              m +   m * nb;   //magma_ilaenv( 1, "DGELQF", " ",   m, n, -1, -1 )
                    wrkbl  = max(wrkbl,   m +   m * nb ); //magma_ilaenv( 1, "DORGLQ", " ",   m, n,  m, -1 ))
                    wrkbl  = max(wrkbl, 3*m + 2*m * nb ); // dgebrd
                    wrkbl  = max(wrkbl, 3*m +   m * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", m, m,  m, -1 ))
                    wrkbl  = max(wrkbl, 3*m +   m * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", m, m,  m, -1 ))
                    wrkbl  = max(wrkbl, 3*m + bdspac);
                    maxwrk = wrkbl + m*m;
                    minwrk = maxwrk;  // lapack was: bdspac + m*m + 3*m
                }
                else if (wantqa) {
                    /* Path 4t (N much larger than M, JOBZ='A') */
                    wrkbl  =              m +   m * nb;   //magma_ilaenv( 1, "DGELQF", " ",   m, n, -1, -1 )
                    wrkbl  = max(wrkbl,   m +   n);       // min for dorgqr; preferred is below
                    wrkbl  = max(wrkbl, 3*m + 2*m * nb ); // dgebrd
                    wrkbl  = max(wrkbl, 3*m +   m * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", m, m,  m, -1 ))
                    wrkbl  = max(wrkbl, 3*m +   m * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", m, m,  m, -1 ))
                    wrkbl  = max(wrkbl, 3*m + bdspac);
                    minwrk = wrkbl + m*m;  // corrected lapack was: bdspac + m*m + 2*m + n
                    // include preferred size for dorgqr
                    wrkbl  = max(wrkbl,   m +   n * nb ); //magma_ilaenv( 1, "DORGLQ", " ",   n, n,  m, -1 ))
                    maxwrk = wrkbl + m*m;
                }
            }
            else {
                /* Path 5t (N greater than M, but not much larger) */
                wrkbl  = 3*m + (m + n) * nb;  // dgebrd
                if (wantqn) {
                    maxwrk = max(wrkbl, 3*m + bdspac);
                    minwrk = maxwrk;  // lapack was: 3*m + max(n, bdspac)
                }
                else if (wantqo) {
                    wrkbl  = max(wrkbl, 3*m + m * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", m, m, n, -1 ))
                    wrkbl  = max(wrkbl, 3*m + m * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", m, n, m, -1 ))
                    maxwrk = max(wrkbl, 3*m + m*n + bdspac);
                    minwrk = max(wrkbl, 3*m + m*m + bdspac);  // lapack was: 3*m + max(n, m*m + bdspac)
                }
                else if (wantqs) {
                    wrkbl  = max(wrkbl, 3*m + m * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", m, m, n, -1 ))
                    wrkbl  = max(wrkbl, 3*m + m * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", m, n, m, -1 ))
                    maxwrk = max(wrkbl, 3*m + bdspac);
                    minwrk = maxwrk;  // lapack was: 3*m + max(n, bdspac)
                }
                else if (wantqa) {
                    wrkbl  = max(wrkbl, 3*m + n * nb ); //magma_ilaenv( 1, "DORMBR", "QLN", m, m, n, -1 ))
                    wrkbl  = max(wrkbl, 3*m + m * nb ); //magma_ilaenv( 1, "DORMBR", "PRT", n, n, m, -1 ))
                    maxwrk = max(wrkbl, 3*m + bdspac);
                    minwrk = maxwrk;  // lapack was: 3*m + max(n, bdspac)
                }
            }
        }
        maxwrk = max(maxwrk,minwrk);
        work[1] = (double) maxwrk;

        if (lwork < minwrk && ! lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return *info;
    }

    /* Get machine constants */
    eps = lapackf77_dlamch("P");
    smlnum = sqrt(lapackf77_dlamch("S")) / eps;
    bignum = 1. / smlnum;

    /* Scale A if max element outside range [SMLNUM,BIGNUM] */
    anrm = lapackf77_dlange("M", &m, &n, A(1,1), &lda, dum);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
        iscl = 1;
        lapackf77_dlascl("G", &izero, &izero, &anrm, &smlnum, &m, &n, A(1,1), &lda, &ierr);
    }
    else if (anrm > bignum) {
        iscl = 1;
        lapackf77_dlascl("G", &izero, &izero, &anrm, &bignum, &m, &n, A(1,1), &lda, &ierr);
    }

    if (m >= n) {
        /* A has at least as many rows as columns. */
        /* If A has sufficiently more rows than columns, first reduce using */
        /* the QR decomposition (if sufficient workspace available) */
        if (m >= mnthr) {
            if (wantqn) {
                /* Path 1 (M much larger than N, JOBZ='N') */
                /* No singular vectors to be computed */
                itau = 1;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (Workspace: need 2*N, prefer N + N*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Zero out below R */
                lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero, A(2,1), &lda);
                ie    = 1;
                itauq = ie    + n;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in A */
                /* (LAPACK Workspace: need 4*N, prefer 3*N + 2*N*NB) */
                /* (MAGMA  Workspace: need 3*N + 2*N*NB) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&n, &n, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_dgebrd(n, n, A(1,1), lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif
                nwork = ie + n;

                /* Perform bidiagonal SVD, computing singular values only */
                /* (Workspace: need N + BDSPAC) */
                lapackf77_dbdsdc("U", "N", &n, s, &work[ie], dum, &ione, dum, &ione, dum, idum, &work[nwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 2 (M much larger than N, JOBZ = 'O') */
                /* N left  singular vectors to be overwritten on A and */
                /* N right singular vectors to be computed in VT */
                ir = 1;

                /* WORK[IR] is LDWRKR by N, at least N*N, up to M*N */
                //if (lwork >= lda*n + 3*n + max( 2*n*nb, n*n + bdspac )) {
                //    ldwrkr = lda;
                //}
                //else {
                    ldwrkr = min( m, (lwork - (3*n + max( 2*n*nb, n*n + bdspac ))) / n );
                //}
                itau = ir + ldwrkr*n;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (Workspace: need [N*N] + 2*N, prefer N*N + N + N*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Copy R to WORK[IR], zeroing out below it */
                lapackf77_dlacpy("U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr);
                lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr);

                /* Generate Q in A */
                /* (Workspace: need [N*N] + 2*N, prefer N*N + N + N*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dorgqr(&m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                ie    = itau;
                itauq = ie    + n;
                itaup = itauq + n;
                nwork = itaup + n;
                
                /* Bidiagonalize R in WORK[IR] */  /* was: R in VT, copying result to ... */
                /* (LAPACK Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB) */
                /* (MAGMA  Workspace: need N*N + 3*N + 2*N*NB) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&n, &n, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_dgebrd(n, n, &work[ir], ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif

                /* WORK[IU] is N by N */
                iu = nwork;
                nwork = iu + n*n;

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in WORK[IU] and */
                /* computing right singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + N*N + BDSPAC + [N*N + 2*N]) */
                lapackf77_dbdsdc("U", "I", &n, s, &work[ie], &work[iu], &n, VT, &ldvt, dum, idum, &work[nwork], iwork, info);

                /* Overwrite WORK[IU] by left  singular vectors of R, and */
                /* overwrite VT       by right singular vectors of R */
                /* (Workspace: need 2*N*N + 3*N + [N], prefer 2*N*N + 2*N + N*NB + [N]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iu], &n,    &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT,        &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, n, n, n, &work[ir], ldwrkr, &work[itauq], &work[iu], n,    &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   n, n, n, &work[ir], ldwrkr, &work[itaup], VT,        ldvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply Q in A by left singular vectors of R in WORK[IU], */
                /* storing result in WORK[IR] and copying to A */
                /* (Workspace: need 2*N*N + [3*N], prefer M*N + N*N + [3*N]) */
                for (i = 1; i <= m; i += ldwrkr) {
                    blk = min(m - i + 1, ldwrkr);
                    blasf77_dgemm("N", "N", &blk, &n, &n, &c_one, A(i,1), &lda, &work[iu], &n, &c_zero, &work[ir], &ldwrkr);
                    lapackf77_dlacpy("F", &blk, &n, &work[ir], &ldwrkr, A(i,1), &lda);
                }
            }
            else if (wantqs) {
                /* Path 3 (M much larger than N, JOBZ='S') */
                /* N left  singular vectors to be computed in U and */
                /* N right singular vectors to be computed in VT */
                ir = 1;

                /* WORK[IR] is N by N */
                ldwrkr = n;
                itau = ir + ldwrkr*n;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Copy R to WORK[IR], zeroing out below it */
                lapackf77_dlacpy("U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr);
                lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr);

                /* Generate Q in A */
                /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dorgqr(&m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                ie    = itau;
                itauq = ie    + n;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in WORK[IR] */
                /* (LAPACK Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB) */
                /* (MAGMA  Workspace: need N*N + 3*N + 2*N*NB) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&n, &n, &work[ir], &ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_dgebrd(n, n, &work[ir], ldwrkr, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in U and */
                /* computing right singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + BDSPAC + [N*N + 2*N]) */
                lapackf77_dbdsdc("U", "I", &n, s, &work[ie], U, &ldu, VT, &ldvt, dum, idum, &work[nwork], iwork, info);

                /* Overwrite U  by left  singular vectors of R, and */
                /* overwrite VT by right singular vectors of R */
                /* (Workspace: need N*N + 3*N + [N], prefer N*N + 2*N + N*NB + [N]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, n, n, n, &work[ir], ldwrkr, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   n, n, n, &work[ir], ldwrkr, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply Q in A by left singular vectors of R in WORK[IR], */
                /* storing result in U */
                /* (Workspace: need N*N) */
                lapackf77_dlacpy("F", &n, &n, U, &ldu, &work[ir], &ldwrkr);
                blasf77_dgemm("N", "N", &m, &n, &n, &c_one, A(1,1), &lda, &work[ir], &ldwrkr, &c_zero, U, &ldu);
            }
            else if (wantqa) {
                /* Path 4 (M much larger than N, JOBZ='A') */
                /* M left  singular vectors to be computed in U and */
                /* N right singular vectors to be computed in VT */
                iu = 1;

                /* WORK[IU] is N by N */
                ldwrku = n;
                itau = iu + ldwrku*n;
                nwork = itau + n;

                /* Compute A=Q*R, copying result to U */
                /* (was:       need N*N + N + M, prefer N*N + N + M*NB in lapack 3.4.2; correct in clapack 3.2.1) */
                /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                lapackf77_dlacpy("L", &m, &n, A(1,1), &lda, U, &ldu);

                /* Generate Q in U */
                /* (Workspace: need [N*N] + N + M, prefer [N*N] + N + M*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dorgqr(&m, &m, &n, U, &ldu, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Produce R in A, zeroing out other entries */
                lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero, A(2,1), &lda);
                ie    = itau;
                itauq = ie    + n;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in A */
                /* (LAPACK Workspace: need [N*N] + 4*N, prefer [N*N] + 3*N + 2*N*NB) */
                /* (MAGMA  Workspace: need [N*N] + 3*N + 2*N*NB) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&n, &n, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_dgebrd(n, n, A(1,1), lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in WORK[IU] and */
                /* computing right singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + N*N + BDSPAC + [2*N]) */
                lapackf77_dbdsdc("U", "I", &n, s, &work[ie], &work[iu], &n, VT, &ldvt, dum, idum, &work[nwork], iwork, info);

                /* Overwrite WORK[IU] by left  singular vectors of R, and */
                /* overwrite VT       by right singular vectors of R */
                /* (Workspace: need N*N + 3*N + [N], prefer N*N + 2*N + N*NB + [N]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &n, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, A(1,1), &lda, &work[itaup], VT,        &ldvt,   &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, n, n, n, A(1,1), lda, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   n, n, n, A(1,1), lda, &work[itaup], VT,        ldvt,   &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply Q in U by left singular vectors of R in WORK[IU], */
                /* storing result in A */
                /* (Workspace: need N*N) */
                blasf77_dgemm("N", "N", &m, &n, &n, &c_one, U, &ldu, &work[iu], &ldwrku, &c_zero, A(1,1), &lda);

                /* Copy left singular vectors of A from A to U */
                lapackf77_dlacpy("F", &m, &n, A(1,1), &lda, U, &ldu);
            }
        }
        else {
            /* M < MNTHR */
            /* Path 5 (M at least N, but not much larger) */
            /* Reduce to bidiagonal form without QR decomposition */
            ie    = 1;
            itauq = ie    + n;
            itaup = itauq + n;
            nwork = itaup + n;

            /* Bidiagonalize A */
            /* (LAPACK Workspace: need 3*N + M, prefer 3*N + (M + N)*NB) */
            /* (MAGMA  Workspace: need 3*N + (M + N)*NB) */
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_dgebrd(&m, &n, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
            #else
            magma_dgebrd(m, n, A(1,1), lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
            #endif
            
            if (wantqn) {
                /* Path 5n (M >= N, JOBZ='N') */
                /* Perform bidiagonal SVD, computing singular values only */
                /* (Workspace: need N + BDSPAC + [2*N]) */
                lapackf77_dbdsdc("U", "N", &n, s, &work[ie], dum, &ione, dum, &ione, dum, idum, &work[nwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 5o (M >= N, JOBZ='O') */
                iu = nwork;
                if (lwork >= m*n + 3*n + bdspac) {
                    /* WORK[ IU ] is M by N */
                    ldwrku = m;
                    nwork = iu + ldwrku*n;
                    lapackf77_dlaset("F", &m, &n, &c_zero, &c_zero, &work[iu], &ldwrku);
                }
                else {
                    /* WORK[ IU ] is N by N */
                    ldwrku = n;
                    nwork = iu + ldwrku*n;

                    /* WORK[IR] is LDWRKR by N */
                    ir = nwork;
                    ldwrkr = (lwork - n*n - 3*n) / n;
                }
                nwork = iu + ldwrku*n;  // todo redundant?

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in WORK[IU] and */
                /* computing right singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + N*N + BDSPAC + [2*N]) */
                lapackf77_dbdsdc("U", "I", &n, s, &work[ie], &work[iu], &ldwrku, VT, &ldvt, dum, idum, &work[nwork], iwork, info);

                /* Overwrite VT by right singular vectors of A */
                /* (Workspace: need N*N + 2*N + [2*N], prefer N*N + N + N*NB + [2*N]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif

                /* TODO should be m*n + 4*n + n*nb, instead of bdspac? */
                /* Affects n = 1, ..., 9 for nb=32. */
                if (lwork >= m*n + 3*n + bdspac) {
                    /* Overwrite WORK[IU] by left singular vectors of A */
                    /* (was:       need N*N + 2*N + [2*N], prefer N*N + N + N*NB + [2*N]) */
                    /* (Workspace: need M*N + 2*N + [2*N], prefer M*N + N + N*NB + [2*N]) */
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr("Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr);
                    #else
                    magma_dormbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, n, n, A(1,1), lda, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr);
                    #endif
                
                    /* Copy left singular vectors of A from WORK[IU] to A */
                    lapackf77_dlacpy("F", &m, &n, &work[iu], &ldwrku, A(1,1), &lda);
                }
                else {
                    /* Generate Q in A */
                    /* (Workspace: need N*N + 2*N + [2*N], prefer N*N + N + N*NB + [2*N]) */
                    lnwork = lwork - nwork + 1;
                    lapackf77_dorgbr("Q", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[nwork], &lnwork, &ierr);

                    /* Multiply Q in A by left singular vectors of */
                    /* bidiagonal matrix in WORK[IU], storing result in */
                    /* WORK[IR] and copying to A */
                    /* (Workspace: need 2*N*N + [3*N], prefer N*N + M*N + [3*N]) */
                    for (i = 1; i <= m; i += ldwrkr) {
                        blk = min(m - i + 1, ldwrkr);
                        blasf77_dgemm("N", "N", &blk, &n, &n, &c_one, A(i,1), &lda, &work[iu], &ldwrku, &c_zero, &work[ir], &ldwrkr);
                        lapackf77_dlacpy("F", &blk, &n, &work[ir], &ldwrkr, A(i,1), &lda);
                    }
                }
            }
            else if (wantqs) {
                /* Path 5s (M >= N, JOBZ='S') */
                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in U and */
                /* computing right singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + BDSPAC + [2*N]) */
                lapackf77_dlaset("F", &m, &n, &c_zero, &c_zero, U, &ldu);
                lapackf77_dbdsdc("U", "I", &n, s, &work[ie], U, &ldu, VT, &ldvt, dum, idum, &work[nwork], iwork, info);

                /* Overwrite U  by left  singular vectors of A, and */
                /* overwrite VT by right singular vectors of A */
                /* (Workspace: need 3*N + [N], prefer 2*N + N*NB + [N]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, m, n, n, A(1,1), lda, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif
            }
            else if (wantqa) {
                /* Path 5a (M >= N, JOBZ='A') */
                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in U and */
                /* computing right singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + BDSPAC + [2*N]) */
                lapackf77_dlaset("F", &m, &m, &c_zero, &c_zero, U, &ldu);
                lapackf77_dbdsdc("U", "I", &n, s, &work[ie], U, &ldu, VT, &ldvt, dum, idum, &work[nwork], iwork, info);

                /* Set the right corner of U to identity matrix */
                if (m > n) {
                    i__1 = m - n;
                    lapackf77_dlaset("F", &i__1, &i__1, &c_zero, &c_one, U(n,n), &ldu);
                }

                /* Overwrite U  by left  singular vectors of A, and */
                /* overwrite VT by right singular vectors of A */
                /* (Workspace: need 2*N + M + [N], prefer 2*N + M*NB + [N]) */
                /* (was:       need N*N + 2*N + M + [N], prefer N*N + 2*N + M*NB + [N]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   n, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif
            }
        }
    }
    else {
        /* A has more columns than rows. */
        /* If A has sufficiently more columns than rows, first reduce using */
        /* the LQ decomposition (if sufficient workspace available) */
        if (n >= mnthr) {
            if (wantqn) {
                /* Path 1t (N much larger than M, JOBZ='N') */
                /* No singular vectors to be computed */
                itau = 1;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (Workspace: need 2*M, prefer M + M*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Zero out above L */
                lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero, A(1,2), &lda);
                ie    = 1;
                itauq = ie    + m;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in A */
                /* (LAPACK Workspace: need 4*M, prefer 3*M + 2*M*NB) */
                /* (MAGMA  Workspace: need 3*M + 2*M*NB) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&m, &m, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_dgebrd(m, m, A(1,1), lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif
                nwork = ie + m;

                /* Perform bidiagonal SVD, computing singular values only */
                /* (Workspace: need M + BDSPAC) */
                lapackf77_dbdsdc("U", "N", &m, s, &work[ie], dum, &ione, dum, &ione, dum, idum, &work[nwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 2t (N much larger than M, JOBZ='O') */
                /* M right singular vectors to be overwritten on A and */
                /* M left  singular vectors to be computed in U */
                // LAPACK put ivt first, but it isn't needed until after gebrd;
                // putting it later matches Path 2.
                il = 1;

                /* WORK[IL] is M by chunk, at least M*M, up to M*N */
                //if (lwork >= m*n + 3*m + max( 2*m*nb, m*m + bdspac )) {
                //    ldwrkl = m;
                //    chunk = n;
                //}
                //else {
                    ldwrkl = m;
                    // was: chunk = (lwork - m*m) / m;
                    chunk = min( n, (lwork - (3*m + max( 2*m*nb, m*m + bdspac ))) / m );
                //}
                itau = il + ldwrkl*m;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (Workspace: need [M*M] + 2*M, prefer M*M + M + M*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Copy L to WORK[IL], zeroing out above it */
                lapackf77_dlacpy("L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl);
                lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl);

                /* Generate Q in A */
                /* (Workspace: need [M*M] + 2*M, prefer M*M + M + M*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dorglq(&m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                ie    = itau;
                itauq = ie    + m;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in WORK[IL] */
                /* (LAPACK Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB) */
                /* (MAGMA  Workspace: need M*M + 3*M + 2*M*NB) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&m, &m, &work[il], &ldwrkl, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_dgebrd(m, m, &work[il], ldwrkl, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif

                /* WORK[IVT] is M by M */
                ivt = nwork;
                nwork = ivt + m*m;

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in U, and */
                /* computing right singular vectors of bidiagonal matrix in WORK[IVT] */
                /* (Workspace: need M + M*M + BDSPAC + [M*M + 2*M]) */
                lapackf77_dbdsdc("U", "I", &m, s, &work[ie], U, &ldu, &work[ivt], &m, dum, idum, &work[nwork], iwork, info);

                /* Overwrite U         by left  singular vectors of L, and */
                /* overwrite WORK[IVT] by right singular vectors of L */
                /* (Workspace: need 2*M*M + 3*M + [M], prefer 2*M*M + 2*M + M*NB + [M]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U,          &ldu, &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], &work[ivt], &m,   &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, m, &work[il], ldwrkl, &work[itauq], U,          ldu, &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   m, m, m, &work[il], ldwrkl, &work[itaup], &work[ivt], m,   &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply right singular vectors of L in WORK[IVT] by Q in A, */
                /* storing result in WORK[IL] and copying to A */
                /* (Workspace: need 2*M*M + [3*M], prefer M*N + M*M + [3*M]) */
                for (i = 1; i <= n; i += chunk) {
                    blk = min(n - i + 1, chunk);
                    blasf77_dgemm("N", "N", &m, &blk, &m, &c_one, &work[ivt], &m, A(1,i), &lda, &c_zero, &work[il], &ldwrkl);
                    lapackf77_dlacpy("F", &m, &blk, &work[il], &ldwrkl, A(1,i), &lda);
                }
            }
            else if (wantqs) {
                /* Path 3t (N much larger than M, JOBZ='S') */
                /* M right singular vectors to be computed in VT and */
                /* M left  singular vectors to be computed in U */
                il = 1;

                /* WORK[IL] is M by M */
                ldwrkl = m;
                itau = il + ldwrkl*m;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Copy L to WORK[IL], zeroing out above it */
                lapackf77_dlacpy("L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl);
                lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl);

                /* Generate Q in A */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dorglq(&m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                ie    = itau;
                itauq = ie    + m;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in WORK[IU] */ /* was: ..., copying result to U */
                /* (LAPACK Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB) */
                /* (MAGMA  Workspace: need M*M + 3*M + 2*M*NB) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&m, &m, &work[il], &ldwrkl, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_dgebrd(m, m, &work[il], ldwrkl, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in U and */
                /* computing right singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need M + BDSPAC + [M*M + 2*M]) */
                lapackf77_dbdsdc("U", "I", &m, s, &work[ie], U, &ldu, VT, &ldvt, dum, idum, &work[nwork], iwork, info);

                /* Overwrite U  by left  singular vectors of L, and */
                /* overwrite VT by right singular vectors of L */
                /* (Workspace: need M*M + 3*M + [M], prefer M*M + 2*M + M*NB + [M]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, m, &work[il], ldwrkl, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   m, m, m, &work[il], ldwrkl, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply right singular vectors of L in WORK[IL] by Q in A, */
                /* storing result in VT */
                /* (Workspace: need M*M) */
                lapackf77_dlacpy("F", &m, &m, VT, &ldvt, &work[il], &ldwrkl);
                blasf77_dgemm("N", "N", &m, &n, &m, &c_one, &work[il], &ldwrkl, A(1,1), &lda, &c_zero, VT, &ldvt);
            }
            else if (wantqa) {
                /* Path 4t (N much larger than M, JOBZ='A') */
                /* N right singular vectors to be computed in VT and */
                /* M left  singular vectors to be computed in U */
                ivt = 1;

                /* WORK[IVT] is M by M */
                ldwkvt = m;
                itau = ivt + ldwkvt*m;
                nwork = itau + m;

                /* Compute A=L*Q, copying result to VT */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                lapackf77_dlacpy("U", &m, &n, A(1,1), &lda, VT, &ldvt);

                /* Generate Q in VT */
                /* (was:       need [M*M] + 2*M,   prefer [M*M] + M + M*NB) */
                /* (Workspace: need [M*M] + M + N, prefer [M*M] + M + N*NB) */
                lnwork = lwork - nwork + 1;
                lapackf77_dorglq(&n, &n, &m, VT, &ldvt, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Produce L in A, zeroing out other entries */
                lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero, A(1,2), &lda);
                ie    = itau;
                itauq = ie    + m;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in A */
                /* (LAPACK Workspace: need [M*M] + 4*M, prefer [M*M] + 3*M + 2*M*NB) */
                /* (MAGMA  Workspace: need [M*M] + 3*M + 2*M*NB) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&m, &m, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_dgebrd(m, m, A(1,1), lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in U and */
                /* computing right singular vectors of bidiagonal matrix in WORK[IVT] */
                /* (Workspace: need M + M*M + BDSPAC + [2*M]) */
                lapackf77_dbdsdc("U", "I", &m, s, &work[ie], U, &ldu, &work[ivt], &ldwkvt, dum, idum, &work[nwork], iwork, info);

                /* Overwrite U         by left  singular vectors of L, and */
                /* overwrite WORK[IVT] by right singular vectors of L */
                /* (Workspace: need M*M + 3*M + [M], prefer M*M + 2*M + M*NB + [M]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &m, &m, &m, A(1,1), &lda, &work[itauq], U,          &ldu,    &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &m, &m, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, m, A(1,1), lda, &work[itauq], U,          ldu,    &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   m, m, m, A(1,1), lda, &work[itaup], &work[ivt], ldwkvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply right singular vectors of L in WORK[IVT] by Q in VT, */
                /* storing result in A */
                /* (Workspace: need M*M) */
                blasf77_dgemm("N", "N", &m, &n, &m, &c_one, &work[ivt], &ldwkvt, VT, &ldvt, &c_zero, A(1,1), &lda);

                /* Copy right singular vectors of A from A to VT */
                lapackf77_dlacpy("F", &m, &n, A(1,1), &lda, VT, &ldvt);
            }
        }
        else {
            /* N < MNTHR */
            /* Path 5t (N greater than M, but not much larger) */
            /* Reduce to bidiagonal form without LQ decomposition */
            ie    = 1;
            itauq = ie    + m;
            itaup = itauq + m;
            nwork = itaup + m;

            /* Bidiagonalize A */
            /* (LAPACK Workspace: need 3*M + N, prefer 3*M + (M + N)*NB) */
            /* (MAGMA  Workspace: need 3*M + (M + N)*NB) */
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_dgebrd(&m, &n, A(1,1), &lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
            #else
            magma_dgebrd(m, n, A(1,1), lda, s, &work[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
            #endif
            
            if (wantqn) {
                /* Path 5tn (N > M, JOBZ='N') */
                /* Perform bidiagonal SVD, computing singular values only */
                /* (Workspace: need M + BDSPAC + [2*M]) */
                lapackf77_dbdsdc("L", "N", &m, s, &work[ie], dum, &ione, dum, &ione, dum, idum, &work[nwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 5to (N > M, JOBZ='O') */
                ldwkvt = m;
                ivt = nwork;
                if (lwork >= m*n + 3*m + bdspac) {
                    /* WORK[ IVT ] is M by N */
                    lapackf77_dlaset("F", &m, &n, &c_zero, &c_zero, &work[ivt], &ldwkvt);
                    nwork = ivt + ldwkvt*n;
                }
                else {
                    /* WORK[ IVT ] is M by M */
                    nwork = ivt + ldwkvt*m;
                    il = nwork;

                    /* WORK[IL] is M by CHUNK */
                    chunk = (lwork - m*m - 3*m) / m;
                }

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in U and */
                /* computing right singular vectors of bidiagonal matrix in WORK[IVT] */
                /* (Workspace: need M + M*M + BDSPAC + [2*M]) */
                /* (was:       need     M*M + BDSPAC) */
                lapackf77_dbdsdc("L", "I", &m, s, &work[ie], U, &ldu, &work[ivt], &ldwkvt, dum, idum, &work[nwork], iwork, info);

                /* Overwrite U by left singular vectors of A */
                /* (Workspace: need M*M + 2*M + [2*M], prefer M*M + M + M*NB + [2*M]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                /* TODO should be m*n + 4*m + m*nb, instead of bdspac? */
                /* Affects m = 1, ..., 9 for nb=32. */
                if (lwork >= m*n + 3*m + bdspac) {
                    /* Overwrite WORK[IVT] by left singular vectors of A */
                    /* (was:       need M*M + 2*M + [2*M], prefer M*M + M + M*NB + [2*M]) */
                    /* (Workspace: need M*N + 2*M + [2*M], prefer M*N + M + M*NB + [2*M]) */
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_dormbr("P", "R", "T", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &lnwork, &ierr);
                    #else
                    magma_dormbr(MagmaP, MagmaRight, MagmaTrans, m, n, m, A(1,1), lda, &work[itaup], &work[ivt], ldwkvt, &work[nwork], lnwork, &ierr);
                    #endif
                
                    /* Copy right singular vectors of A from WORK[IVT] to A */
                    lapackf77_dlacpy("F", &m, &n, &work[ivt], &ldwkvt, A(1,1), &lda);
                }
                else {
                    /* Generate P**T in A */
                    /* (Workspace: need M*M + 2*M + [2*M], prefer M*M + M + M*NB + [2*M]) */
                    lnwork = lwork - nwork + 1;
                    lapackf77_dorgbr("P", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[nwork], &lnwork, &ierr);

                    /* Multiply Q in A by right singular vectors of */
                    /* bidiagonal matrix in WORK[IVT], storing result in */
                    /* WORK[IL] and copying to A */
                    /* (Workspace: need 2*M*M + [3*M], prefer M*M + M*N + [3*M]) */
                    for (i = 1; i <= n; i += chunk) {
                        blk = min(n - i + 1, chunk);
                        blasf77_dgemm("N", "N", &m, &blk, &m, &c_one, &work[ivt], &ldwkvt, A(1,i), &lda, &c_zero, &work[il], &m);
                        lapackf77_dlacpy("F", &m, &blk, &work[il], &m, A(1,i), &lda);
                    }
                }
            }
            else if (wantqs) {
                /* Path 5ts (N > M, JOBZ='S') */
                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in U and */
                /* computing right singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need M + BDSPAC + [2*M]) */
                lapackf77_dlaset("F", &m, &n, &c_zero, &c_zero, VT, &ldvt);
                lapackf77_dbdsdc("L", "I", &m, s, &work[ie], U, &ldu, VT, &ldvt, dum, idum, &work[nwork], iwork, info);

                /* Overwrite U  by left  singular vectors of A, and */
                /* overwrite VT by right singular vectors of A */
                /* (Workspace: need 3*M + [M], prefer 2*M + M*NB + [M]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &m, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   m, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif
            }
            else if (wantqa) {
                /* Path 5ta (N > M, JOBZ='A') */
                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in U and */
                /* computing right singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need M + BDSPAC + [2*M]) */
                lapackf77_dlaset("F", &n, &n, &c_zero, &c_zero, VT, &ldvt);
                lapackf77_dbdsdc("L", "I", &m, s, &work[ie], U, &ldu, VT, &ldvt, dum, idum, &work[nwork], iwork, info);

                /* Set the right corner of VT to identity matrix */
                if (n > m) {
                    i__1 = n - m;
                    lapackf77_dlaset("F", &i__1, &i__1, &c_zero, &c_one, VT(m,m), &ldvt);
                }

                /* Overwrite U  by left  singular vectors of A, and */
                /* overwrite VT by right singular vectors of A */
                /* (Workspace: need 2*M + N + [M], prefer 2*M + N*NB + [M]) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_dormbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U,  &ldu,  &work[nwork], &lnwork, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_dormbr(MagmaQ, MagmaLeft,  MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U,  ldu,  &work[nwork], lnwork, &ierr);
                magma_dormbr(MagmaP, MagmaRight, MagmaTrans,   n, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif
            }
        }
    }

    /* Undo scaling if necessary */
    if (iscl == 1) {
        if (anrm > bignum) {
            lapackf77_dlascl("G", &izero, &izero, &bignum, &anrm, &minmn, &ione, s, &minmn, &ierr);
        }
        if (anrm < smlnum) {
            lapackf77_dlascl("G", &izero, &izero, &smlnum, &anrm, &minmn, &ione, s, &minmn, &ierr);
        }
    }

    /* Return optimal workspace in WORK[0] */
    work[1] = (double) maxwrk;

    return *info;
} /* magma_dgesdd */
