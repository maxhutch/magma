/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
       @generated from zgesdd.cpp normal z -> c, Fri Jan 30 19:00:19 2015

*/
#include "common_magma.h"

#define PRECISION_c
#define COMPLEX

// Version 1 - LAPACK
// Version 2 - MAGMA
#define VERSION 2

/**
    Purpose
    -------
    CGESDD computes the singular value decomposition (SVD) of a complex
    M-by-N matrix A, optionally computing the left and right singular
    vectors, by using divide-and-conquer method. The SVD is written

         A = U * SIGMA * conjugate-transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
    V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note that the routine returns VT = V**H, not V.

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
      -     = MagmaAllVec:  all M columns of U and all N rows of V**H are
                            returned in the arrays U and VT;
      -     = MagmaSomeVec: the first min(M,N) columns of U and
                            the first min(M,N) rows of V**H are
                            returned in the arrays U and VT;
      -     = MagmaOverwriteVec:
                    If M >= N, the first N columns of U are overwritten
                    on the array A and all rows of V**H are returned in
                    the array VT;
                    otherwise, all columns of U are returned in the
                    array U and the first M rows of V**H are overwritten
                    on the array A;
      -     = MagmaNoVec:   no columns of U or rows of V**H are computed.

    @param[in]
    m       INTEGER
            The number of rows of the input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the input matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
      -     if JOBZ = MagmaOverwriteVec,
                if M >= N, A is overwritten with the first N columns
                of U (the left singular vectors, stored columnwise);
                otherwise, A is overwritten with the first M rows
                of V**H (the right singular vectors, stored rowwise).
      -     if JOBZ != MagmaOverwriteVec, the contents of A are destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    s       REAL array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).

    @param[out]
    U       COMPLEX array, dimension (LDU,UCOL)
            UCOL = M if JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N;
            UCOL = min(M,N) if JOBZ = MagmaSomeVec.
      -     If JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N,
            U contains the M-by-M unitary matrix U;
      -     if JOBZ = MagmaSomeVec, U contains the first min(M,N) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBZ = MagmaOverwriteVec and M >= N, or JOBZ = MagmaNoVec, U is not referenced.

    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBZ = MagmaSomeVec or MagmaAllVec or JOBZ = MagmaOverwriteVec and M < N, LDU >= M.

    @param[out]
    VT      COMPLEX array, dimension (LDVT,N)
      -     If JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M >= N,
            VT contains the N-by-N unitary matrix V**H;
      -     if JOBZ = MagmaSomeVec, VT contains the first min(M,N) rows of
            V**H (the right singular vectors, stored rowwise);
      -     if JOBZ = MagmaOverwriteVec and M < N, or JOBZ = MagmaNoVec, VT is not referenced.

    @param[in]
    ldvt    INTEGER
            The leading dimension of the array VT.  LDVT >= 1; if
            JOBZ = MagmaAllVec or JOBZ = MagmaOverwriteVec and M >= N, LDVT >= N;
            if JOBZ = MagmaSomeVec, LDVT >= min(M,N).

    @param[out]
    work    (workspace) COMPLEX array, dimension (MAX(1,lwork))
            On exit, if INFO = 0, WORK[0] returns the optimal lwork.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            Let x = max(M,N) and y = min(M,N). The optimal block size
            nb can be obtained through magma_get_dgesvd_nb(N).
            The threshold for x >> y currently is x >= int( y*17/9 ).
            *Required size different than in LAPACK.* In most cases, these
            sizes should give optimal performance for both MAGMA and LAPACK.
      -     If JOBZ = MagmaNoVec,
                if x >> y, LWORK >= 2*y + (2*y)*nb;
                otherwise, LWORK >= 2*y + (x+y)*nb.
      -     If JOBZ = MagmaOverwriteVec,
                if x >> y, LWORK >= 2*y*y + 2*y +      (2*y)*nb;
                otherwise, LWORK >=         2*y + max( (x+y)*nb, y*y + x    ),
                   prefer  LWORK >=         2*y + max( (x+y)*nb, x*y + y*nb ).
      -     If JOBZ = MagmaSomeVec,
                if x >> y, LWORK >= y*y + 2*y + (2*y)*nb;
                otherwise, LWORK >=       2*y + (x+y)*nb.
      -     If JOBZ = MagmaAllVec,
                if x >> y, LWORK >= y*y + 2*y + max( (2*y)*nb, x    ),
                   prefer  LWORK >= y*y + 2*y + max( (2*y)*nb, x*nb );
                otherwise, LWORK >=       2*y +      (x+y)*nb.
      \n
            If lwork = -1, a workspace query is assumed.  The optimal
            size for the WORK array is calculated and stored in WORK[0],
            and no other work except argument checking is performed.

    @param
    rwork   (workspace) REAL array, dimension (MAX(1,LRWORK))
            Let x = max(M,N) and y = min(M,N).
            These sizes should work for both MAGMA and LAPACK.
            If JOBZ =  MagmaNoVec, LRWORK >= 5*y.
            If JOBZ != MagmaNoVec,
                if x >> y,  LRWORK >=      5*y*y + 5*y;
                otherwise,  LRWORK >= max( 5*y*y + 5*y,
                                           2*x*y + 2*y*y + y ).
    \n
            For JOBZ = MagmaNoVec, some implementations seem to have a bug requiring
            LRWORK >= 7*y in some cases.

    @param
    iwork   (workspace) INTEGER array, dimension (8*min(M,N))

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  The updating process of SBDSDC did not converge.

    Further Details
    ---------------
    Based on contributions by
    Ming Gu and Huan Ren, Computer Science Division, University of
    California at Berkeley, USA

    @ingroup magma_cgesvd_driver
    ********************************************************************/
extern "C" magma_int_t
magma_cgesdd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float *s,
    magmaFloatComplex *U, magma_int_t ldu,
    magmaFloatComplex *VT, magma_int_t ldvt,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *iwork,
    magma_int_t *info)
{
    #define A(i_,j_) (A + (i_) + (j_)*lda)
    #define U(i_,j_) (U + (i_) + (j_)*ldu)

    /* Constants */
    const magmaFloatComplex c_zero = MAGMA_C_ZERO;
    const magmaFloatComplex c_one  = MAGMA_C_ONE;
    const magma_int_t izero = 0;
    const magma_int_t ione  = 1;

    /* Local variables */
    magma_int_t lnwork, i__1;
    magma_int_t i, ie, il, ir, iu, blk, nb;
    float anrm, dum[1], eps, bignum, smlnum;
    magma_int_t iru, ivt, iscl;
    magma_int_t idum[1], ierr, itau, irvt;
    magma_int_t chunk;
    magma_int_t wrkbl, itaup, itauq;
    magma_int_t nwork;
    magma_int_t ldwrkl;
    magma_int_t ldwrkr, minwrk, ldwrku, maxwrk;
    magma_int_t ldwkvt;
    magma_int_t nrwork;
    
    /* Parameter adjustments */
    A  -= 1 + lda;
    --work;
    --rwork;

    /* Function Body */
    *info = 0;
    const magma_int_t m_1 = m - 1;
    const magma_int_t n_1 = n - 1;
    const magma_int_t minmn   = min(m,n);
    /* Note: rwork in path 5, jobz=O depends on mnthr1 < 2 * minmn. */
    const magma_int_t mnthr1  = (magma_int_t) (minmn * 17. / 9.);
    const magma_int_t mnthr2  = (magma_int_t) (minmn * 5. / 3.);
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
    /* CWorkspace refers to complex workspace, and RWorkspace to */
    /* real workspace. NB refers to the optimal block size for the */
    /* immediately following subroutine, as returned by ILAENV.) */
    /* We assume MAGMA's nb >= LAPACK's nb for all routines, */
    /* because calling Fortran's ILAENV is not portable. */
    if (*info == 0 && m > 0 && n > 0) {
        if (m >= n) {
            /* There is no complex work space needed for bidiagonal SVD */
            /* The real work space needed for bidiagonal SVD is */
            /* BDSPAC for computing singular values and singular vectors; */
            /* BDSPAN for computing singular values only. */
            /* BDSPAC is r.ie (n) + r.iru (n*n) + r.irvt (n*n) + r.nrwork (3*N*N + 4*N) */
            /* BDSPAN is r.ie (n) + r.nrwork (4*N) -- doesn't need 7*N? */
            /* BDSPAC = 5*N*N + 5*N */  /* lapack was 5*N*N + 7*N -- seems bigger than needed. */
            /* BDSPAN = 5*N */          /* lapack was MAX(7*N + 4, 3*N + 2 + SMLSIZ*(SMLSIZ + 8)) -- seems bigger than needed. */
            if (m >= mnthr1) {
                if (wantqn) {
                    /* Path 1 (M much larger than N, JOBZ='N') */
                    maxwrk =                n +   n * nb;   //magma_ilaenv( 1, "CGEQRF", " ",   m, n, -1, -1 );
                    maxwrk = max( maxwrk, 2*n + 2*n * nb ); // cgebrd
                    minwrk = maxwrk;  // lapack was: 3*n
                }
                else if (wantqo) {
                    /* Path 2 (M much larger than N, JOBZ='O') */
                    wrkbl  =               n +   n * nb;   //magma_ilaenv( 1, "CGEQRF", " ",   m, n, -1, -1 );
                    wrkbl  = max( wrkbl,   n +   n * nb ); //magma_ilaenv( 1, "CUNGQR", " ",   m, n,  n, -1 ));
                    wrkbl  = max( wrkbl, 2*n + 2*n * nb ); // cgebrd
                    wrkbl  = max( wrkbl, 2*n +   n * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", n, n,  n, -1 ));
                    wrkbl  = max( wrkbl, 2*n +   n * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", n, n,  n, -1 ));
                    maxwrk = m*n + n*n + wrkbl;  // TODO is m*n vs n*n significant speedup?
                    minwrk = 2*n*n     + wrkbl;  // lapack was: 2*n*n + 3*n
                }
                else if (wantqs) {
                    /* Path 3 (M much larger than N, JOBZ='S') */
                    wrkbl  =               n +   n * nb;   //magma_ilaenv( 1, "CGEQRF", " ",   m, n, -1, -1 );
                    wrkbl  = max( wrkbl,   n +   n * nb ); //magma_ilaenv( 1, "CUNGQR", " ",   m, n,  n, -1 ));
                    wrkbl  = max( wrkbl, 2*n + 2*n * nb ); // cgebrd
                    wrkbl  = max( wrkbl, 2*n +   n * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", n, n,  n, -1 ));
                    wrkbl  = max( wrkbl, 2*n +   n * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", n, n,  n, -1 ));
                    maxwrk = n*n + wrkbl;
                    minwrk = maxwrk;  // lapack was: n*n + 3*n
                }
                else if (wantqa) {
                    /* Path 4 (M much larger than N, JOBZ='A') */
                    wrkbl  =               n +   n * nb;   //magma_ilaenv( 1, "CGEQRF", " ",   m, n, -1, -1 );
                    wrkbl  = max( wrkbl,   n +   m );      // min for cungqr; preferred is below
                    wrkbl  = max( wrkbl, 2*n + 2*n * nb ); // cgebrd
                    wrkbl  = max( wrkbl, 2*n +   n * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", n, n,  n, -1 ));
                    wrkbl  = max( wrkbl, 2*n +   n * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", n, n,  n, -1 ));
                    minwrk = n*n + wrkbl;  // lapack was: n*n + 2*n + m
                    // include preferred size for cungqr
                    wrkbl  = max( wrkbl,   n +   m * nb ); //magma_ilaenv( 1, "CUNGQR", " ",   m, m,  n, -1 ));
                    maxwrk = n*n + wrkbl;
                }
            }
            else if (m >= mnthr2) {
                /* Path 5 (M much larger than N, but not as much as MNTHR1) */
                maxwrk = 2*n + (m + n) * nb;  // cgebrd
                minwrk = maxwrk;  // lapack was: 2*n + m
                if (wantqo) {
                    maxwrk = max( maxwrk, 2*n + n * nb );  //magma_ilaenv( 1, "CUNGBR", "P", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + n * nb );  //magma_ilaenv( 1, "CUNGBR", "Q", m, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + n*n + m ); // lapack was: maxwrk += m*n  // todo no m*n?  // extra +m for lapack compatability; not needed
                    minwrk = maxwrk;                       // lapack was: minwrk += n*n
                }
                else if (wantqs) {
                    maxwrk = max( maxwrk, 2*n + n * nb );  //magma_ilaenv( 1, "CUNGBR", "P", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + n * nb );  //magma_ilaenv( 1, "CUNGBR", "Q", m, n, n, -1 ));
                }
                else if (wantqa) {
                    maxwrk = max( maxwrk, 2*n + n * nb );  //magma_ilaenv( 1, "CUNGBR", "P", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + m * nb );  //magma_ilaenv( 1, "CUNGBR", "Q", m, m, n, -1 ));
                }
            }
            else {
                /* Path 6 (M at least N, but not much larger) */
                maxwrk = 2*n + (m + n) * nb;  // cgebrd
                minwrk = maxwrk;  // lapack was: 2*n + m
                if (wantqo) {
                    maxwrk = max( maxwrk, 2*n + n*n + n * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + m*n + n * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", m, n, n, -1 ));
                    // lapack was maxwrk += m*n n*n and m*n put into unmbr MAX above
                    minwrk = max( minwrk, 2*n + n*n );  // lapack was minwrk += n*n
                }
                else if (wantqs) {
                    maxwrk = max( maxwrk, 2*n + n * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + n * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", m, n, n, -1 ));
                }
                else if (wantqa) {
                    maxwrk = max( maxwrk, 2*n + n * nb ); //magma_ilaenv( 1, "CUNGBR", "PRC", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + m * nb ); //magma_ilaenv( 1, "CUNGBR", "QLN", m, m, n, -1 ));
                }
            }
        }
        else {
            /* There is no complex work space needed for bidiagonal SVD */
            /* The real work space needed for bidiagonal SVD is */
            /* BDSPAC for computing singular values and singular vectors; */
            /* BDSPAN for computing singular values only. */
            /* BDSPAC = 5*M*M + 5*M */  /* lapack was 5*M*M + 7*M -- seems bigger than needed. */
            /* BDSPAN = 5*M */          /* lapack was MAX(7*M + 4, 3*M + 2 + SMLSIZ*(SMLSIZ + 8)) -- seems bigger than needed */
            if (n >= mnthr1) {
                if (wantqn) {
                    /* Path 1t (N much larger than M, JOBZ='N') */
                    maxwrk =                m +   m * nb;   //magma_ilaenv( 1, "CGELQF", " ",   m, n, -1, -1 );
                    maxwrk = max( maxwrk, 2*m + 2*m * nb ); // cgebrd
                    minwrk = maxwrk;  // lapack was: 3*m
                }
                else if (wantqo) {
                    /* Path 2t (N much larger than M, JOBZ='O') */
                    wrkbl  =               m +   m * nb;   //magma_ilaenv( 1, "CGELQF", " ",   m, n, -1, -1 );
                    wrkbl  = max( wrkbl,   m +   m * nb ); //magma_ilaenv( 1, "CUNGLQ", " ",   m, n,  m, -1 ));
                    wrkbl  = max( wrkbl, 2*m + 2*m * nb ); // cgebrd
                    wrkbl  = max( wrkbl, 2*m +   m * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", m, m,  m, -1 ));
                    wrkbl  = max( wrkbl, 2*m +   m * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", m, m,  m, -1 ));
                    maxwrk = m*n + m*m + wrkbl;
                    minwrk = 2*m*m     + wrkbl;  // lapack was: 2*m*m + 3*m
                }
                else if (wantqs) {
                    /* Path 3t (N much larger than M, JOBZ='S') */
                    wrkbl  =               m +   m * nb;   //magma_ilaenv( 1, "CGELQF", " ",   m, n, -1, -1 );
                    wrkbl  = max( wrkbl,   m +   m * nb ); //magma_ilaenv( 1, "CUNGLQ", " ",   m, n,  m, -1 ));
                    wrkbl  = max( wrkbl, 2*m + 2*m * nb ); // cgebrd
                    wrkbl  = max( wrkbl, 2*m +   m * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", m, m,  m, -1 ));
                    wrkbl  = max( wrkbl, 2*m +   m * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", m, m,  m, -1 ));
                    maxwrk = m*m + wrkbl;
                    minwrk = maxwrk;  // lapack was: m*m + 3*m
                }
                else if (wantqa) {
                    /* Path 4t (N much larger than M, JOBZ='A') */
                    wrkbl  =               m +   m * nb;   //magma_ilaenv( 1, "CGELQF", " ",   m, n, -1, -1 );
                    wrkbl  = max( wrkbl,   m +   n );      // min for cungqr; preferred is below
                    wrkbl  = max( wrkbl, 2*m + 2*m * nb ); // cgebrd
                    wrkbl  = max( wrkbl, 2*m +   m * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", m, m,  m, -1 ));
                    wrkbl  = max( wrkbl, 2*m +   m * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", m, m,  m, -1 ));
                    minwrk = m*m + wrkbl;  // lapack was: m*m + 2*m + n
                    // include preferred size for cungqr
                    wrkbl  = max( wrkbl,   m +   n * nb ); //magma_ilaenv( 1, "CUNGLQ", " ",   n, n,  m, -1 ));
                    maxwrk = m*m + wrkbl;
                }
            }
            else if (n >= mnthr2) {
                /* Path 5t (N much larger than M, but not as much as MNTHR1) */
                maxwrk = 2*m + (m + n) * nb;  // cgebrd
                minwrk = maxwrk;  // lapack was: 2*m + n
                if (wantqo) {
                    maxwrk = max( maxwrk, 2*m + m * nb );  //magma_ilaenv( 1, "CUNGBR", "P", m, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * nb );  //magma_ilaenv( 1, "CUNGBR", "Q", m, m, n, -1 ));
                    maxwrk = max( maxwrk, 2*m + m*m + n ); // lapack was: maxwrk += m*n;  // todo no m*n?  // extra +n for lapack compatability; not needed
                    minwrk = maxwrk;                       // lapack was: minwrk += m*m;
                }
                else if (wantqs) {
                    maxwrk = max( maxwrk, 2*m + m * nb ); //magma_ilaenv( 1, "CUNGBR", "P", m, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * nb ); //magma_ilaenv( 1, "CUNGBR", "Q", m, m, n, -1 ));
                }
                else if (wantqa) {
                    maxwrk = max( maxwrk, 2*m + n * nb ); //magma_ilaenv( 1, "CUNGBR", "P", n, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * nb ); //magma_ilaenv( 1, "CUNGBR", "Q", m, m, n, -1 ));
                }
            }
            else {
                /* Path 6t (N greater than M, but not much larger) */
                maxwrk = 2*m + (m + n) * nb;  // cgebrd
                minwrk = maxwrk;  // lapack was: 2*m + n
                if (wantqo) {
                    maxwrk = max( maxwrk, 2*m + m*m + m * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", m, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m*n + m * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", m, m, n, -1 ));
                    // lapack was: maxwrk += m*n; m*m and m*n put into unmbr MAX above
                    minwrk = max( minwrk, 2*m + m*m );  // lapack was: minwrk += m*m;
                }
                else if (wantqs) {
                    maxwrk = max( maxwrk, 2*m + m * nb ); //magma_ilaenv( 1, "CUNMBR", "PRC", m, n, m, -1 ));  // lapack was GBR
                    maxwrk = max( maxwrk, 2*m + m * nb ); //magma_ilaenv( 1, "CUNMBR", "QLN", m, m, n, -1 ));  // lapack was GBR
                }
                else if (wantqa) {
                    maxwrk = max( maxwrk, 2*m + n * nb ); //magma_ilaenv( 1, "CUNGBR", "PRC", n, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * nb ); //magma_ilaenv( 1, "CUNGBR", "QLN", m, m, n, -1 ));
                }
            }
        }
        maxwrk = max(maxwrk, minwrk);
    }
    if (*info == 0) {
        work[1] = MAGMA_C_MAKE( maxwrk, 0 );
        if (lwork < minwrk && ! lquery) {
            *info = -13;
        }
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return *info;
    }

    /* Get machine constants */
    eps = lapackf77_slamch("P");
    smlnum = sqrt(lapackf77_slamch("S")) / eps;
    bignum = 1. / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = lapackf77_clange("M", &m, &n, A(1,1), &lda, dum);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
        iscl = 1;
        lapackf77_clascl("G", &izero, &izero, &anrm, &smlnum, &m, &n, A(1,1), &lda, &ierr);
    }
    else if (anrm > bignum) {
        iscl = 1;
        lapackf77_clascl("G", &izero, &izero, &anrm, &bignum, &m, &n, A(1,1), &lda, &ierr);
    }

    if (m >= n) {
        /* A has at least as many rows as columns. */
        /* If A has sufficiently more rows than columns, first reduce using */
        /* the QR decomposition (if sufficient workspace available) */
        if (m >= mnthr1) {
            if (wantqn) {
                /* Path 1 (M much larger than N, JOBZ='N') */
                /* No singular vectors to be computed */
                itau = 1;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (CWorkspace: need 2*N, prefer N + N*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Zero out below R */
                lapackf77_claset("L", &n_1, &n_1, &c_zero, &c_zero, A(2,1), &lda);
                ie = 1;
                itauq = 1;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in A */
                /* (LAPACK CWorkspace: need 3*N, prefer 2*N + 2*N*NB) */
                /* (MAGMA  CWorkspace: need 2*N + 2*N*NB) */
                /* (RWorkspace: need N) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cgebrd(&n, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_cgebrd(n, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif
                nrwork = ie + n;

                /* Perform bidiagonal SVD, compute singular values only */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAN) */
                lapackf77_sbdsdc("U", "N", &n, s, &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 2 (M much larger than N, JOBZ='O') */
                /* N left  singular vectors to be overwritten on A and */
                /* N right singular vectors to be computed in VT */
                iu = 1;

                /* WORK[IU] is N by N */
                ldwrku = n;
                ir = iu + ldwrku*n;
                if (lwork >= m*n + n*n + 3*n) {
                    /* WORK[IR] is M by N */
                    /* replace one N*N with M*N in comments denoted ## below */
                    ldwrkr = m;
                }
                else {
                    ldwrkr = n;  //(lwork - n*n - 3*n) / n;
                }
                itau = ir + ldwrkr*n;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (CWorkspace: need [N*N] + N*N + 2*N, prefer [N*N] + N*N + N + N*NB) ## */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Copy R to WORK[ IR ], zeroing out below it */
                lapackf77_clacpy("U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr);
                lapackf77_claset("L", &n_1, &n_1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr);

                /* Generate Q in A */
                /* (CWorkspace: need [2*N*N] + 2*N, prefer [2*N*N] + N + N*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cungqr(&m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                ie = 1;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in WORK[IR] */
                /* (LAPACK CWorkspace: need [N*N] + N*N + 3*N,          prefer [N*N] + N*N + 2*N + 2*N*NB) ## */
                /* (MAGMA  CWorkspace: need [N*N] + N*N + 2*N + 2*N*NB) ## */
                /* (RWorkspace: need N) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cgebrd(&n, &n, &work[ir], &ldwrkr, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_cgebrd(n, n, &work[ir], ldwrkr, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif
                
                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru    = ie   + n;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_sbdsdc("U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix WORK[IU] */
                /* Overwrite WORK[IU] by the left singular vectors of R */
                /* (CWorkspace: need 2*N*N + 3*N, prefer 2*N*N + 2*N + N*NB) ## */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &n, &n, &rwork[iru], &n, &work[iu], &ldwrku);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, n, n, n, &work[ir], ldwrkr, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by the right singular vectors of R */
                /* (CWorkspace: need [N*N] + N*N + 3*N, prefer [N*N] + N*N + 2*N + N*NB) ## */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &n, &n, &rwork[irvt], &n, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, n, n, n, &work[ir], ldwrkr, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply Q in A by left singular vectors of R in WORK[IU], */
                /* storing result in WORK[IR] and copying to A */
                /* (CWorkspace: need 2*N*N, prefer N*N + M*N) */
                /* (RWorkspace: need 0) */
                for (i = 1; i <= m; i += ldwrkr) {
                    chunk = min(m - i + 1, ldwrkr);
                    blasf77_cgemm("N", "N", &chunk, &n, &n, &c_one, A(i,1), &lda, &work[iu], &ldwrku, &c_zero, &work[ir], &ldwrkr);
                    lapackf77_clacpy("F", &chunk, &n, &work[ir], &ldwrkr, A(i,1), &lda);
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
                /* (CWorkspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Copy R to WORK[IR], zeroing out below it */
                lapackf77_clacpy("U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr);
                lapackf77_claset("L", &n_1, &n_1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr);

                /* Generate Q in A */
                /* (CWorkspace: need [N*N] + 2*N, prefer [N*N] + N + N*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cungqr(&m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                ie = 1;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in WORK[IR] */
                /* (LAPACK CWorkspace: need N*N + 3*N, prefer N*N + 2*N + 2*N*NB) */
                /* (MAGMA  CWorkspace: need N*N + 2*N + 2*N*NB) */
                /* (RWorkspace: need N) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cgebrd(&n, &n, &work[ir], &ldwrkr, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_cgebrd(n, n, &work[ir], ldwrkr, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru    = ie   + n;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_sbdsdc("U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of R */
                /* (CWorkspace: need N*N + 3*N, prefer N*N + 2*N + N*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &n, &n, &rwork[iru], &n, U, &ldu);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, n, n, n, &work[ir], ldwrkr, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of R */
                /* (CWorkspace: need N*N + 3*N, prefer N*N + 2*N + N*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &n, &n, &rwork[irvt], &n, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, n, n, n, &work[ir], ldwrkr, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply Q in A by left singular vectors of R in WORK[IR], */
                /* storing result in U */
                /* (CWorkspace: need N*N) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("F", &n, &n, U, &ldu, &work[ir], &ldwrkr);
                blasf77_cgemm("N", "N", &m, &n, &n, &c_one, A(1,1), &lda, &work[ir], &ldwrkr, &c_zero, U, &ldu);
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
                /* (CWorkspace: need [N*N] + 2*N, prefer [N*N] + N + N*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                lapackf77_clacpy("L", &m, &n, A(1,1), &lda, U, &ldu);

                /* Generate Q in U */
                /* (CWorkspace: need [N*N] + N + M, prefer [N*N] + N + M*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cungqr(&m, &m, &n, U, &ldu, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Produce R in A, zeroing out below it */
                lapackf77_claset("L", &n_1, &n_1, &c_zero, &c_zero, A(2,1), &lda);
                ie = 1;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in A */
                /* (LAPACK CWorkspace: need [N*N] + 3*N, prefer [N*N] + 2*N + 2*N*NB) */
                /* (MAGMA  CWorkspace: need [N*N] + 2*N + 2*N*NB) */
                /* (RWorkspace: need N) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cgebrd(&n, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_cgebrd(n, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif
                iru    = ie   + n;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                lapackf77_sbdsdc("U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix WORK[IU] */
                /* Overwrite WORK[IU] by left singular vectors of R */
                /* (CWorkspace: need N*N + 3*N, prefer N*N + 2*N + N*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &n, &n, &rwork[iru], &n, &work[iu], &ldwrku);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &n, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, n, n, n, A(1,1), lda, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of R */
                /* (CWorkspace: need [N*N] + 3*N, prefer [N*N] + 2*N + N*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &n, &n, &rwork[irvt], &n, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply Q in U by left singular vectors of R in WORK[IU], */
                /* storing result in A */
                /* (CWorkspace: need N*N) */
                /* (RWorkspace: need 0) */
                blasf77_cgemm("N", "N", &m, &n, &n, &c_one, U, &ldu, &work[iu], &ldwrku, &c_zero, A(1,1), &lda);

                /* Copy left singular vectors of A from A to U */
                lapackf77_clacpy("F", &m, &n, A(1,1), &lda, U, &ldu);
            }
        }
        else if (m >= mnthr2) {
            /* MNTHR2 <= M < MNTHR1 */
            /* Path 5 (M much larger than N, but not as much as MNTHR1) */
            /* Reduce to bidiagonal form without QR decomposition, use */
            /* CUNGBR and matrix multiplication to compute singular vectors */
            ie = 1;
            nrwork = ie + n;
            itauq = 1;
            itaup = itauq + n;
            nwork = itaup + n;

            /* Bidiagonalize A */
            /* (LAPACK CWorkspace: need 2*N + M, prefer 2*N + (M + N)*NB) */
            /* (MAGMA  CWorkspace: need 2*N + (M + N)*NB) */
            /* (RWorkspace: need N) */
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_cgebrd(&m, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
            #else
            magma_cgebrd(m, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
            #endif
            
            if (wantqn) {
                /* Path 5n (M > N, JOBZ=N) */
                /* Compute singular values only */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAN) */
                lapackf77_sbdsdc("U", "N", &n, s, &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 5o (M > N, JOBZ=O) */
                iu     = nwork;
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;

                /* Copy A to VT, generate P**H */
                /* (CWorkspace: need [N] + 2*N, prefer [N] + N + N*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("U", &n, &n, A(1,1), &lda, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr);

                /* Generate Q in A */
                /* (CWorkspace: need [N] + 2*N, prefer [N] + N + N*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("Q", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[nwork], &lnwork, &ierr);

                if (lwork >= m*n + 3*n) {
                    /* WORK[ IU ] is M by N */
                    ldwrku = m;
                }
                else {
                    /* WORK[IU] is LDWRKU by N */
                    ldwrku = n;  //(lwork - 3*n) / n;
                }
                nwork = iu + ldwrku*n;

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                lapackf77_sbdsdc("U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], iwork, info);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in WORK[IU], copying to VT */
                /* (was:        need 0) */
                /* (CWorkspace: need [2*N] + N*N) */
                /* (RWorkspace: need [N + N*N] + 3*N*N) */
                lapackf77_clarcm(&n, &n, &rwork[irvt], &n, VT, &ldvt, &work[iu], &ldwrku, &rwork[nrwork]);
                lapackf77_clacpy("F", &n, &n, &work[iu], &ldwrku, VT, &ldvt);

                /* Multiply Q in A by real matrix RWORK[IRU], */
                /* storing the result in WORK[IU], copying to A */
                /* (CWorkspace: need [2*N] + N*N, prefer [2*N] + M*N) */
                /* (RWorkspace: need [N] + 3*N*N, prefer [N] + N*N + 2*M*N) < N + 5*N*N since M < 2*N here */
                nrwork = irvt;
                for (i = 1; i <= m; i += ldwrku) {
                    chunk = min(m - i + 1, ldwrku);
                    lapackf77_clacrm(&chunk, &n, A(i,1), &lda, &rwork[iru], &n, &work[iu], &ldwrku, &rwork[nrwork]);
                    lapackf77_clacpy("F", &chunk, &n, &work[iu], &ldwrku, A(i,1), &lda);
                }
            }
            else if (wantqs) {
                /* Path 5s (M > N, JOBZ=S) */
                /* Copy A to VT, generate P**H */
                /* (CWorkspace: need [N] + 2*N, prefer [N] + N + N*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("U", &n, &n, A(1,1), &lda, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr);

                /* Copy A to U, generate Q */
                /* (CWorkspace: need [N] + 2*N, prefer [N] + N + N*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("L", &m, &n, A(1,1), &lda, U, &ldu);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("Q", &m, &n, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr);

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_sbdsdc("U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], iwork, info);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in A, copying to VT */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need [N + N*N] + 3*N*N) */
                lapackf77_clarcm(&n, &n, &rwork[irvt], &n, VT, &ldvt, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_clacpy("F", &n, &n, A(1,1), &lda, VT, &ldvt);

                /* Multiply Q in U by real matrix RWORK[IRU], */
                /* storing the result in A, copying to U */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need [N] + N*N + 2*M*N) < N + 5*N*N since M < 2*N here */
                nrwork = irvt;
                lapackf77_clacrm(&m, &n, U, &ldu, &rwork[iru], &n, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_clacpy("F", &m, &n, A(1,1), &lda, U, &ldu);
            }
            else {
                /* Path 5a (M > N, JOBZ=A) */
                /* Copy A to VT, generate P**H */
                /* (CWorkspace: need [N] + 2*N, prefer [N] + N + N*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("U", &n, &n, A(1,1), &lda, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("P", &n, &n, &n, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr);

                /* Copy A to U, generate Q */
                /* (CWorkspace: need [N] + 2*N, prefer [N] + N + N*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("L", &m, &n, A(1,1), &lda, U, &ldu);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("Q", &m, &m, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr);

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_sbdsdc("U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], iwork, info);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in A, copying to VT */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need [N + N*N] + 3*N*N) */
                lapackf77_clarcm(&n, &n, &rwork[irvt], &n, VT, &ldvt, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_clacpy("F", &n, &n, A(1,1), &lda, VT, &ldvt);

                /* Multiply Q in U by real matrix RWORK[IRU], */
                /* storing the result in A, copying to U */
                /* (CWorkspace: need 0) */
                /* (was:        need [N] + 3*N*N) */
                /* (RWorkspace: need [N] + N*N + 2*M*N) < 5*N*N since M < 2*N here */
                nrwork = irvt;
                lapackf77_clacrm(&m, &n, U, &ldu, &rwork[iru], &n, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_clacpy("F", &m, &n, A(1,1), &lda, U, &ldu);
            }
        }
        else {
            /* M < MNTHR2 */
            /* Path 6 (M at least N, but not much larger) */
            /* Reduce to bidiagonal form without QR decomposition */
            /* Use CUNMBR to compute singular vectors */
            ie = 1;
            nrwork = ie + n;
            itauq = 1;
            itaup = itauq + n;
            nwork = itaup + n;

            /* Bidiagonalize A */
            /* (LAPACK CWorkspace: need 2*N + M, prefer 2*N + (M + N)*NB) */
            /* (MAGMA  CWorkspace: need 2*N + (M + N)*NB) */
            /* (RWorkspace: need N) */
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_cgebrd(&m, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
            #else
            magma_cgebrd(m, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
            #endif
            
            if (wantqn) {
                /* Path 6n (M >= N, JOBZ=N) */
                /* Compute singular values only */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAN) */
                lapackf77_sbdsdc("U", "N", &n, s, &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 6o (M >= N, JOBZ=O) */
                iu     = nwork;
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                if (lwork >= m*n + 3*n) {
                    /* WORK[ IU ] is M by N */
                    ldwrku = m;
                }
                else {
                    /* WORK[ IU ] is LDWRKU by N */
                    ldwrku = n;  //(lwork - 3*n) / n;
                }
                nwork = iu + ldwrku*n;

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                lapackf77_sbdsdc("U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (CWorkspace: need [N + N*N] + 2*N, prefer [N + N*N] + N + N*NB) */
                /* (was:        need 0) */
                /* (RWorkspace: need [N + N*N] + N*N) */
                lapackf77_clacp2("F", &n, &n, &rwork[irvt], &n, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif

                if (lwork >= m*n + 3*n) {
                    /* Copy real matrix RWORK[IRU] to complex matrix WORK[IU] */
                    /* Overwrite WORK[IU] by left singular vectors of A, copying */
                    /* to A */
                    /* (CWorkspace: need [N] + M*N + 2*N, prefer [N] + M*N + N + N*NB) */
                    /* (was:        need 0) */
                    /* (RWorkspace: need [N] + N*N) */
                    lapackf77_claset("F", &m, &n, &c_zero, &c_zero, &work[iu], &ldwrku);
                    lapackf77_clacp2("F", &n, &n, &rwork[iru], &n, &work[iu], &ldwrku);
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_cunmbr("Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &lnwork, &ierr);
                    #else
                    magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, n, n, A(1,1), lda, &work[itauq], &work[iu], ldwrku, &work[nwork], lnwork, &ierr);
                    #endif
                    lapackf77_clacpy("F", &m, &n, &work[iu], &ldwrku, A(1,1), &lda);
                }
                else {
                    /* Generate Q in A */
                    /* (CWorkspace: need [N + N*N] + 2*N, prefer [N + N*N] + N + N*NB) */
                    /* (RWorkspace: need 0) */
                    lnwork = lwork - nwork + 1;
                    lapackf77_cungbr("Q", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[nwork], &lnwork, &ierr);

                    /* Multiply Q in A by real matrix RWORK[IRU], */
                    /* storing the result in WORK[IU], copying to A */
                    /* (CWorkspace: need [2*N] + N*N, prefer [2*N] + M*N) */
                    /* (RWorkspace: need [N] + 3*N*N, prefer [N] + N*N + 2*M*N) < 5*N*N since M < 2*N here */
                    nrwork = irvt;
                    for (i = 1; i <= m; i += ldwrku) {
                        chunk = min(m - i + 1, ldwrku);
                        lapackf77_clacrm(&chunk, &n, A(i,1), &lda, &rwork[iru], &n, &work[iu], &ldwrku, &rwork[nrwork]);
                        lapackf77_clacpy("F", &chunk, &n, &work[iu], &ldwrku, A(i,1), &lda);
                    }
                }
            }
            else if (wantqs) {
                /* Path 6s (M >= N, JOBZ=S) */
                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_sbdsdc("U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (CWorkspace: need 3*N, prefer 2*N + N*NB) */
                /* (RWorkspace: need [N + N*N] + N*N) */
                lapackf77_claset("F", &m, &n, &c_zero, &c_zero, U, &ldu);
                lapackf77_clacp2("F", &n, &n, &rwork[iru], &n, U, &ldu);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, n, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (CWorkspace: need 3*N, prefer 2*N + N*NB) */
                /* (RWorkspace: need [N + N*N] + N*N) */
                lapackf77_clacp2("F", &n, &n, &rwork[irvt], &n, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif
            }
            else {
                /* Path 6a (M >= N, JOBZ=A) */
                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru    = nrwork;
                irvt   = iru  + n*n;
                nrwork = irvt + n*n;
                lapackf77_sbdsdc("U", "I", &n, s, &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], iwork, info);

                /* Set the right corner of U to identity matrix */
                lapackf77_claset("F", &m, &m, &c_zero, &c_zero, U, &ldu);
                if (m > n) {
                    i__1 = m - n;
                    lapackf77_claset("F", &i__1, &i__1, &c_zero, &c_one, U(n,n), &ldu);
                }

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (CWorkspace: need [N] + N + M, prefer [N] + N + M*NB) */
                /* (was:        need 0 */
                /* (RWorkspace: need [N] + N*N) */
                lapackf77_clacp2("F", &n, &n, &rwork[iru], &n, U, &ldu);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (CWorkspace: need 3*N, prefer 2*N + N*NB) */
                /* (was:        need 0) */
                /* (RWorkspace: need [N + N*N] + N*N) */
                lapackf77_clacp2("F", &n, &n, &rwork[irvt], &n, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, n, n, n, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif
            }
        }
    }
    else {
        /* A has more columns than rows. */
        /* If A has sufficiently more columns than rows, first reduce using */
        /* the LQ decomposition (if sufficient workspace available) */
        if (n >= mnthr1) {
            if (wantqn) {
                /* Path 1t (N much larger than M, JOBZ='N') */
                /* No singular vectors to be computed */
                itau = 1;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (CWorkspace: need 2*M, prefer M + M*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Zero out above L */
                lapackf77_claset("U", &m_1, &m_1, &c_zero, &c_zero, A(1,2), &lda);
                ie = 1;
                itauq = 1;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in A */
                /* (LAPACK CWorkspace: need 3*M, prefer 2*M + 2*M*NB) */
                /* (MAGMA  CWorkspace: need 2*M + 2*M*NB) */
                /* (RWorkspace: need M) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cgebrd(&m, &m, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_cgebrd(m, m, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif
                nrwork = ie + m;

                /* Perform bidiagonal SVD, compute singular values only */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAN) */
                lapackf77_sbdsdc("U", "N", &m, s, &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 2t (N much larger than M, JOBZ='O') */
                /* M right singular vectors to be overwritten on A and */
                /* M left  singular vectors to be computed in U */
                ivt = 1;

                /* WORK[IVT] is M by M */
                ldwkvt = m;
                il = ivt + ldwkvt*m;
                if (lwork >= m*n + m*m + 3*m) {
                    /* WORK[IL] is M by N */
                    /* replace one M*M with M*N in comments denoted ## above */
                    ldwrkl = m;
                    chunk = n;
                }
                else {
                    /* WORK[IL] is M by CHUNK */
                    ldwrkl = m;
                    chunk = m;  //(lwork - m*m - 3*m) / m;
                }
                itau = il + ldwrkl*chunk;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (was:        need [M*M] + 2*M, prefer M + M*NB) */
                /* (CWorkspace: need [M*M] + M*M + 2*M, prefer [M*M] + M*M + M + M*NB) ## */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Copy L to WORK[IL], zeroing out above it */
                lapackf77_clacpy("L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl);
                lapackf77_claset("U", &m_1, &m_1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl);

                /* Generate Q in A */
                /* (CWorkspace: need [2*M*M] + 2*M, prefer [2*M*M] + M + M*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cunglq(&m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                ie = 1;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in WORK[IL] */
                /* (LAPACK CWorkspace: need [M*M] + M*M + 3*M, prefer [M*M] + M*M + 2*M + 2*M*NB) ## */
                /* (MAGMA  CWorkspace: need [M*M] + M*M + 2*M + 2*M*NB) ## */
                /* (RWorkspace: need M) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cgebrd(&m, &m, &work[il], &ldwrkl, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_cgebrd(m, m, &work[il], ldwrkl, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru    = ie   + m;
                irvt   = iru  + m*m;
                nrwork = irvt + m*m;
                lapackf77_sbdsdc("U", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix WORK[IU] */
                /* Overwrite WORK[IU] by the left singular vectors of L */
                /* (was:        need   N*N + 3*N, prefer   N*N + 2*N + N*NB) */
                /* (CWorkspace: need 2*M*M + 3*M, prefer 2*M*M + 2*M + M*NB) ## */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &m, &m, &rwork[iru], &m, U, &ldu);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, m, m, &work[il], ldwrkl, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy real matrix RWORK[IRVT] to complex matrix WORK[IVT] */
                /* Overwrite WORK[IVT] by the right singular vectors of L */
                /* (was:        need [M*M] + N*N + 3*N, prefer [M*M] + M*N + 2*N + N*NB) */
                /* (CWorkspace: need [M*M] + M*M + 3*M, prefer [M*M] + M*M + 2*M + M*NB) ## */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &m, &m, &rwork[irvt], &m, &work[ivt], &ldwkvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, m, m, m, &work[il], ldwrkl, &work[itaup], &work[ivt], ldwkvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply right singular vectors of L in WORK[IL] by Q in A, */
                /* storing result in WORK[IL] and copying to A */
                /* (CWorkspace: need 2*M*M, prefer M*M + M*N) */
                /* (RWorkspace: need 0) */
                for (i = 1; i <= n; i += chunk) {
                    blk = min(n - i + 1, chunk);
                    blasf77_cgemm("N", "N", &m, &blk, &m, &c_one, &work[ivt], &m, A(1,i), &lda, &c_zero, &work[il], &ldwrkl);
                    lapackf77_clacpy("F", &m, &blk, &work[il], &ldwrkl, A(1,i), &lda);
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
                /* (CWorkspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Copy L to WORK[IL], zeroing out above it */
                lapackf77_clacpy("L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl);
                lapackf77_claset("U", &m_1, &m_1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl);

                /* Generate Q in A */
                /* (CWorkspace: need [M*M] + 2*M, prefer [M*M] + M + M*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cunglq(&m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                ie = 1;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in WORK[IL] */
                /* (LAPACK CWorkspace: need M*M + 3*M, prefer M*M + 2*M + 2*M*NB) */
                /* (MAGMA  CWorkspace: need M*M + 2*M + 2*M*NB) */
                /* (RWorkspace: need M) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cgebrd(&m, &m, &work[il], &ldwrkl, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_cgebrd(m, m, &work[il], ldwrkl, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru    = ie   + m;
                irvt   = iru  + m*m;
                nrwork = irvt + m*m;
                lapackf77_sbdsdc("U", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of L */
                /* (CWorkspace: need M*M + 3*M, prefer M*M + 2*M + M*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &m, &m, &rwork[iru], &m, U, &ldu);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, m, m, &work[il], ldwrkl, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by left singular vectors of L */
                /* (CWorkspace: need M*M + 3*M, prefer M*M + 2*M + M*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &m, &m, &rwork[irvt], &m, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, m, m, m, &work[il], ldwrkl, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy VT to WORK[IL], multiply right singular vectors of L */
                /* in WORK[IL] by Q in A, storing result in VT */
                /* (CWorkspace: need M*M) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("F", &m, &m, VT, &ldvt, &work[il], &ldwrkl);
                blasf77_cgemm("N", "N", &m, &n, &m, &c_one, &work[il], &ldwrkl, A(1,1), &lda, &c_zero, VT, &ldvt);
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
                /* (CWorkspace: need [M*M] + 2*M, prefer [M*M] + M + M*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &lnwork, &ierr);
                lapackf77_clacpy("U", &m, &n, A(1,1), &lda, VT, &ldvt);

                /* Generate Q in VT */
                /* (CWorkspace: need [M*M] + M + N, prefer [M*M] + M + N*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cunglq(&n, &n, &m, VT, &ldvt, &work[itau], &work[nwork], &lnwork, &ierr);

                /* Produce L in A, zeroing out above it */
                lapackf77_claset("U", &m_1, &m_1, &c_zero, &c_zero, A(1,2), &lda);
                ie = 1;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in A */
                /* (LAPACK CWorkspace: need [M*M] + 3*M, prefer [M*M] + 2*M + 2*M*NB) */
                /* (MAGMA  CWorkspace: need [M*M] + 2*M + 2*M*NB) */
                /* (RWorkspace: need M) */
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cgebrd(&m, &m, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
                #else
                magma_cgebrd(m, m, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
                #endif
                iru    = ie   + m;
                irvt   = iru  + m*m;
                nrwork = irvt + m*m;

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                lapackf77_sbdsdc("U", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of L */
                /* (CWorkspace: need [M*M] + 3*M, prefer [M*M] + 2*M + M*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &m, &m, &rwork[iru], &m, U, &ldu);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &m, &m, &m, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, m, m, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy real matrix RWORK[IRVT] to complex matrix WORK[IVT] */
                /* Overwrite WORK[IVT] by right singular vectors of L */
                /* (CWorkspace: need M*M + 3*M, prefer M*M + 2*M + M*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacp2("F", &m, &m, &rwork[irvt], &m, &work[ivt], &ldwkvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &m, &m, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, m, m, m, A(1,1), lda, &work[itaup], &work[ivt], ldwkvt, &work[nwork], lnwork, &ierr);
                #endif

                /* Multiply right singular vectors of L in WORK[IVT] by Q in VT, */
                /* storing result in A */
                /* (CWorkspace: need M*M) */
                /* (RWorkspace: need 0) */
                blasf77_cgemm("N", "N", &m, &n, &m, &c_one, &work[ivt], &ldwkvt, VT, &ldvt, &c_zero, A(1,1), &lda);

                /* Copy right singular vectors of A from A to VT */
                lapackf77_clacpy("F", &m, &n, A(1,1), &lda, VT, &ldvt);
            }
        }
        else if (n >= mnthr2) {
            /* MNTHR2 <= N < MNTHR1 */
            /* Path 5t (N much larger than M, but not as much as MNTHR1) */
            /* Reduce to bidiagonal form without LQ decomposition, use */
            /* CUNGBR and matrix multiplication to compute singular vectors */
            ie = 1;
            nrwork = ie + m;
            itauq = 1;
            itaup = itauq + m;
            nwork = itaup + m;

            /* Bidiagonalize A */
            /* (LAPACK CWorkspace: need 2*M + N, prefer 2*M + (M + N)*NB) */
            /* (MAGMA  CWorkspace: need 2*M + (M + N)*NB) */
            /* (RWorkspace: need M) */
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_cgebrd(&m, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
            #else
            magma_cgebrd(m, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
            #endif

            if (wantqn) {
                /* Path 5tn (N > M, JOBZ=N) */
                /* Compute singular values only */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAN) */
                lapackf77_sbdsdc("L", "N", &m, s, &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 5to (N > M, JOBZ=O) */
                ivt    = nwork;
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;

                /* Copy A to U, generate Q */
                /* (CWorkspace: need [M] + 2*M, prefer [M] + M + M*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("L", &m, &m, A(1,1), &lda, U, &ldu);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("Q", &m, &m, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr);

                /* Generate P**H in A */
                /* (CWorkspace: need [M] + 2*M, prefer [M] + M + M*NB) */
                /* (RWorkspace: need 0) */
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("P", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[nwork], &lnwork, &ierr);

                ldwkvt = m;
                if (lwork >= m*n + 3*m) {
                    /* WORK[ IVT ] is M by N */
                    nwork = ivt + ldwkvt*n;
                    chunk = n;
                }
                else {
                    /* WORK[ IVT ] is M by CHUNK */
                    chunk = m;  //(lwork - 3*m) / m;
                    nwork = ivt + ldwkvt*chunk;
                }

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                lapackf77_sbdsdc("L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], iwork, info);

                /* Multiply Q in U by real matrix RWORK[IRVT] */
                /* storing the result in WORK[IVT], copying to U */
                /* (was:        need 0) */
                /* (CWorkspace: need [2*M] + M*M) */
                /* (was:        need 2*M*M) */
                /* (RWorkspace: need [M + M*M] + 3*M*M) */
                lapackf77_clacrm(&m, &m, U, &ldu, &rwork[iru], &m, &work[ivt], &ldwkvt, &rwork[nrwork]);
                lapackf77_clacpy("F", &m, &m, &work[ivt], &ldwkvt, U, &ldu);

                /* Multiply RWORK[IRVT] by P**H in A, */
                /* storing the result in WORK[IVT], copying to A */
                /* (CWorkspace: need [2*M] + M*M, prefer [2*M] + M*N) */
                /* (was:        need 2*M*M, prefer 2*M*N) */
                /* (RWorkspace: need [M] + 3*M*M, prefer [M] + M*M + 2*M*N) < M + 5*M*M since N < 2*M here */
                nrwork = iru;
                for (i = 1; i <= n; i += chunk) {
                    blk = min(n - i + 1, chunk);
                    lapackf77_clarcm(&m, &blk, &rwork[irvt], &m, A(1,i), &lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
                    lapackf77_clacpy("F", &m, &blk, &work[ivt], &ldwkvt, A(1,i), &lda);
                }
            }
            else if (wantqs) {
                /* Path 5ts (N > M, JOBZ=S) */
                /* Copy A to U, generate Q */
                /* (CWorkspace: need [M] + 2*M, prefer [M] + M + M*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("L", &m, &m, A(1,1), &lda, U, &ldu);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("Q", &m, &m, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr);

                /* Copy A to VT, generate P**H */
                /* (CWorkspace: need [M] + 2*M, prefer [M] + M + M*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("U", &m, &n, A(1,1), &lda, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("P", &m, &n, &m, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr);

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_sbdsdc("L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], iwork, info);

                /* Multiply Q in U by real matrix RWORK[IRU], */
                /* storing the result in A, copying to U */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need [M + M*M] + 3*M*M) */
                lapackf77_clacrm(&m, &m, U, &ldu, &rwork[iru], &m, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_clacpy("F", &m, &m, A(1,1), &lda, U, &ldu);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in A, copying to VT */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need [M] + M*M + 2*M*N) < M + 5*M*M since N < 2*M here */
                nrwork = iru;
                lapackf77_clarcm(&m, &n, &rwork[irvt], &m, VT, &ldvt, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_clacpy("F", &m, &n, A(1,1), &lda, VT, &ldvt);
            }
            else {
                /* Path 5ta (N > M, JOBZ=A) */
                /* Copy A to U, generate Q */
                /* (CWorkspace: need [M] + 2*M, prefer [M] + M + M*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("L", &m, &m, A(1,1), &lda, U, &ldu);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("Q", &m, &m, &n, U, &ldu, &work[itauq], &work[nwork], &lnwork, &ierr);

                /* Copy A to VT, generate P**H */
                /* (CWorkspace: need [M] + 2*M, prefer [M] + M + M*NB) */
                /* (RWorkspace: need 0) */
                lapackf77_clacpy("U", &m, &n, A(1,1), &lda, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                lapackf77_cungbr("P", &n, &n, &m, VT, &ldvt, &work[itaup], &work[nwork], &lnwork, &ierr);

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_sbdsdc("L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], iwork, info);

                /* Multiply Q in U by real matrix RWORK[IRU], */
                /* storing the result in A, copying to U */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need [M + M*M] + 3*M*M) */
                lapackf77_clacrm(&m, &m, U, &ldu, &rwork[iru], &m, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_clacpy("F", &m, &m, A(1,1), &lda, U, &ldu);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in A, copying to VT */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need [M] + M*M + 2*M*N) < [M] + 5*M*M since N < 2*M here */
                // LAPACK doesn't reset nrwork here, so it needs an extra M*M:
                // ([M] + 2*M*M + 2*M*N) < [M] + 6*M*M since N < 2*M here */
                nrwork = iru;
                lapackf77_clarcm(&m, &n, &rwork[irvt], &m, VT, &ldvt, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_clacpy("F", &m, &n, A(1,1), &lda, VT, &ldvt);
            }
        }
        else {
            /* N < MNTHR2 */
            /* Path 6t (N greater than M, but not much larger) */
            /* Reduce to bidiagonal form without LQ decomposition */
            /* Use CUNMBR to compute singular vectors */
            ie = 1;
            nrwork = ie + m;
            itauq = 1;
            itaup = itauq + m;
            nwork = itaup + m;

            /* Bidiagonalize A */
            /* (LAPACK CWorkspace: need 2*M + N, prefer 2*M + (M + N)*NB) */
            /* (MAGMA  CWorkspace: need 2*M + (M + N)*NB) */
            /* (RWorkspace: need M) */
            lnwork = lwork - nwork + 1;
            #if VERSION == 1
            lapackf77_cgebrd(&m, &n, A(1,1), &lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &lnwork, &ierr);
            #else
            magma_cgebrd(m, n, A(1,1), lda, s, &rwork[ie], &work[itauq], &work[itaup], &work[nwork], lnwork, &ierr);
            #endif
            
            if (wantqn) {
                /* Path 6tn (N >= M, JOBZ=N) */
                /* Compute singular values only */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAN) */
                lapackf77_sbdsdc("L", "N", &m, s, &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], iwork, info);
            }
            else if (wantqo) {
                /* Path 6to (N >= M, JOBZ=O) */
                ldwkvt = m;
                ivt = nwork;
                if (lwork >= m*n + 3*m) {
                    /* WORK[ IVT ] is M by N */
                    lapackf77_claset("F", &m, &n, &c_zero, &c_zero, &work[ivt], &ldwkvt);
                    nwork = ivt + ldwkvt*n;
                }
                else {
                    /* WORK[ IVT ] is M by CHUNK */
                    chunk = m;  //(lwork - 3*m) / m;
                    nwork = ivt + ldwkvt*chunk;
                }

                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_sbdsdc("L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (CWorkspace: need [M + M*M] + 2*M, prefer [M + M*M] + M + M*NB) */
                /* (was:        need 0) */
                /* (RWorkspace: need [M + M*M] + M*M) */
                lapackf77_clacp2("F", &m, &m, &rwork[iru], &m, U, &ldu);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                if (lwork >= m*n + 3*m) {
                    /* Copy real matrix RWORK[IRVT] to complex matrix WORK[IVT] */
                    /* Overwrite WORK[IVT] by right singular vectors of A, */
                    /* copying to A */
                    /* (CWorkspace: need [M] + M*N + 2*M, prefer [M] + M*N + M + M*NB) */
                    /* (was:        need 0) */
                    /* (RWorkspace: need [M] + M*M) */
                    lapackf77_clacp2("F", &m, &m, &rwork[irvt], &m, &work[ivt], &ldwkvt);
                    lnwork = lwork - nwork + 1;
                    #if VERSION == 1
                    lapackf77_cunmbr("P", "R", "C", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &lnwork, &ierr);
                    #else
                    magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, m, n, m, A(1,1), lda, &work[itaup], &work[ivt], ldwkvt, &work[nwork], lnwork, &ierr);
                    #endif
                    lapackf77_clacpy("F", &m, &n, &work[ivt], &ldwkvt, A(1,1), &lda);
                }
                else {
                    /* Generate P**H in A */
                    /* (CWorkspace: need [M + M*M] + 2*M, prefer [M + M*M] + M + M*NB) */
                    /* (RWorkspace: need 0) */
                    lnwork = lwork - nwork + 1;
                    lapackf77_cungbr("P", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[nwork], &lnwork, &ierr);

                    /* Multiply Q in A by real matrix RWORK[IRU], */
                    /* storing the result in WORK[IU], copying to A */
                    /* (CWorkspace: need [2*M] + M*M, prefer [2*M] + M*N) */
                    /* (RWorkspace: need [M] + 3*M*M, prefer [M] + M*M + 2*M*N) < 5*M*M since N < 2*M here */
                    nrwork = iru;
                    for (i = 1; i <= n; i += chunk) {
                        blk = min(n - i + 1, chunk);
                        lapackf77_clarcm(&m, &blk, &rwork[irvt], &m, A(1,i), &lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
                        lapackf77_clacpy("F", &m, &blk, &work[ivt], &ldwkvt, A(1,i), &lda);
                    }
                }
            }
            else if (wantqs) {
                /* Path 6ts (N >= M, JOBZ=S) */
                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;
                lapackf77_sbdsdc("L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (CWorkspace: need 3*M, prefer 2*M + M*NB) */
                /* (RWorkspace: need [M + M*M] + M*M) */
                lapackf77_clacp2("F", &m, &m, &rwork[iru], &m, U, &ldu);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (CWorkspace: need 3*M, prefer 2*M + M*NB) */
                /* (RWorkspace: need [M + M*M] + M*M) */
                lapackf77_claset("F", &m, &n, &c_zero, &c_zero, VT, &ldvt);
                lapackf77_clacp2("F", &m, &m, &rwork[irvt], &m, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &m, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, m, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif
            }
            else {
                /* Path 6ta (N >= M, JOBZ=A) */
                /* Perform bidiagonal SVD, */
                /* computing left  singular vectors of bidiagonal matrix in RWORK[IRU] and */
                /* computing right singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt   = nrwork;
                iru    = irvt + m*m;
                nrwork = iru  + m*m;

                lapackf77_sbdsdc("L", "I", &m, s, &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], iwork, info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (CWorkspace: need 3*M, prefer 2*M + M*NB) */
                /* (RWorkspace: need [M] + M*M) */
                lapackf77_clacp2("F", &m, &m, &rwork[iru], &m, U, &ldu);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U, &ldu, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaQ, MagmaLeft, MagmaNoTrans, m, m, n, A(1,1), lda, &work[itauq], U, ldu, &work[nwork], lnwork, &ierr);
                #endif

                /* Set all of VT to identity matrix */
                lapackf77_claset("F", &n, &n, &c_zero, &c_one, VT, &ldvt);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (CWorkspace: need 2*M + N, prefer 2*M + N*NB) */
                /* (RWorkspace: need [M + M*M] + M*M) */
                lapackf77_clacp2("F", &m, &m, &rwork[irvt], &m, VT, &ldvt);
                lnwork = lwork - nwork + 1;
                #if VERSION == 1
                lapackf77_cunmbr("P", "R", "C", &n, &n, &m, A(1,1), &lda, &work[itaup], VT, &ldvt, &work[nwork], &lnwork, &ierr);
                #else
                magma_cunmbr(MagmaP, MagmaRight, MagmaConjTrans, n, n, m, A(1,1), lda, &work[itaup], VT, ldvt, &work[nwork], lnwork, &ierr);
                #endif
            }
        }
    }

    /* Undo scaling if necessary */
    if (iscl == 1) {
        if (anrm > bignum) {
            lapackf77_slascl("G", &izero, &izero, &bignum, &anrm, &minmn, &ione, s, &minmn, &ierr);
        }
        if (*info != 0 && anrm > bignum) {
            magma_int_t minmn_1 = minmn - 1;
            lapackf77_slascl("G", &izero, &izero, &bignum, &anrm, &minmn_1, &ione, &rwork[ie], &minmn, &ierr);
        }
        if (anrm < smlnum) {
            lapackf77_slascl("G", &izero, &izero, &smlnum, &anrm, &minmn, &ione, s, &minmn, &ierr);
        }
        if (*info != 0 && anrm < smlnum) {
            magma_int_t minmn_1 = minmn - 1;
            lapackf77_slascl("G", &izero, &izero, &smlnum, &anrm, &minmn_1, &ione, &rwork[ie], &minmn, &ierr);
        }
    }

    /* Return optimal workspace in WORK[0] */
    work[1] = MAGMA_C_MAKE( maxwrk, 0 );

    return *info;
} /* magma_cgesdd */
