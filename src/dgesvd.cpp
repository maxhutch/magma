/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
    
       @author Stan Tomov
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
    DGESVD computes the singular value decomposition (SVD) of a real
    M-by-N matrix A, optionally computing the left and/or right singular
    vectors. The SVD is written
        
        A = U * SIGMA * conjugate-transpose(V)
    
    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.
    
    Note that the routine returns V**T, not V.
    
    Arguments
    ---------
    @param[in]
    jobu    magma_vec_t
            Specifies options for computing all or part of the matrix U:
      -     = MagmaAllVec:        all M columns of U are returned in array U:
      -     = MagmaSomeVec:       the first min(m,n) columns of U (the left singular
                                  vectors) are returned in the array U;
      -     = MagmaOverwriteVec:  the first min(m,n) columns of U (the left singular
                                  vectors) are overwritten on the array A;
      -     = MagmaNoVec:         no columns of U (no left singular vectors) are
                                  computed.
    
    @param[in]
    jobvt   magma_vec_t
            Specifies options for computing all or part of the matrix V**T:
      -     = MagmaAllVec:        all N rows of V**T are returned in the array VT;
      -     = MagmaSomeVec:       the first min(m,n) rows of V**T (the right singular
                                  vectors) are returned in the array VT;
      -     = MagmaOverwriteVec:  the first min(m,n) rows of V**T (the right singular
                                  vectors) are overwritten on the array A;
      -     = MagmaNoVec:         no rows of V**T (no right singular vectors) are
                                  computed.
    \n
            JOBVT and JOBU cannot both be MagmaOverwriteVec.
    
    @param[in]
    m       INTEGER
            The number of rows of the input matrix A.  M >= 0.
    
    @param[in]
    n       INTEGER
            The number of columns of the input matrix A.  N >= 0.
    
    @param[in,out]
    A       DOUBLE_PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
      -     if JOBU = MagmaOverwriteVec,  A is overwritten with the first min(m,n)
            columns of U (the left singular vectors, stored columnwise);
      -     if JOBVT = MagmaOverwriteVec, A is overwritten with the first min(m,n)
            rows of V**T (the right singular vectors, stored rowwise);
      -     if JOBU != MagmaOverwriteVec and JOBVT != MagmaOverwriteVec,
            the contents of A are destroyed.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    @param[out]
    s       DOUBLE_PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).
    
    @param[out]
    U       DOUBLE_PRECISION array, dimension (LDU,UCOL)
            (LDU,M) if JOBU = MagmaAllVec or (LDU,min(M,N)) if JOBU = MagmaSomeVec.
      -     If JOBU = MagmaAllVec, U contains the M-by-M orthogonal matrix U;
      -     if JOBU = MagmaSomeVec, U contains the first min(m,n) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBU = MagmaNoVec or MagmaOverwriteVec, U is not referenced.
    
    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBU = MagmaSomeVec or MagmaAllVec, LDU >= M.
    
    @param[out]
    VT      DOUBLE_PRECISION array, dimension (LDVT,N)
      -     If JOBVT = MagmaAllVec, VT contains the N-by-N orthogonal matrix
            V**T;
      -     if JOBVT = MagmaSomeVec, VT contains the first min(m,n) rows of
            V**T (the right singular vectors, stored rowwise);
      -     if JOBVT = MagmaNoVec or MagmaOverwriteVec, VT is not referenced.
    
    @param[in]
    ldvt    INTEGER
            The leading dimension of the array VT.  LDVT >= 1;
      -     if JOBVT = MagmaAllVec, LDVT >= N;
      -     if JOBVT = MagmaSomeVec, LDVT >= min(M,N).
    
    @param[out]
    work    (workspace) DOUBLE_PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the required LWORK.
            if INFO > 0, WORK(2:MIN(M,N)) contains the unconverged
            superdiagonal elements of an upper bidiagonal matrix B
            whose diagonal is in S (not necessarily sorted). B
            satisfies A = U * B * VT, so it has the same singular values
            as A, and singular vectors related by U and VT.
            
    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            LWORK >= (M+N)*nb + 3*min(M,N).
            For optimum performance with some paths
            (m >> n and jobu=A,S,O; or n >> m and jobvt=A,S,O),
            LWORK >= (M+N)*nb + 3*min(M,N) + 2*min(M,N)**2 (see comments inside code).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the required size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if DBDSQR did not converge, INFO specifies how many
                superdiagonals of an intermediate bidiagonal form B
                did not converge to zero. See the description of RWORK
                above for details.
    

    @ingroup magma_dgesvd_driver
    ********************************************************************/
extern "C" magma_int_t
magma_dgesvd(
    magma_vec_t jobu, magma_vec_t jobvt, magma_int_t m, magma_int_t n,
    double *A,    magma_int_t lda, double *s,
    double *U,    magma_int_t ldu,
    double *VT,   magma_int_t ldvt,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork,
    #endif
    magma_int_t *info )
{
    #define A(i_,j_)  (A  + (i_) + (j_)*lda)
    #define U(i_,j_)  (U  + (i_) + (j_)*ldu)
    #define VT(i_,j_) (VT + (i_) + (j_)*ldvt)
    
    const char* jobu_  = lapack_vec_const( jobu  );
    const char* jobvt_ = lapack_vec_const( jobvt );
    
    const double c_zero          = MAGMA_D_ZERO;
    const double c_one           = MAGMA_D_ONE;
    const magma_int_t izero      = 0;
    const magma_int_t ione       = 1;
    const magma_int_t ineg_one   = -1;
    
    // System generated locals
    magma_int_t lwork2, m_1, n_1;
    
    // Local variables
    magma_int_t i, ie, ir, iu, blk, ncu;
    double dummy[1], eps;
    double cdummy[1];
    magma_int_t nru, iscl;
    double anrm;
    magma_int_t ierr, itau, ncvt, nrvt;
    magma_int_t chunk, minmn, wrkbrd, wrkbl, itaup, itauq, mnthr, iwork;
    magma_int_t want_ua, want_va, want_un, want_uo, want_vn, want_vo, want_us, want_vs;
    magma_int_t bdspac;
    double bignum;
    magma_int_t ldwrkr, ldwrku, maxwrk, minwrk;
    double smlnum;
    magma_int_t lquery, want_uas, want_vas;
    magma_int_t nb;
    
    // Function Body
    *info = 0;
    minmn = min(m,n);
    mnthr = (magma_int_t)( minmn * 1.6 );
    bdspac = 5*n;
    ie = 0;
    
    want_ua  = (jobu == MagmaAllVec);
    want_us  = (jobu == MagmaSomeVec);
    want_uo  = (jobu == MagmaOverwriteVec);
    want_un  = (jobu == MagmaNoVec);
    want_uas = want_ua || want_us;
    
    want_va  = (jobvt == MagmaAllVec);
    want_vs  = (jobvt == MagmaSomeVec);
    want_vo  = (jobvt == MagmaOverwriteVec);
    want_vn  = (jobvt == MagmaNoVec);
    want_vas = want_va || want_vs;
    
    lquery = (lwork == -1);
    
    // Test the input arguments
    if (! (want_ua || want_us || want_uo || want_un)) {
        *info = -1;
    } else if (! (want_va || want_vs || want_vo || want_vn) || (want_vo && want_uo) ) {
        *info = -2;
    } else if (m < 0) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < max(1,m)) {
        *info = -6;
    } else if ((ldu < 1) || (want_uas && (ldu < m)) ) {
        *info = -9;
    } else if ((ldvt < 1) || (want_va && (ldvt < n)) || (want_vs && (ldvt < minmn)) ) {
        *info = -11;
    }
    
    // Compute workspace
    lapackf77_dgesvd(jobu_, jobvt_, &m, &n, A, &lda, s, U, &ldu, VT, &ldvt,
                     work, &ineg_one, info);
    maxwrk = (magma_int_t) MAGMA_D_REAL( work[0] );
    if (*info == 0) {
        // Return required workspace in WORK[0]
        nb = magma_get_dgesvd_nb(n);
        minwrk = (m + n)*nb + 3*minmn;
        
        // multiply by 1+eps (in Double!) to ensure length gets rounded up,
        // if it cannot be exactly represented in floating point.
        real_Double_t one_eps = 1. + lapackf77_dlamch("Epsilon");
        work[0] = MAGMA_D_MAKE( minwrk * one_eps, 0 );
        if ( !lquery && (lwork < minwrk) ) {
            *info = -13;
        }
    }
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }
    
    // Quick return if possible
    if (m == 0 || n == 0) {
        return *info;
    }
    
    wrkbl = maxwrk; // Not optimal
    wrkbrd = (m + n)*nb + 3*minmn;
    
    // Parameter adjustments for Fortran indexing
    --work;
    
    // Get machine constants
    eps = lapackf77_dlamch("P");
    smlnum = magma_dsqrt(lapackf77_dlamch("S")) / eps;
    bignum = 1. / smlnum;
    
    // Scale A if max element outside range [SMLNUM,BIGNUM]
    anrm = lapackf77_dlange("M", &m, &n, A, &lda, dummy);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
        iscl = 1;
        lapackf77_dlascl("G", &izero, &izero, &anrm, &smlnum, &m, &n,
                         A, &lda, &ierr);
    }
    else if (anrm > bignum) {
        iscl = 1;
        lapackf77_dlascl("G", &izero, &izero, &anrm, &bignum, &m, &n,
                         A, &lda, &ierr);
    }
    
    m_1 = m - 1;
    n_1 = n - 1;
    
    if (m >= n) {
        // A has at least as many rows as columns. If A has sufficiently
        // more rows than columns, first reduce using the QR
        // decomposition (if sufficient workspace available)
        if (m >= mnthr) {
            if (want_un) {
                // Path 1 (M much larger than N, JOBU='N')
                // No left singular vectors to be computed
                // printf("Path 1\n");
                itau = 1;
                iwork = itau + n;
                
                // Compute A=Q*R
                // (Workspace: need 2*N, prefer N + N*NB)
                lwork2 = lwork - iwork + 1;
                lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                 &work[iwork], &lwork2, &ierr);
                
                // Zero out below R
                lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                 A(1,0), &lda);
                ie = 1;
                itauq = ie + n;
                itaup = itauq + n;
                iwork = itaup + n;
                
                // Bidiagonalize R in A
                // (Workspace: need 3*N + (M+N)*NB)  [was: need 4*N, prefer 3*N + 2*N*NB]
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&n, &n, A, &lda, s, &work[ie],
                             &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                #else
                magma_dgebrd(n, n, A, lda, s, &work[ie],
                             &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                #endif
                
                ncvt = 0;
                if (want_vo || want_vas) {
                    // If right singular vectors desired, generate P'.
                    // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("P", &n, &n, &n, A, &lda,
                                     &work[itaup], &work[iwork], &lwork2, &ierr);
                    ncvt = n;
                }
                iwork = ie + n;
                
                // Perform bidiagonal QR iteration, computing right
                // singular vectors of A in A if desired
                // (Workspace: need BDSPAC)
                lapackf77_dbdsqr("U", &n, &ncvt, &izero, &izero, s, &work[ie],
                                 A, &lda, cdummy, &ione,
                                 cdummy, &ione, &work[iwork], info);
                
                // If right singular vectors desired in VT, copy them there
                if (want_vas) {
                    lapackf77_dlacpy("F", &n, &n,
                                     A, &lda,
                                     VT, &ldvt);
                }
            }
            else if (want_uo && want_vn) {
                // Path 2 (M much larger than N, JOBU='O', JOBVT='N')
                // N left singular vectors to be overwritten on A and
                // no right singular vectors to be computed
                // printf("Path 2\n");
                if (lwork >= n*n + max(wrkbrd, bdspac)) {
                    // Sufficient workspace for a fast algorithm
                    ir = 1;
                    if (lwork >= max(wrkbl, lda*n + n) + lda*n) {
                        // WORK(IU) is LDA by N, WORK(IR) is LDA by N
                        ldwrku = lda;
                        ldwrkr = lda;
                    }
                    else /* if (complicated condition) */ {
                        if (lwork >= max(wrkbl, lda*n + n) + n*n) {
                            // WORK(IU) is LDA by N, WORK(IR) is N by N
                            ldwrku = lda;
                            ldwrkr = n;
                        }
                        else {
                            // WORK(IU) is LDWRKU by N, WORK(IR) is N by N
                            ldwrku = (lwork - n*n - n) / n;
                            ldwrkr = n;
                        }
                    }
                    itau = ir + ldwrkr * n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R
                    // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                     &work[iwork], &lwork2, &ierr);
                    
                    // Copy R to WORK(IR) and zero out below it
                    lapackf77_dlacpy("U", &n, &n,
                                     A, &lda,
                                     &work[ir], &ldwrkr);
                    lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                     &work[ir+1], &ldwrkr);
                    
                    // Generate Q in A
                    // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr(&m, &n, &n, A, &lda,
                                     &work[itau], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dorgqr2(m, n, n, A, lda, &work[itau], &ierr);
                    #endif

                    ie = itau;
                    itauq = ie + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in WORK(IR)
                    // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd(&n, &n, &work[ir], &ldwrkr, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dgebrd(n, n, &work[ir], ldwrkr, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                    #endif
                    
                    // Generate left vectors bidiagonalizing R
                    // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("Q", &n, &n, &n, &work[ir], &ldwrkr,
                                     &work[itauq], &work[iwork], &lwork2, &ierr);
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IR)
                    // (Workspace: need N*N + BDSPAC)
                    lapackf77_dbdsqr("U", &n, &izero, &n, &izero, s, &work[ie],
                                     cdummy, &ione, &work[ir], &ldwrkr,
                                     cdummy, &ione, &work[iwork], info);
                    iu = ie + n;
                    
                    // Multiply Q in A by left singular vectors of R in
                    // WORK(IR), storing result in WORK(IU) and copying to A
                    // (Workspace: need N*N + 2*N, prefer N*N + M*N + N)
                    for (i = 1; (ldwrku < 0 ? i >= m : i <= m); i += ldwrku) {
                        chunk = min(m - i + 1,ldwrku);
                        blasf77_dgemm("N", "N", &chunk, &n, &n,
                                      &c_one,  A(i-1,0), &lda,
                                               &work[ir], &ldwrkr,
                                      &c_zero, &work[iu], &ldwrku);
                        lapackf77_dlacpy("F", &chunk, &n,
                                         &work[iu], &ldwrku,
                                         A(i-1,0), &lda);
                    }
                }
                else {
                    // Insufficient workspace for a fast algorithm
                    ie = 1;
                    itauq = ie + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize A
                    // (Workspace: need 3*N + M, prefer 3*N + (M + N)*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd(&m, &n, A, &lda, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dgebrd(m, n, A, lda, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                    #endif
                    
                    // Generate left vectors bidiagonalizing A
                    // (Workspace: need 4*N, prefer 3*N + N*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("Q", &m, &n, &n, A, &lda,
                                     &work[itauq], &work[iwork], &lwork2, &ierr);
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in A
                    // (Workspace: need BDSPAC)
                    lapackf77_dbdsqr("U", &n, &izero, &m, &izero, s, &work[ie],
                                     cdummy, &ione, A, &lda,
                                     cdummy, &ione, &work[iwork], info);
                }
            }
            else if (want_uo && want_vas) {
                // Path 3 (M much larger than N, JOBU='O', JOBVT='S' or 'A')
                // N left singular vectors to be overwritten on A and
                // N right singular vectors to be computed in VT
                // printf("Path 3\n");
                if (lwork >= n*n + max(wrkbrd, bdspac)) {
                    // Sufficient workspace for a fast algorithm
                    ir = 1;
                    if (lwork >= max(wrkbl, lda*n + n) + lda*n) {
                        // WORK(IU) is LDA by N and WORK(IR) is LDA by N
                        ldwrku = lda;
                        ldwrkr = lda;
                    }
                    else /* if (complicated condition) */ {
                        if (lwork >= max(wrkbl, lda*n + n) + n*n) {
                            // WORK(IU) is LDA by N and WORK(IR) is N by N
                            ldwrku = lda;
                            ldwrkr = n;
                        }
                        else {
                            // WORK(IU) is LDWRKU by N and WORK(IR) is N by N
                            ldwrku = (lwork - n*n - n) / n;
                            ldwrkr = n;
                        }
                    }
                    itau = ir + ldwrkr * n;
                    iwork = itau + n;
                    
                    // Compute A=Q*R
                    // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                     &work[iwork], &lwork2, &ierr);
                    
                    // Copy R to VT, zeroing out below it
                    lapackf77_dlacpy("U", &n, &n,
                                     A, &lda,
                                     VT, &ldvt);
                    if (n > 1) {
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         VT(1,0), &ldvt);
                    }
                    
                    // Generate Q in A
                    // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr(&m, &n, &n, A, &lda,
                                     &work[itau], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dorgqr2(m, n, n, A, lda, &work[itau], &ierr);
                    #endif

                    ie = itau;
                    itauq = ie + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in VT, copying result to WORK(IR)
                    // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd(&n, &n, VT, &ldvt, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dgebrd(n, n, VT, ldvt, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                    #endif
                    lapackf77_dlacpy("L", &n, &n,
                                     VT, &ldvt,
                                     &work[ir], &ldwrkr);
                    
                    // Generate left vectors bidiagonalizing R in WORK(IR)
                    // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("Q", &n, &n, &n, &work[ir], &ldwrkr,
                                     &work[itauq], &work[iwork], &lwork2, &ierr);
                    
                    // Generate right vectors bidiagonalizing R in VT
                    // (Workspace: need N*N + 4*N-1, prefer N*N + 3*N + (N-1)*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("P", &n, &n, &n, VT, &ldvt,
                                     &work[itaup], &work[iwork], &lwork2, &ierr);
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of R in WORK(IR) and computing right
                    // singular vectors of R in VT
                    // (Workspace: need N*N + BDSPAC)
                    lapackf77_dbdsqr("U", &n, &n, &n, &izero, s, &work[ie],
                                     VT, &ldvt, &work[ir], &ldwrkr,
                                     cdummy, &ione, &work[iwork], info);
                    iu = ie + n;
                    
                    // Multiply Q in A by left singular vectors of R in
                    // WORK(IR), storing result in WORK(IU) and copying to A
                    // (Workspace: need N*N + 2*N, prefer N*N + M*N + N)
                    for (i = 1; (ldwrku < 0 ? i >= m : i <= m); i += ldwrku) {
                        chunk = min(m - i + 1,ldwrku);
                        blasf77_dgemm("N", "N", &chunk, &n, &n,
                                      &c_one,  A(i-1,0), &lda,
                                               &work[ir], &ldwrkr,
                                      &c_zero, &work[iu], &ldwrku);
                        lapackf77_dlacpy("F", &chunk, &n,
                                         &work[iu], &ldwrku,
                                         A(i-1,0), &lda);
                    }
                }
                else {
                    // Insufficient workspace for a fast algorithm
                    itau = 1;
                    iwork = itau + n;
                    
                    // Compute A=Q*R
                    // (Workspace: need 2*N, prefer N + N*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                     &work[iwork], &lwork2, &ierr);
                    
                    // Copy R to VT, zeroing out below it
                    lapackf77_dlacpy("U", &n, &n,
                                     A, &lda,
                                     VT, &ldvt);
                    if (n > 1) {
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         VT(1,0), &ldvt);
                    }
                    
                    // Generate Q in A
                    // (Workspace: need 2*N, prefer N + N*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dorgqr(&m, &n, &n, A, &lda,
                                     &work[itau], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dorgqr2(m, n, n, A, lda, &work[itau], &ierr);
                    #endif

                    ie = itau;
                    itauq = ie + n;
                    itaup = itauq + n;
                    iwork = itaup + n;
                    
                    // Bidiagonalize R in VT
                    // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd(&n, &n, VT, &ldvt, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dgebrd(n, n, VT, ldvt, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                    #endif
                    
                    // Multiply Q in A by left vectors bidiagonalizing R
                    // (Workspace: need 3*N + M, prefer 3*N + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dormbr("Q", "R", "N", &m, &n, &n,
                                     VT, &ldvt, &work[itauq],
                                     A, &lda, &work[iwork], &lwork2, &ierr);
                    
                    // Generate right vectors bidiagonalizing R in VT
                    // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("P", &n, &n, &n, VT, &ldvt,
                                     &work[itaup], &work[iwork], &lwork2, &ierr);
                    iwork = ie + n;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in A and computing right
                    // singular vectors of A in VT
                    // (Workspace: need BDSPAC)
                    lapackf77_dbdsqr("U", &n, &n, &m, &izero, s, &work[ie],
                                     VT, &ldvt, A, &lda,
                                     cdummy, &ione, &work[iwork], info);
                }
            }
            else if (want_us) {
                if (want_vn) {
                    // Path 4 (M much larger than N, JOBU='S', JOBVT='N')
                    // N left singular vectors to be computed in U and
                    // no right singular vectors to be computed
                    // printf("Path 4\n");
                    if (lwork >= n*n + max(wrkbrd, bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        ir = 1;
                        if (lwork >= wrkbl + lda*n) {
                            // WORK(IR) is LDA by N
                            ldwrkr = lda;
                        }
                        else {
                            // WORK(IR) is N by N
                            ldwrkr = n;
                        }
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;
                        
                        // Compute A=Q*R
                        // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                        
                        // Copy R to WORK(IR), zeroing out below it
                        lapackf77_dlacpy("U", &n, &n,
                                         A, &lda,
                                         &work[ir], &ldwrkr);
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         &work[ir+1], &ldwrkr);
                        
                        // Generate Q in A
                        // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &n, &n, A, &lda,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, n, n, A, lda, &work[itau], &ierr);
                        #endif

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Bidiagonalize R in WORK(IR)
                        // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, &work[ir], &ldwrkr, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, &work[ir], ldwrkr, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Generate left vectors bidiagonalizing R in WORK(IR)
                        // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &n, &n, &n, &work[ir], &ldwrkr,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of R in WORK(IR)
                        // (Workspace: need N*N + BDSPAC)
                        lapackf77_dbdsqr("U", &n, &izero, &n, &izero, s, &work[ie],
                                         cdummy, &ione, &work[ir], &ldwrkr,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply Q in A by left singular vectors of R in
                        // WORK(IR), storing result in U
                        // (Workspace: need N*N)
                        blasf77_dgemm("N", "N", &m, &n, &n,
                                      &c_one,  A, &lda,
                                               &work[ir], &ldwrkr,
                                      &c_zero, U, &ldu);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + n;
                        
                        // Compute A=Q*R, copying result to U
                        // (Workspace: need 2*N, prefer N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("L", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Generate Q in U
                        // (Workspace: need 2*N, prefer N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &n, &n, U, &ldu,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, n, n, U, ldu, &work[itau], &ierr);
                        #endif

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Zero out below R in A
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         A(1,0), &lda);
                        
                        // Bidiagonalize R in A
                        // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, A, &lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, A, lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply Q in U by left vectors bidiagonalizing R
                        // (Workspace: need 3*N + M, prefer 3*N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("Q", "R", "N", &m, &n, &n,
                                         A, &lda, &work[itauq],
                                         U, &ldu, &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in U
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &n, &izero, &m, &izero, s, &work[ie],
                                         cdummy, &ione, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
                else if (want_vo) {
                    // Path 5 (M much larger than N, JOBU='S', JOBVT='O')
                    // N left singular vectors to be computed in U and
                    // N right singular vectors to be overwritten on A
                    // printf("Path 5\n");
                    if (lwork >= 2*n*n + max(wrkbrd, bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        iu = 1;
                        if (lwork >= wrkbl + 2*lda*n) {
                            // WORK(IU) is LDA by N and WORK(IR) is LDA by N
                            ldwrku = lda;
                            ir = iu + ldwrku * n;
                            ldwrkr = lda;
                        }
                        else if (lwork >= wrkbl + (lda + n) * n) {
                            // WORK(IU) is LDA by N and WORK(IR) is N by N
                            ldwrku = lda;
                            ir = iu + ldwrku * n;
                            ldwrkr = n;
                        }
                        else {
                            // WORK(IU) is N by N and WORK(IR) is N by N
                            ldwrku = n;
                            ir = iu + ldwrku * n;
                            ldwrkr = n;
                        }
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;
                        
                        // Compute A=Q*R
                        // (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                        
                        // Copy R to WORK(IU), zeroing out below it
                        lapackf77_dlacpy("U", &n, &n,
                                         A, &lda,
                                         &work[iu], &ldwrku);
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         &work[iu+1], &ldwrku);
                        
                        // Generate Q in A
                        // (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &n, &n, A, &lda,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, n, n, A, lda, &work[itau], &ierr);
                        #endif

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Bidiagonalize R in WORK(IU), copying result to
                        // WORK(IR)
                        // (Workspace: need 2*N*N + 4*N,
                        //           prefer 2*N*N + 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, &work[iu], &ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, &work[iu], ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        lapackf77_dlacpy("U", &n, &n,
                                         &work[iu], &ldwrku,
                                         &work[ir], &ldwrkr);
                        
                        // Generate left bidiagonalizing vectors in WORK(IU)
                        // (Workspace: need 2*N*N + 4*N,
                        //           prefer 2*N*N + 3*N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &n, &n, &n, &work[iu], &ldwrku,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        
                        // Generate right bidiagonalizing vectors in WORK(IR)
                        // (Workspace: need 2*N*N + 4*N-1,
                        //           prefer 2*N*N + 3*N + (N-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &n, &n, &n, &work[ir], &ldwrkr,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of R in WORK(IU) and computing
                        // right singular vectors of R in WORK(IR)
                        // (Workspace: need 2*N*N + BDSPAC)
                        lapackf77_dbdsqr("U", &n, &n, &n, &izero, s, &work[ie],
                                         &work[ir], &ldwrkr, &work[iu], &ldwrku,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply Q in A by left singular vectors of R in
                        // WORK(IU), storing result in U
                        // (Workspace: need N*N)
                        blasf77_dgemm("N", "N", &m, &n, &n,
                                      &c_one,  A, &lda,
                                               &work[iu], &ldwrku,
                                      &c_zero, U, &ldu);
                        
                        // Copy right singular vectors of R to A
                        // (Workspace: need N*N)
                        lapackf77_dlacpy("F", &n, &n,
                                         &work[ir], &ldwrkr,
                                         A, &lda);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + n;
                        
                        // Compute A=Q*R, copying result to U
                        // (Workspace: need 2*N, prefer N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("L", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Generate Q in U
                        // (Workspace: need 2*N, prefer N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &n, &n, U, &ldu,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, n, n, U, ldu, &work[itau], &ierr);
                        #endif

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Zero out below R in A
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         A(1,0), &lda);
                        
                        // Bidiagonalize R in A
                        // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, A, &lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, A, lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply Q in U by left vectors bidiagonalizing R
                        // (Workspace: need 3*N + M, prefer 3*N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("Q", "R", "N", &m, &n, &n,
                                         A, &lda, &work[itauq],
                                         U, &ldu, &work[iwork], &lwork2, &ierr);
                        
                        // Generate right vectors bidiagonalizing R in A
                        // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &n, &n, &n, A, &lda,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in U and computing right
                        // singular vectors of A in A
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &n, &n, &m, &izero, s, &work[ie],
                                         A, &lda, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
                else if (want_vas) {
                    // Path 6 (M much larger than N, JOBU='S', JOBVT='S' or 'A')
                    // N left singular vectors to be computed in U and
                    // N right singular vectors to be computed in VT
                    // printf("Path 6\n");
                    if (lwork >= n*n + max(wrkbrd, bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        iu = 1;
                        if (lwork >= wrkbl + lda*n) {
                            // WORK(IU) is LDA by N
                            ldwrku = lda;
                        }
                        else {
                            // WORK(IU) is N by N
                            ldwrku = n;
                        }
                        itau = iu + ldwrku * n;
                        iwork = itau + n;
                        
                        // Compute A=Q*R
                        // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                                                
                        // Copy R to WORK(IU), zeroing out below it
                        lapackf77_dlacpy("U", &n, &n,
                                         A, &lda,
                                         &work[iu], &ldwrku);
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         &work[iu+1], &ldwrku);
                        
                        // Generate Q in A
                        // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &n, &n, A, &lda,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, n, n, A, lda, &work[itau], &ierr);
                        #endif

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Bidiagonalize R in WORK(IU), copying result to VT
                        // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, &work[iu], &ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, &work[iu], ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        lapackf77_dlacpy("U", &n, &n,
                                         &work[iu], &ldwrku,
                                         VT, &ldvt);
                        
                        // Generate left bidiagonalizing vectors in WORK(IU)
                        // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &n, &n, &n, &work[iu], &ldwrku,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        
                        // Generate right bidiagonalizing vectors in VT
                        // (Workspace: need N*N + 4*N-1, prefer N*N + 3*N + (N-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &n, &n, &n, VT, &ldvt,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of R in WORK(IU) and computing
                        // right singular vectors of R in VT
                        // (Workspace: need N*N + BDSPAC)
                        lapackf77_dbdsqr("U", &n, &n, &n, &izero, s, &work[ie],
                                         VT, &ldvt, &work[iu], &ldwrku,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply Q in A by left singular vectors of R in
                        // WORK(IU), storing result in U
                        // (Workspace: need N*N)
                        blasf77_dgemm("N", "N", &m, &n, &n,
                                      &c_one,  A, &lda,
                                               &work[iu], &ldwrku,
                                      &c_zero, U, &ldu);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + n;
                        
                        // Compute A=Q*R, copying result to U
                        // (Workspace: need 2*N, prefer N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);

                        lapackf77_dlacpy("L", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Generate Q in U
                        // (Workspace: need 2*N, prefer N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &n, &n, U, &ldu,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, n, n, U, ldu, &work[itau], &ierr);
                        #endif
 
                        // Copy R to VT, zeroing out below it
                        lapackf77_dlacpy("U", &n, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        if (n > 1) {
                            lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                             VT(1,0), &ldvt);
                        }
                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Bidiagonalize R in VT
                        // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, VT, &ldvt, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, VT, ldvt, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply Q in U by left bidiagonalizing vectors in VT
                        // (Workspace: need 3*N + M, prefer 3*N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("Q", "R", "N", &m, &n, &n,
                                         VT, &ldvt, &work[itauq],
                                         U, &ldu, &work[iwork], &lwork2, &ierr);
                        
                        // Generate right bidiagonalizing vectors in VT
                        // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &n, &n, &n, VT, &ldvt,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in U and computing right
                        // singular vectors of A in VT
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &n, &n, &m, &izero, s, &work[ie],
                                         VT, &ldvt, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
            }
            else if (want_ua) {
                if (want_vn) {
                    // Path 7 (M much larger than N, JOBU='A', JOBVT='N')
                    // M left singular vectors to be computed in U and
                    // no right singular vectors to be computed
                    // printf("Path 7\n");
                    if (lwork >= n*n + max( max(n + m, wrkbrd), bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        ir = 1;
                        if (lwork >= wrkbl + lda*n) {
                            // WORK(IR) is LDA by N
                            ldwrkr = lda;
                        }
                        else {
                            // WORK(IR) is N by N
                            ldwrkr = n;
                        }
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;
                        
                        // Compute A=Q*R, copying result to U
                        // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("L", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Copy R to WORK(IR), zeroing out below it
                        lapackf77_dlacpy("U", &n, &n,
                                         A, &lda,
                                         &work[ir], &ldwrkr);
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         &work[ir+1], &ldwrkr);
                        
                        // Generate Q in U
                        // (Workspace: need N*N + N + M, prefer N*N + N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &m, &n, U, &ldu,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, m, n, U, ldu, &work[itau], &ierr);
                        #endif

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Bidiagonalize R in WORK(IR)
                        // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, &work[ir], &ldwrkr, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, &work[ir], ldwrkr, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Generate left bidiagonalizing vectors in WORK(IR)
                        // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &n, &n, &n, &work[ir], &ldwrkr,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of R in WORK(IR)
                        // (Workspace: need N*N + BDSPAC)
                        lapackf77_dbdsqr("U", &n, &izero, &n, &izero, s, &work[ie],
                                         cdummy, &ione, &work[ir], &ldwrkr,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply Q in U by left singular vectors of R in
                        // WORK(IR), storing result in A
                        // (Workspace: need N*N)
                        blasf77_dgemm("N", "N", &m, &n, &n,
                                      &c_one,  U, &ldu,
                                               &work[ir], &ldwrkr,
                                      &c_zero, A, &lda);
                        
                        // Copy left singular vectors of A from A to U
                        lapackf77_dlacpy("F", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + n;
                        
                        // Compute A=Q*R, copying result to U
                        // (Workspace: need 2*N, prefer N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("L", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Generate Q in U
                        // (Workspace: need N + M, prefer N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &m, &n, U, &ldu,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, m, n, U, ldu, &work[itau], &ierr);
                        #endif

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Zero out below R in A
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         A(1,0), &lda);
                        
                        // Bidiagonalize R in A
                        // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, A, &lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, A, lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply Q in U by left bidiagonalizing vectors in A
                        // (Workspace: need 3*N + M, prefer 3*N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("Q", "R", "N", &m, &n, &n,
                                         A, &lda, &work[itauq],
                                         U, &ldu, &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in U
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &n, &izero, &m, &izero, s, &work[ie],
                                         cdummy, &ione, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
                else if (want_vo) {
                    // Path 8 (M much larger than N, JOBU='A', JOBVT='O')
                    // M left singular vectors to be computed in U and
                    // N right singular vectors to be overwritten on A
                    // printf("Path 8\n");
                    if (lwork >= 2*n*n + max( max(n + m, wrkbrd), bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        iu = 1;
                        if (lwork >= wrkbl + 2*lda*n) {
                            // WORK(IU) is LDA by N and WORK(IR) is LDA by N
                            ldwrku = lda;
                            ir = iu + ldwrku * n;
                            ldwrkr = lda;
                        }
                        else if (lwork >= wrkbl + (lda + n) * n) {
                            // WORK(IU) is LDA by N and WORK(IR) is N by N
                            ldwrku = lda;
                            ir = iu + ldwrku * n;
                            ldwrkr = n;
                        }
                        else {
                            // WORK(IU) is N by N and WORK(IR) is N by N
                            ldwrku = n;
                            ir = iu + ldwrku * n;
                            ldwrkr = n;
                        }
                        itau = ir + ldwrkr * n;
                        iwork = itau + n;
                        
                        // Compute A=Q*R, copying result to U
                        // (Workspace: need 2*N*N + 2*N, prefer 2*N*N + N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("L", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Generate Q in U
                        // (Workspace: need 2*N*N + N + M, prefer 2*N*N + N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &m, &n, U, &ldu,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, m, n, U, ldu, &work[itau], &ierr);
                        #endif
                        
                        // Copy R to WORK(IU), zeroing out below it
                        lapackf77_dlacpy("U", &n, &n,
                                         A, &lda,
                                         &work[iu], &ldwrku);
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         &work[iu+1], &ldwrku);
                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Bidiagonalize R in WORK(IU), copying result to
                        // WORK(IR)
                        // (Workspace: need 2*N*N + 4*N, prefer 2*N*N + 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, &work[iu], &ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, &work[iu], ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        lapackf77_dlacpy("U", &n, &n,
                                         &work[iu], &ldwrku,
                                         &work[ir], &ldwrkr);
                        
                        // Generate left bidiagonalizing vectors in WORK(IU)
                        // (Workspace: need 2*N*N + 4*N, prefer 2*N*N + 3*N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &n, &n, &n, &work[iu], &ldwrku,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        
                        // Generate right bidiagonalizing vectors in WORK(IR)
                        // (Workspace: need 2*N*N + 4*N-1, prefer 2*N*N + 3*N + (N-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &n, &n, &n, &work[ir], &ldwrkr,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of R in WORK(IU) and computing
                        // right singular vectors of R in WORK(IR)
                        // (Workspace: need 2*N*N + BDSPAC)
                        lapackf77_dbdsqr("U", &n, &n, &n, &izero, s, &work[ie],
                                         &work[ir], &ldwrkr, &work[iu], &ldwrku,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply Q in U by left singular vectors of R in
                        // WORK(IU), storing result in A
                        // (Workspace: need N*N)
                        blasf77_dgemm("N", "N", &m, &n, &n,
                                      &c_one,  U, &ldu,
                                               &work[iu], &ldwrku,
                                      &c_zero, A, &lda);
                        
                        // Copy left singular vectors of A from A to U
                        lapackf77_dlacpy("F", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Copy right singular vectors of R from WORK(IR) to A
                        lapackf77_dlacpy("F", &n, &n,
                                         &work[ir], &ldwrkr,
                                         A, &lda);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + n;
                        
                        // Compute A=Q*R, copying result to U
                        // (Workspace: need 2*N, prefer N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("L", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Generate Q in U
                        // (Workspace: need N + M, prefer N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &m, &n, U, &ldu,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, m, n, U, ldu, &work[itau], &ierr);
                        #endif

                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Zero out below R in A
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         A(1,0), &lda);
                        
                        // Bidiagonalize R in A
                        // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, A, &lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, A, lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply Q in U by left bidiagonalizing vectors in A
                        // (Workspace: need 3*N + M, prefer 3*N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("Q", "R", "N", &m, &n, &n,
                                         A, &lda, &work[itauq],
                                         U, &ldu, &work[iwork], &lwork2, &ierr);
                        
                        // Generate right bidiagonalizing vectors in A
                        // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &n, &n, &n, A, &lda,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in U and computing right
                        // singular vectors of A in A
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &n, &n, &m, &izero, s, &work[ie],
                                         A, &lda, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
                else if (want_vas) {
                    // Path 9 (M much larger than N, JOBU='A', JOBVT='S' or 'A')
                    // M left singular vectors to be computed in U and
                    // N right singular vectors to be computed in VT
                    // printf("Path 9\n");
                    if (lwork >= n*n + max( max(n + m, wrkbrd), bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        iu = 1;
                        if (lwork >= wrkbl + lda*n) {
                            // WORK(IU) is LDA by N
                            ldwrku = lda;
                        }
                        else {
                            // WORK(IU) is N by N
                            ldwrku = n;
                        }
                        itau = iu + ldwrku * n;
                        iwork = itau + n;
                        
                        // Compute A=Q*R, copying result to U
                        // (Workspace: need N*N + 2*N, prefer N*N + N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);

                        lapackf77_dlacpy("L", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Generate Q in U
                        // (Workspace: need N*N + N + M, prefer N*N + N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &m, &n, U, &ldu,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, m, n, U, ldu, &work[itau], &ierr);
                        #endif
                        
                        // Copy R to WORK(IU), zeroing out below it
                        lapackf77_dlacpy("U", &n, &n,
                                         A, &lda,
                                         &work[iu], &ldwrku);
                        lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                         &work[iu+1], &ldwrku);
                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Bidiagonalize R in WORK(IU), copying result to VT
                        // (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, &work[iu], &ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, &work[iu], ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        lapackf77_dlacpy("U", &n, &n,
                                         &work[iu], &ldwrku,
                                         VT, &ldvt);
                        
                        // Generate left bidiagonalizing vectors in WORK(IU)
                        // (Workspace: need N*N + 4*N, prefer N*N + 3*N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &n, &n, &n, &work[iu], &ldwrku,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);

                        // Generate right bidiagonalizing vectors in VT
                        // (Workspace: need N*N + 4*N-1, prefer N*N + 3*N + (N-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &n, &n, &n, VT, &ldvt,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of R in WORK(IU) and computing
                        // right singular vectors of R in VT
                        // (Workspace: need N*N + BDSPAC)
                        lapackf77_dbdsqr("U", &n, &n, &n, &izero, s, &work[ie],
                                         VT, &ldvt, &work[iu], &ldwrku,
                                         cdummy, &ione, &work[iwork], info);
 
                        // Multiply Q in U by left singular vectors of R in
                        // WORK(IU), storing result in A
                        // (Workspace: need N*N)
                        blasf77_dgemm("N", "N", &m, &n, &n,
                                      &c_one,  U, &ldu,
                                               &work[iu], &ldwrku,
                                      &c_zero, A, &lda);

                        // Copy left singular vectors of A from A to U
                        lapackf77_dlacpy("F", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + n;
                        
                        // Compute A=Q*R, copying result to U
                        // (Workspace: need 2*N, prefer N + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgeqrf(&m, &n, A, &lda, &work[itau],
                                         &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("L", &m, &n,
                                         A, &lda,
                                         U, &ldu);
                        
                        // Generate Q in U
                        // (Workspace: need N + M, prefer N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dorgqr(&m, &m, &n, U, &ldu,
                                         &work[itau], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dorgqr2(m, m, n, U, ldu, &work[itau], &ierr);
                        #endif
                        
                        // Copy R from A to VT, zeroing out below it
                        lapackf77_dlacpy("U", &n, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        if (n > 1) {
                            lapackf77_dlaset("L", &n_1, &n_1, &c_zero, &c_zero,
                                             VT(1,0), &ldvt);
                        }
                        ie = itau;
                        itauq = ie + n;
                        itaup = itauq + n;
                        iwork = itaup + n;
                        
                        // Bidiagonalize R in VT
                        // (Workspace: need 4*N, prefer 3*N + 2*N*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&n, &n, VT, &ldvt, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(n, n, VT, ldvt, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply Q in U by left bidiagonalizing vectors in VT
                        // (Workspace: need 3*N + M, prefer 3*N + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("Q", "R", "N", &m, &n, &n,
                                         VT, &ldvt, &work[itauq],
                                         U, &ldu, &work[iwork], &lwork2, &ierr);
                        
                        // Generate right bidiagonalizing vectors in VT
                        // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &n, &n, &n, VT, &ldvt,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + n;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in U and computing right
                        // singular vectors of A in VT
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &n, &n, &m, &izero, s, &work[ie],
                                         VT, &ldvt, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
            }
        }
        else {
            // M < MNTHR
            // Path 10 (M at least N, but not much larger)
            // Reduce to bidiagonal form without QR decomposition
            // printf("Path 10\n");
            ie = 1;
            itauq = ie + n;
            itaup = itauq + n;
            iwork = itaup + n;
            
            // Bidiagonalize A
            // (Workspace: need 3*N + M, prefer 3*N + (M + N)*NB)
            lwork2 = lwork - iwork + 1;

            #if VERSION == 1
            lapackf77_dgebrd(&m, &n, A, &lda, s, &work[ie],
                         &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
            #else
            magma_dgebrd(m, n, A, lda, s, &work[ie],
                         &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
            #endif
            
            if (want_uas) {
                // If left singular vectors desired in U, copy result to U
                // and generate left bidiagonalizing vectors in U
                // (Workspace: need 3*N + NCU, prefer 3*N + NCU*NB)
                lapackf77_dlacpy("L", &m, &n,
                                 A, &lda,
                                 U, &ldu);
                if (want_us) {
                    ncu = n;
                }
                if (want_ua) {
                    ncu = m;
                }
                lwork2 = lwork - iwork + 1;
                lapackf77_dorgbr("Q", &m, &ncu, &n, U, &ldu,
                                 &work[itauq], &work[iwork], &lwork2, &ierr);
            }
            if (want_vas) {
                // If right singular vectors desired in VT, copy result to
                // VT and generate right bidiagonalizing vectors in VT
                // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
                lapackf77_dlacpy("U", &n, &n,
                                 A, &lda,
                                 VT, &ldvt);
                lwork2 = lwork - iwork + 1;
                lapackf77_dorgbr("P", &n, &n, &n, VT, &ldvt,
                                 &work[itaup], &work[iwork], &lwork2, &ierr);
            }
            if (want_uo) {
                // If left singular vectors desired in A, generate left
                // bidiagonalizing vectors in A
                // (Workspace: need 4*N, prefer 3*N + N*NB)
                lwork2 = lwork - iwork + 1;
                lapackf77_dorgbr("Q", &m, &n, &n, A, &lda,
                                 &work[itauq], &work[iwork], &lwork2, &ierr);
            }
            if (want_vo) {
                // If right singular vectors desired in A, generate right
                // bidiagonalizing vectors in A
                // (Workspace: need 4*N-1, prefer 3*N + (N-1)*NB)
                lwork2 = lwork - iwork + 1;
                lapackf77_dorgbr("P", &n, &n, &n, A, &lda,
                                 &work[itaup], &work[iwork], &lwork2, &ierr);
            }

            iwork = ie + n;
            if (want_uas || want_uo) {
                nru = m;
            }
            if (want_un) {
                nru = 0;
            }
            if (want_vas || want_vo) {
                ncvt = n;
            }
            if (want_vn) {
                ncvt = 0;
            }
            if (! want_uo && ! want_vo) {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in U and computing right singular
                // vectors in VT
                // (Workspace: need BDSPAC)
                lapackf77_dbdsqr("U", &n, &ncvt, &nru, &izero, s, &work[ie],
                                 VT, &ldvt, U, &ldu,
                                 cdummy, &ione, &work[iwork], info);
            }
            else if (! want_uo && want_vo) {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in U and computing right singular
                // vectors in A
                // (Workspace: need BDSPAC)
                lapackf77_dbdsqr("U", &n, &ncvt, &nru, &izero, s, &work[ie],
                                 A, &lda, U, &ldu,
                                 cdummy, &ione, &work[iwork], info);
            }
            else {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in A and computing right singular
                // vectors in VT
                // (Workspace: need BDSPAC)
                lapackf77_dbdsqr("U", &n, &ncvt, &nru, &izero, s, &work[ie],
                                 VT, &ldvt, A, &lda,
                                 cdummy, &ione, &work[iwork], info);
            }
        }
    }
    else {
        // A has more columns than rows. If A has sufficiently more
        // columns than rows, first reduce using the LQ decomposition (if
        // sufficient workspace available)
        if (n >= mnthr) {
            if (want_vn) {
                // Path 1t (N much larger than M, JOBVT='N')
                // No right singular vectors to be computed
                // printf("Path 1t\n");
                itau = 1;
                iwork = itau + m;
                
                // Compute A=L*Q
                // (Workspace: need 2*M, prefer M + M*NB)
                lwork2 = lwork - iwork + 1;
                lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                
                // Zero out above L
                lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                 A(0,1), &lda);
                ie = 1;
                itauq = ie + m;
                itaup = itauq + m;
                iwork = itaup + m;
                
                // Bidiagonalize L in A
                // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
                lwork2 = lwork - iwork + 1;
                #if VERSION == 1
                lapackf77_dgebrd(&m, &m, A, &lda, s, &work[ie],
                             &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                #else
                magma_dgebrd(m, m, A, lda, s, &work[ie],
                             &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                #endif
                
                if (want_uo || want_uas) {
                    // If left singular vectors desired, generate Q
                    // (Workspace: need 4*M, prefer 3*M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("Q", &m, &m, &m, A, &lda,
                                     &work[itauq], &work[iwork], &lwork2, &ierr);
                }
                iwork = ie + m;
                nru = 0;
                if (want_uo || want_uas) {
                    nru = m;
                }
                
                // Perform bidiagonal QR iteration, computing left singular
                // vectors of A in A if desired
                // (Workspace: need BDSPAC)
                lapackf77_dbdsqr("U", &m, &izero, &nru, &izero, s, &work[ie],
                                 cdummy, &ione, A, &lda,
                                 cdummy, &ione, &work[iwork], info);
                
                // If left singular vectors desired in U, copy them there
                if (want_uas) {
                    lapackf77_dlacpy("F", &m, &m,
                                     A, &lda,
                                     U, &ldu);
                }
            }
            else if (want_vo && want_un) {
                // Path 2t (N much larger than M, JOBU='N', JOBVT='O')
                // M right singular vectors to be overwritten on A and
                // no left singular vectors to be computed
                // printf("Path 2t\n");
                if (lwork >= m*m + max(wrkbrd, bdspac)) {
                    // Sufficient workspace for a fast algorithm
                    ir = 1;
                    if (lwork >= max(wrkbl, lda*n + m) + lda*m) {
                        // WORK(IU) is LDA by N and WORK(IR) is LDA by M
                        ldwrku = lda;
                        chunk = n;
                        ldwrkr = lda;
                    }
                    else /* if (complicated condition) */ {
                        if (lwork >= max(wrkbl, lda*n + m) + m*m) {
                            // WORK(IU) is LDA by N and WORK(IR) is M by M
                            ldwrku = lda;
                            chunk = n;
                            ldwrkr = m;
                        }
                        else {
                            // WORK(IU) is M by CHUNK and WORK(IR) is M by M
                            ldwrku = m;
                            chunk = (lwork - m*m - m) / m;
                            ldwrkr = m;
                        }
                    }
                    itau = ir + ldwrkr * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                    
                    // Copy L to WORK(IR) and zero out above it
                    lapackf77_dlacpy("L", &m, &m,
                                     A, &lda,
                                     &work[ir], &ldwrkr);
                    lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                     &work[ir+ldwrkr], &ldwrkr);
                    
                    // Generate Q in A
                    // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorglq(&m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                    ie = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in WORK(IR)
                    // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd(&m, &m, &work[ir], &ldwrkr, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dgebrd(m, m, &work[ir], ldwrkr, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                    #endif
                    
                    // Generate right vectors bidiagonalizing L
                    // (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("P", &m, &m, &m, &work[ir], &ldwrkr,
                                     &work[itaup], &work[iwork], &lwork2, &ierr);
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing right
                    // singular vectors of L in WORK(IR)
                    // (Workspace: need M*M + BDSPAC)
                    lapackf77_dbdsqr("U", &m, &m, &izero, &izero, s, &work[ie],
                                     &work[ir], &ldwrkr, cdummy, &ione,
                                     cdummy, &ione, &work[iwork], info);
                    iu = ie + m;
                    
                    // Multiply right singular vectors of L in WORK(IR) by Q
                    // in A, storing result in WORK(IU) and copying to A
                    // (Workspace: need M*M + 2*M, prefer M*M + M*N + M)
                    for (i = 1; (chunk < 0 ? i >= n : i <= n); i += chunk) {
                        blk = min(n - i + 1,chunk);
                        blasf77_dgemm("N", "N", &m, &blk, &m,
                                      &c_one,  &work[ir], &ldwrkr,
                                               A(0,i-1), &lda,
                                      &c_zero, &work[iu], &ldwrku);
                        lapackf77_dlacpy("F", &m, &blk,
                                         &work[iu], &ldwrku,
                                         A(0,i-1), &lda);
                    }
                }
                else {
                    // Insufficient workspace for a fast algorithm
                    ie = 1;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize A
                    // (Workspace: need 3*M + N, prefer 3*M + (M + N)*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd(&m, &n, A, &lda, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dgebrd(m, n, A, lda, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                    #endif
                    
                    // Generate right vectors bidiagonalizing A
                    // (Workspace: need 4*M, prefer 3*M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("P", &m, &n, &m, A, &lda,
                                     &work[itaup], &work[iwork], &lwork2, &ierr);
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing right
                    // singular vectors of A in A
                    // (Workspace: need BDSPAC)
                    lapackf77_dbdsqr("L", &m, &n, &izero, &izero, s, &work[ie],
                                     A, &lda, cdummy, &ione,
                                     cdummy, &ione, &work[iwork], info);
                }
            }
            else if (want_vo && want_uas) {
                // Path 3t (N much larger than M, JOBU='S' or 'A', JOBVT='O')
                // M right singular vectors to be overwritten on A and
                // M left singular vectors to be computed in U
                // printf("Path 3t\n");
                if (lwork >= m*m + max(wrkbrd, bdspac)) {
                    // Sufficient workspace for a fast algorithm
                    ir = 1;
                    if (lwork >= max(wrkbl, lda*n + m) + lda*m) {
                        // WORK(IU) is LDA by N and WORK(IR) is LDA by M
                        ldwrku = lda;
                        chunk = n;
                        ldwrkr = lda;
                    }
                    else /* if (complicated condition) */ {
                        if (lwork >= max(wrkbl, lda*n + m) + m*m) {
                            // WORK(IU) is LDA by N and WORK(IR) is M by M
                            ldwrku = lda;
                            chunk = n;
                            ldwrkr = m;
                        }
                        else {
                            // WORK(IU) is M by CHUNK and WORK(IR) is M by M
                            ldwrku = m;
                            chunk = (lwork - m*m - m) / m;
                            ldwrkr = m;
                        }
                    }
                    itau = ir + ldwrkr * m;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                    
                    // Copy L to U, zeroing about above it
                    lapackf77_dlacpy("L", &m, &m,
                                     A, &lda,
                                     U, &ldu);
                    lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                     U(0,1), &ldu);
                    
                    // Generate Q in A
                    // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorglq(&m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                    ie = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in U, copying result to WORK(IR)
                    // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd(&m, &m, U, &ldu, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dgebrd(m, m, U, ldu, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                    #endif
                    lapackf77_dlacpy("U", &m, &m,
                                     U, &ldu,
                                     &work[ir], &ldwrkr);
                    
                    // Generate right vectors bidiagonalizing L in WORK(IR)
                    // (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("P", &m, &m, &m, &work[ir], &ldwrkr,
                                     &work[itaup], &work[iwork], &lwork2, &ierr);
                    
                    // Generate left vectors bidiagonalizing L in U
                    // (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("Q", &m, &m, &m, U, &ldu,
                                     &work[itauq], &work[iwork], &lwork2, &ierr);
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of L in U, and computing right
                    // singular vectors of L in WORK(IR)
                    // (Workspace: need M*M + BDSPAC)
                    lapackf77_dbdsqr("U", &m, &m, &m, &izero, s, &work[ie],
                                     &work[ir], &ldwrkr, U, &ldu,
                                     cdummy, &ione, &work[iwork], info);
                    iu = ie + m;
                    
                    // Multiply right singular vectors of L in WORK(IR) by Q
                    // in A, storing result in WORK(IU) and copying to A
                    // (Workspace: need M*M + 2*M, prefer M*M + M*N + M))
                    for (i = 1; (chunk < 0 ? i >= n : i <= n); i += chunk) {
                        blk = min(n - i + 1,chunk);
                        blasf77_dgemm("N", "N", &m, &blk, &m,
                                      &c_one,  &work[ir], &ldwrkr,
                                               A(0,i-1), &lda,
                                      &c_zero, &work[iu], &ldwrku);
                        lapackf77_dlacpy("F", &m, &blk,
                                         &work[iu], &ldwrku,
                                         A(0,i-1), &lda);
                    }
                }
                else {
                    // Insufficient workspace for a fast algorithm
                    itau = 1;
                    iwork = itau + m;
                    
                    // Compute A=L*Q
                    // (Workspace: need 2*M, prefer M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                    
                    // Copy L to U, zeroing out above it
                    lapackf77_dlacpy("L", &m, &m,
                                     A, &lda,
                                     U, &ldu);
                    lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                     U(0,1), &ldu);
                    
                    // Generate Q in A
                    // (Workspace: need 2*M, prefer M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorglq(&m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                    ie = itau;
                    itauq = ie + m;
                    itaup = itauq + m;
                    iwork = itaup + m;
                    
                    // Bidiagonalize L in U
                    // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
                    lwork2 = lwork - iwork + 1;
                    #if VERSION == 1
                    lapackf77_dgebrd(&m, &m, U, &ldu, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                    #else
                    magma_dgebrd(m, m, U, ldu, s, &work[ie],
                                 &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                    #endif
                    
                    // Multiply right vectors bidiagonalizing L by Q in A
                    // (Workspace: need 3*M + N, prefer 3*M + N*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dormbr("P", "L", "T", &m, &n, &m,
                                     U, &ldu, &work[itaup],
                                     A, &lda, &work[iwork], &lwork2, &ierr);
                    
                    // Generate left vectors bidiagonalizing L in U
                    // (Workspace: need 4*M, prefer 3*M + M*NB)
                    lwork2 = lwork - iwork + 1;
                    lapackf77_dorgbr("Q", &m, &m, &m, U, &ldu,
                                     &work[itauq], &work[iwork], &lwork2, &ierr);
                    iwork = ie + m;
                    
                    // Perform bidiagonal QR iteration, computing left
                    // singular vectors of A in U and computing right
                    // singular vectors of A in A
                    // (Workspace: need BDSPAC)
                    lapackf77_dbdsqr("U", &m, &n, &m, &izero, s, &work[ie],
                                     A, &lda, U, &ldu,
                                     cdummy, &ione, &work[iwork], info);
                }
            }
            else if (want_vs) {
                if (want_un) {
                    // Path 4t (N much larger than M, JOBU='N', JOBVT='S')
                    // M right singular vectors to be computed in VT and
                    // no left singular vectors to be computed
                    // printf("Path 4t\n");
                    if (lwork >= m*m + max(wrkbrd, bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        ir = 1;
                        if (lwork >= wrkbl + lda*m) {
                            // WORK(IR) is LDA by M
                            ldwrkr = lda;
                        }
                        else {
                            // WORK(IR) is M by M
                            ldwrkr = m;
                        }
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;
                        
                        // Compute A=L*Q
                        // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        
                        // Copy L to WORK(IR), zeroing out above it
                        lapackf77_dlacpy("L", &m, &m,
                                         A, &lda,
                                         &work[ir], &ldwrkr);
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         &work[ir+ldwrkr], &ldwrkr);
                        
                        // Generate Q in A
                        // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Bidiagonalize L in WORK(IR)
                        // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, &work[ir], &ldwrkr, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, &work[ir], ldwrkr, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Generate right vectors bidiagonalizing L in
                        // WORK(IR)
                        // (Workspace: need M*M + 4*M, prefer M*M + 3*M + (M-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &m, &m, &m, &work[ir], &ldwrkr,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing right
                        // singular vectors of L in WORK(IR)
                        // (Workspace: need M*M + BDSPAC)
                        lapackf77_dbdsqr("U", &m, &m, &izero, &izero, s, &work[ie],
                                         &work[ir], &ldwrkr, cdummy, &ione,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply right singular vectors of L in WORK(IR) by
                        // Q in A, storing result in VT
                        // (Workspace: need M*M)
                        blasf77_dgemm("N", "N", &m, &n, &m,
                                      &c_one,  &work[ir], &ldwrkr,
                                               A, &lda,
                                      &c_zero, VT, &ldvt);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + m;
                        
                        // Compute A=L*Q
                        // (Workspace: need 2*M, prefer M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        
                        // Copy result to VT
                        lapackf77_dlacpy("U", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Generate Q in VT
                        // (Workspace: need 2*M, prefer M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&m, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Zero out above L in A
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         A(0,1), &lda);
                        
                        // Bidiagonalize L in A
                        // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, A, &lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, A, lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply right vectors bidiagonalizing L by Q in VT
                        // (Workspace: need 3*M + N, prefer 3*M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("P", "L", "T", &m, &n, &m,
                                         A, &lda, &work[itaup],
                                         VT, &ldvt, &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing right
                        // singular vectors of A in VT
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &m, &n, &izero, &izero, s, &work[ie],
                                         VT, &ldvt, cdummy, &ione,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
                else if (want_uo) {
                    // Path 5t (N much larger than M, JOBU='O', JOBVT='S')
                    // M right singular vectors to be computed in VT and
                    // M left singular vectors to be overwritten on A
                    // printf("Path 5t\n");
                    if (lwork >= 2*m*m + max(wrkbrd, bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        iu = 1;
                        if (lwork >= wrkbl + 2*lda*m) {
                            // WORK(IU) is LDA by M and WORK(IR) is LDA by M
                            ldwrku = lda;
                            ir = iu + ldwrku * m;
                            ldwrkr = lda;
                        }
                        else if (lwork >= wrkbl + (lda + m) * m) {
                            // WORK(IU) is LDA by M and WORK(IR) is M by M
                            ldwrku = lda;
                            ir = iu + ldwrku * m;
                            ldwrkr = m;
                        }
                        else {
                            // WORK(IU) is M by M and WORK(IR) is M by M
                            ldwrku = m;
                            ir = iu + ldwrku * m;
                            ldwrkr = m;
                        }
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;
                        
                        // Compute A=L*Q
                        // (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        
                        // Copy L to WORK(IU), zeroing out below it
                        lapackf77_dlacpy("L", &m, &m,
                                         A, &lda,
                                         &work[iu], &ldwrku);
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         &work[iu+ldwrku], &ldwrku);
                        
                        // Generate Q in A
                        // (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Bidiagonalize L in WORK(IU), copying result to
                        // WORK(IR)
                        // (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, &work[iu], &ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, &work[iu], ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        lapackf77_dlacpy("L", &m, &m,
                                         &work[iu], &ldwrku,
                                         &work[ir], &ldwrkr);
                        
                        // Generate right bidiagonalizing vectors in WORK(IU)
                        // (Workspace: need 2*M*M + 4*M-1, prefer 2*M*M + 3*M + (M-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &m, &m, &m, &work[iu], &ldwrku,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        
                        // Generate left bidiagonalizing vectors in WORK(IR)
                        // (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &m, &m, &m, &work[ir], &ldwrkr,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of L in WORK(IR) and computing
                        // right singular vectors of L in WORK(IU)
                        // (Workspace: need 2*M*M + BDSPAC)
                        lapackf77_dbdsqr("U", &m, &m, &m, &izero, s, &work[ie],
                                         &work[iu], &ldwrku, &work[ir], &ldwrkr,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply right singular vectors of L in WORK(IU) by
                        // Q in A, storing result in VT
                        // (Workspace: need M*M)
                        blasf77_dgemm("N", "N", &m, &n, &m,
                                      &c_one,  &work[iu], &ldwrku,
                                               A, &lda,
                                      &c_zero, VT, &ldvt);
                        
                        // Copy left singular vectors of L to A
                        // (Workspace: need M*M)
                        lapackf77_dlacpy("F", &m, &m,
                                         &work[ir], &ldwrkr,
                                         A, &lda);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + m;
                        
                        // Compute A=L*Q, copying result to VT
                        // (Workspace: need 2*M, prefer M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("U", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Generate Q in VT
                        // (Workspace: need 2*M, prefer M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&m, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Zero out above L in A
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         A(0,1), &lda);
                        
                        // Bidiagonalize L in A
                        // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, A, &lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, A, lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply right vectors bidiagonalizing L by Q in VT
                        // (Workspace: need 3*M + N, prefer 3*M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("P", "L", "T", &m, &n, &m,
                                         A, &lda, &work[itaup],
                                         VT, &ldvt, &work[iwork], &lwork2, &ierr);
                        
                        // Generate left bidiagonalizing vectors of L in A
                        // (Workspace: need 4*M, prefer 3*M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &m, &m, &m, A, &lda,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in A and computing right
                        // singular vectors of A in VT
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &m, &n, &m, &izero, s, &work[ie],
                                         VT, &ldvt, A, &lda,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
                else if (want_uas) {
                    // Path 6t (N much larger than M, JOBU='S' or 'A', JOBVT='S')
                    // M right singular vectors to be computed in VT and
                    // M left singular vectors to be computed in U
                    // printf("Path 6t\n");
                    if (lwork >= m*m + max(wrkbrd, bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        iu = 1;
                        if (lwork >= wrkbl + lda*m) {
                            // WORK(IU) is LDA by N
                            ldwrku = lda;
                        }
                        else {
                            // WORK(IU) is LDA by M
                            ldwrku = m;
                        }
                        itau = iu + ldwrku * m;
                        iwork = itau + m;
                        
                        // Compute A=L*Q
                        // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        
                        // Copy L to WORK(IU), zeroing out above it
                        lapackf77_dlacpy("L", &m, &m,
                                         A, &lda,
                                         &work[iu], &ldwrku);
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         &work[iu+ldwrku], &ldwrku);
                        
                        // Generate Q in A
                        // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&m, &n, &m, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Bidiagonalize L in WORK(IU), copying result to U
                        // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, &work[iu], &ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, &work[iu], ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        lapackf77_dlacpy("L", &m, &m,
                                         &work[iu], &ldwrku,
                                         U, &ldu);
                        
                        // Generate right bidiagonalizing vectors in WORK(IU)
                        // (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &m, &m, &m, &work[iu], &ldwrku,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        
                        // Generate left bidiagonalizing vectors in U
                        // (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &m, &m, &m, U, &ldu,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of L in U and computing right
                        // singular vectors of L in WORK(IU)
                        // (Workspace: need M*M + BDSPAC)
                        lapackf77_dbdsqr("U", &m, &m, &m, &izero, s, &work[ie],
                                         &work[iu], &ldwrku, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply right singular vectors of L in WORK(IU) by
                        // Q in A, storing result in VT
                        // (Workspace: need M*M)
                        blasf77_dgemm("N", "N", &m, &n, &m,
                                      &c_one,  &work[iu], &ldwrku,
                                               A, &lda,
                                      &c_zero, VT, &ldvt);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + m;
                        
                        // Compute A=L*Q, copying result to VT
                        // (Workspace: need 2*M, prefer M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("U", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Generate Q in VT
                        // (Workspace: need 2*M, prefer M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&m, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr);
                        
                        // Copy L to U, zeroing out above it
                        lapackf77_dlacpy("L", &m, &m,
                                         A, &lda,
                                         U, &ldu);
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         U(0,1), &ldu);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Bidiagonalize L in U
                        // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, U, &ldu, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, U, ldu, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply right bidiagonalizing vectors in U by Q in VT
                        // (Workspace: need 3*M + N, prefer 3*M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("P", "L", "T", &m, &n, &m,
                                         U, &ldu, &work[itaup],
                                         VT, &ldvt, &work[iwork], &lwork2, &ierr);
                        
                        // Generate left bidiagonalizing vectors in U
                        // (Workspace: need 4*M, prefer 3*M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &m, &m, &m, U, &ldu,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in U and computing right
                        // singular vectors of A in VT
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &m, &n, &m, &izero, s, &work[ie],
                                         VT, &ldvt, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
            }
            else if (want_va) {
                if (want_un) {
                    // Path 7t (N much larger than M, JOBU='N', JOBVT='A')
                    // N right singular vectors to be computed in VT and
                    // no left singular vectors to be computed
                    // printf("Path 7t\n");
                    if (lwork >= m*m + max( max(n + m, wrkbrd), bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        ir = 1;
                        if (lwork >= wrkbl + lda*m) {
                            // WORK(IR) is LDA by M
                            ldwrkr = lda;
                        }
                        else {
                            // WORK(IR) is M by M
                            ldwrkr = m;
                        }
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;
                        
                        // Compute A=L*Q, copying result to VT
                        // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("U", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Copy L to WORK(IR), zeroing out above it
                        lapackf77_dlacpy("L", &m, &m,
                                         A, &lda,
                                         &work[ir], &ldwrkr);
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         &work[ir+ldwrkr], &ldwrkr);
                        
                        // Generate Q in VT
                        // (Workspace: need M*M + M + N, prefer M*M + M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Bidiagonalize L in WORK(IR)
                        // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, &work[ir], &ldwrkr, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, &work[ir], ldwrkr, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Generate right bidiagonalizing vectors in WORK(IR)
                        // (Workspace: need M*M + 4*M-1, prefer M*M + 3*M + (M-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &m, &m, &m, &work[ir], &ldwrkr,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing right
                        // singular vectors of L in WORK(IR)
                        // (Workspace: need M*M + BDSPAC)
                        lapackf77_dbdsqr("U", &m, &m, &izero, &izero, s, &work[ie],
                                         &work[ir], &ldwrkr, cdummy, &ione,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply right singular vectors of L in WORK(IR) by
                        // Q in VT, storing result in A
                        // (Workspace: need M*M)
                        blasf77_dgemm("N", "N", &m, &n, &m,
                                      &c_one,  &work[ir], &ldwrkr,
                                               VT, &ldvt,
                                      &c_zero, A, &lda);
                        
                        // Copy right singular vectors of A from A to VT
                        lapackf77_dlacpy("F", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + m;
                        
                        // Compute A=L*Q, copying result to VT
                        // (Workspace: need 2*M, prefer M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("U", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Generate Q in VT
                        // (Workspace: need M + N, prefer M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Zero out above L in A
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         A(0,1), &lda);
                        
                        // Bidiagonalize L in A
                        // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, A, &lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, A, lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply right bidiagonalizing vectors in A by Q in VT
                        // (Workspace: need 3*M + N, prefer 3*M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("P", "L", "T", &m, &n, &m,
                                         A, &lda, &work[itaup],
                                         VT, &ldvt, &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing right
                        // singular vectors of A in VT
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &m, &n, &izero, &izero, s, &work[ie],
                                         VT, &ldvt, cdummy, &ione,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
                else if (want_uo) {
                    // Path 8t (N much larger than M, JOBU='O', JOBVT='A')
                    // N right singular vectors to be computed in VT and
                    // M left singular vectors to be overwritten on A
                    // printf("Path 8t\n");
                    if (lwork >= 2*m*m + max( max(n + m, wrkbrd), bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        iu = 1;
                        if (lwork >= wrkbl + 2*lda*m) {
                            // WORK(IU) is LDA by M and WORK(IR) is LDA by M
                            ldwrku = lda;
                            ir = iu + ldwrku * m;
                            ldwrkr = lda;
                        }
                        else if (lwork >= wrkbl + (lda + m) * m) {
                            // WORK(IU) is LDA by M and WORK(IR) is M by M
                            ldwrku = lda;
                            ir = iu + ldwrku * m;
                            ldwrkr = m;
                        }
                        else {
                            // WORK(IU) is M by M and WORK(IR) is M by M
                            ldwrku = m;
                            ir = iu + ldwrku * m;
                            ldwrkr = m;
                        }
                        itau = ir + ldwrkr * m;
                        iwork = itau + m;
                        
                        // Compute A=L*Q, copying result to VT
                        // (Workspace: need 2*M*M + 2*M, prefer 2*M*M + M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("U", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Generate Q in VT
                        // (Workspace: need 2*M*M + M + N, prefer 2*M*M + M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr);
                        
                        // Copy L to WORK(IU), zeroing out above it
                        lapackf77_dlacpy("L", &m, &m,
                                         A, &lda,
                                         &work[iu], &ldwrku);
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         &work[iu+ldwrku], &ldwrku);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Bidiagonalize L in WORK(IU), copying result to WORK(IR)
                        // (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, &work[iu], &ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, &work[iu], ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        lapackf77_dlacpy("L", &m, &m,
                                         &work[iu], &ldwrku,
                                         &work[ir], &ldwrkr);
                        
                        // Generate right bidiagonalizing vectors in WORK(IU)
                        // (Workspace: need 2*M*M + 4*M-1, prefer 2*M*M + 3*M + (M-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &m, &m, &m, &work[iu], &ldwrku,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        
                        // Generate left bidiagonalizing vectors in WORK(IR)
                        // (Workspace: need 2*M*M + 4*M, prefer 2*M*M + 3*M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &m, &m, &m, &work[ir], &ldwrkr,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of L in WORK(IR) and computing
                        // right singular vectors of L in WORK(IU)
                        // (Workspace: need 2*M*M + BDSPAC)
                        lapackf77_dbdsqr("U", &m, &m, &m, &izero, s, &work[ie],
                                         &work[iu], &ldwrku, &work[ir], &ldwrkr,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply right singular vectors of L in WORK(IU) by
                        // Q in VT, storing result in A
                        // (Workspace: need M*M)
                        blasf77_dgemm("N", "N", &m, &n, &m,
                                      &c_one,  &work[iu], &ldwrku,
                                               VT, &ldvt,
                                      &c_zero, A, &lda);
                        
                        // Copy right singular vectors of A from A to VT
                        lapackf77_dlacpy("F", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Copy left singular vectors of A from WORK(IR) to A
                        lapackf77_dlacpy("F", &m, &m,
                                         &work[ir], &ldwrkr,
                                         A, &lda);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + m;
                        
                        // Compute A=L*Q, copying result to VT
                        // (Workspace: need 2*M, prefer M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("U", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Generate Q in VT
                        // (Workspace: need M + N, prefer M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Zero out above L in A
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         A(0,1), &lda);
                        
                        // Bidiagonalize L in A
                        // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, A, &lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, A, lda, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply right bidiagonalizing vectors in A by Q in VT
                        // (Workspace: need 3*M + N, prefer 3*M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("P", "L", "T", &m, &n, &m,
                                         A, &lda, &work[itaup],
                                         VT, &ldvt, &work[iwork], &lwork2, &ierr);
                        
                        // Generate left bidiagonalizing vectors in A
                        // (Workspace: need 4*M, prefer 3*M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &m, &m, &m, A, &lda,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in A and computing right
                        // singular vectors of A in VT
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &m, &n, &m, &izero, s, &work[ie],
                                         VT, &ldvt, A, &lda,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
                else if (want_uas) {
                    // Path 9t (N much larger than M, JOBU='S' or 'A', JOBVT='A')
                    // N right singular vectors to be computed in VT and
                    // M left singular vectors to be computed in U
                    // printf("Path 9t\n");
                    if (lwork >= m*m + max( max(n + m, wrkbrd), bdspac)) {
                        // Sufficient workspace for a fast algorithm
                        iu = 1;
                        if (lwork >= wrkbl + lda*m) {
                            // WORK(IU) is LDA by M
                            ldwrku = lda;
                        }
                        else {
                            // WORK(IU) is M by M
                            ldwrku = m;
                        }
                        itau = iu + ldwrku * m;
                        iwork = itau + m;
                        
                        // Compute A=L*Q, copying result to VT
                        // (Workspace: need M*M + 2*M, prefer M*M + M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("U", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Generate Q in VT
                        // (Workspace: need M*M + M + N, prefer M*M + M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr);
                        
                        // Copy L to WORK(IU), zeroing out above it
                        lapackf77_dlacpy("L", &m, &m,
                                         A, &lda,
                                         &work[iu], &ldwrku);
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         &work[iu+ldwrku], &ldwrku);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Bidiagonalize L in WORK(IU), copying result to U
                        // (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, &work[iu], &ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, &work[iu], ldwrku, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        lapackf77_dlacpy("L", &m, &m,
                                         &work[iu], &ldwrku,
                                         U, &ldu);
                        
                        // Generate right bidiagonalizing vectors in WORK(IU)
                        // (Workspace: need M*M + 4*M, prefer M*M + 3*M + (M-1)*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("P", &m, &m, &m, &work[iu], &ldwrku,
                                         &work[itaup], &work[iwork], &lwork2, &ierr);
                        
                        // Generate left bidiagonalizing vectors in U
                        // (Workspace: need M*M + 4*M, prefer M*M + 3*M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &m, &m, &m, U, &ldu,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of L in U and computing right
                        // singular vectors of L in WORK(IU)
                        // (Workspace: need M*M + BDSPAC)
                        lapackf77_dbdsqr("U", &m, &m, &m, &izero, s, &work[ie],
                                         &work[iu], &ldwrku, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                        
                        // Multiply right singular vectors of L in WORK(IU) by
                        // Q in VT, storing result in A
                        // (Workspace: need M*M)
                        blasf77_dgemm("N", "N", &m, &n, &m,
                                      &c_one,  &work[iu], &ldwrku,
                                               VT, &ldvt,
                                      &c_zero, A, &lda);
                        
                        // Copy right singular vectors of A from A to VT
                        lapackf77_dlacpy("F", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                    }
                    else {
                        // Insufficient workspace for a fast algorithm
                        itau = 1;
                        iwork = itau + m;
                        
                        // Compute A=L*Q, copying result to VT
                        // (Workspace: need 2*M, prefer M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dgelqf(&m, &n, A, &lda, &work[itau], &work[iwork], &lwork2, &ierr);
                        lapackf77_dlacpy("U", &m, &n,
                                         A, &lda,
                                         VT, &ldvt);
                        
                        // Generate Q in VT
                        // (Workspace: need M + N, prefer M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorglq(&n, &n, &m, VT, &ldvt, &work[itau], &work[iwork], &lwork2, &ierr);
                        
                        // Copy L to U, zeroing out above it
                        lapackf77_dlacpy("L", &m, &m,
                                         A, &lda,
                                         U, &ldu);
                        lapackf77_dlaset("U", &m_1, &m_1, &c_zero, &c_zero,
                                         U(0,1), &ldu);
                        ie = itau;
                        itauq = ie + m;
                        itaup = itauq + m;
                        iwork = itaup + m;
                        
                        // Bidiagonalize L in U
                        // (Workspace: need 4*M, prefer 3*M + 2*M*NB)
                        lwork2 = lwork - iwork + 1;
                        #if VERSION == 1
                        lapackf77_dgebrd(&m, &m, U, &ldu, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
                        #else
                        magma_dgebrd(m, m, U, ldu, s, &work[ie],
                                     &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
                        #endif
                        
                        // Multiply right bidiagonalizing vectors in U by Q in VT
                        // (Workspace: need 3*M + N, prefer 3*M + N*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dormbr("P", "L", "T", &m, &n, &m,
                                         U, &ldu, &work[itaup],
                                         VT, &ldvt, &work[iwork], &lwork2, &ierr);
                        
                        // Generate left bidiagonalizing vectors in U
                        // (Workspace: need 4*M, prefer 3*M + M*NB)
                        lwork2 = lwork - iwork + 1;
                        lapackf77_dorgbr("Q", &m, &m, &m, U, &ldu,
                                         &work[itauq], &work[iwork], &lwork2, &ierr);
                        iwork = ie + m;
                        
                        // Perform bidiagonal QR iteration, computing left
                        // singular vectors of A in U and computing right
                        // singular vectors of A in VT
                        // (Workspace: need BDSPAC)
                        lapackf77_dbdsqr("U", &m, &n, &m, &izero, s, &work[ie],
                                         VT, &ldvt, U, &ldu,
                                         cdummy, &ione, &work[iwork], info);
                    }
                }
            }
        }
        else {
            // N < MNTHR
            // Path 10t (N greater than M, but not much larger)
            // Reduce to bidiagonal form without LQ decomposition
            // printf("Path 10t\n");
            ie = 1;
            itauq = ie + m;
            itaup = itauq + m;
            iwork = itaup + m;
            
            // Bidiagonalize A
            // (Workspace: need 3*M + N, prefer 3*M + (M + N)*NB)
            lwork2 = lwork - iwork + 1;
            #if VERSION == 1
            lapackf77_dgebrd(&m, &n, A, &lda, s, &work[ie],
                         &work[itauq], &work[itaup], &work[iwork], &lwork2, &ierr);
            #else
            magma_dgebrd(m, n, A, lda, s, &work[ie],
                         &work[itauq], &work[itaup], &work[iwork], lwork2, &ierr);
            #endif
            
            if (want_uas) {
                // If left singular vectors desired in U, copy result to U
                // and generate left bidiagonalizing vectors in U
                // (Workspace: need 4*M-1, prefer 3*M + (M-1)*NB)
                lapackf77_dlacpy("L", &m, &m,
                                 A, &lda,
                                 U, &ldu);
                lwork2 = lwork - iwork + 1;
                lapackf77_dorgbr("Q", &m, &m, &n, U, &ldu,
                                 &work[itauq], &work[iwork], &lwork2, &ierr);
            }
            if (want_vas) {
                // If right singular vectors desired in VT, copy result to
                // VT and generate right bidiagonalizing vectors in VT
                // (Workspace: need 3*M + NRVT, prefer 3*M + NRVT*NB)
                lapackf77_dlacpy("U", &m, &n,
                                 A, &lda,
                                 VT, &ldvt);
                if (want_va) {
                    nrvt = n;
                }
                if (want_vs) {
                    nrvt = m;
                }
                lwork2 = lwork - iwork + 1;
                lapackf77_dorgbr("P", &nrvt, &n, &m, VT, &ldvt,
                                 &work[itaup], &work[iwork], &lwork2, &ierr);
            }
            if (want_uo) {
                // If left singular vectors desired in A, generate left
                // bidiagonalizing vectors in A
                // (Workspace: need 4*M-1, prefer 3*M + (M-1)*NB)
                lwork2 = lwork - iwork + 1;
                lapackf77_dorgbr("Q", &m, &m, &n, A, &lda,
                                 &work[itauq], &work[iwork], &lwork2, &ierr);
            }
            if (want_vo) {
                // If right singular vectors desired in A, generate right
                // bidiagonalizing vectors in A
                // (Workspace: need 4*M, prefer 3*M + M*NB)
                lwork2 = lwork - iwork + 1;
                lapackf77_dorgbr("P", &m, &n, &m, A, &lda,
                                 &work[itaup], &work[iwork], &lwork2, &ierr);
            }
            iwork = ie + m;
            if (want_uas || want_uo) {
                nru = m;
            }
            if (want_un) {
                nru = 0;
            }
            if (want_vas || want_vo) {
                ncvt = n;
            }
            if (want_vn) {
                ncvt = 0;
            }
            if (! want_uo && ! want_vo) {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in U and computing right singular
                // vectors in VT
                // (Workspace: need BDSPAC)
                lapackf77_dbdsqr("L", &m, &ncvt, &nru, &izero, s, &work[ie],
                                 VT, &ldvt, U, &ldu,
                                 cdummy, &ione, &work[iwork], info);
            }
            else if (! want_uo && want_vo) {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in U and computing right singular
                // vectors in A
                // (Workspace: need BDSPAC)
                lapackf77_dbdsqr("L", &m, &ncvt, &nru, &izero, s, &work[ie],
                                 A, &lda, U, &ldu,
                                 cdummy, &ione, &work[iwork], info);
            }
            else {
                // Perform bidiagonal QR iteration, if desired, computing
                // left singular vectors in A and computing right singular
                // vectors in VT
                // (Workspace: need BDSPAC)
                lapackf77_dbdsqr("L", &m, &ncvt, &nru, &izero, s, &work[ie],
                                 VT, &ldvt, A, &lda,
                                 cdummy, &ione, &work[iwork], info);
            }
        }
    }
    
    // If DBDSQR failed to converge, copy unconverged superdiagonals
    // to WORK( 2:MINMN )
    if (*info != 0) {
        if (ie > 2) {
            m_1 = minmn - 1;
            for (i = 1; i <= m_1; ++i) {
                work[i + 1] = work[i + ie - 1];
            }
        }
        if (ie < 2) {
            for (i = minmn - 1; i >= 1; --i) {
                work[i + 1] = work[i + ie - 1];
            }
        }
    }
    
    // Undo scaling if necessary
    if (iscl == 1) {
        if (anrm > bignum) {
            lapackf77_dlascl("G", &izero, &izero, &bignum, &anrm, &minmn, &ione,
                             s, &minmn, &ierr);
        }
        if (*info != 0 && anrm > bignum) {
            m_1 = minmn - 1;
            lapackf77_dlascl("G", &izero, &izero, &bignum, &anrm, &m_1, &ione,
                             &work[2], &minmn, &ierr);
        }
        if (anrm < smlnum) {
            lapackf77_dlascl("G", &izero, &izero, &smlnum, &anrm, &minmn, &ione,
                             s, &minmn, &ierr);
        }
        if (*info != 0 && anrm < smlnum) {
            m_1 = minmn - 1;
            lapackf77_dlascl("G", &izero, &izero, &smlnum, &anrm, &m_1, &ione,
                             &work[2], &minmn, &ierr);
        }
    }

    return *info;
} /* magma_dgesvd */
