/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zgeev_m.cpp normal z -> c, Fri Jan 30 19:00:18 2015
       @author Stan Tomov
       @author Mark Gates
*/
#include "common_magma.h"

#define PRECISION_c
#define COMPLEX

/*
 * Version1 - LAPACK              (lapack_cgehrd and lapack_cunghr)
 * Version2 - MAGMA without dT    (magma_cgehrd2 and lapack_cunghr)
 * Version3 - MAGMA with dT       (magma_cgehrd  and magma_cunghr)
 * Version4 - Multi-GPU magma_cgehrd_m with T on CPU, copied to dT, single-GPU magma_cunghr
 * Version5 - Multi-GPU magma_cgehrd_m with T on CPU, multi-GPU magma_cunghr_m
 */
#define Version5

/*
 * TREVC version 1 - LAPACK
 * TREVC version 2 - new blocked LAPACK
 * TREVC version 3 - blocked, single-threaded (MAGMA)
 * TREVC version 4 - blocked, multi-threaded  (MAGMA)
 * TREVC version 5 - blocked, multi-threaded, GPU (MAGMA)
 */
#define TREVC_VERSION 4

/**
    Purpose
    -------
    CGEEV computes for an N-by-N complex nonsymmetric matrix A, the
    eigenvalues and, optionally, the left and/or right eigenvectors.

    The right eigenvector v(j) of A satisfies
                     A * v(j) = lambda(j) * v(j)
    where lambda(j) is its eigenvalue.
    The left eigenvector u(j) of A satisfies
                  u(j)**H * A = lambda(j) * u(j)**H
    where u(j)**H denotes the conjugate transpose of u(j).

    The computed eigenvectors are normalized to have Euclidean norm
    equal to 1 and largest component real.

    Arguments
    ---------
    @param[in]
    jobvl   magma_vec_t
      -     = MagmaNoVec: left eigenvectors of A are not computed;
      -     = MagmaVec:   left eigenvectors of are computed.

    @param[in]
    jobvr   magma_vec_t
      -     = MagmaNoVec: right eigenvectors of A are not computed;
      -     = MagmaVec:   right eigenvectors of A are computed.

    @param[in]
    n       INTEGER
            The order of the matrix A. N >= 0.

    @param[in,out]
    A       COMPLEX array, dimension (LDA,N)
            On entry, the N-by-N matrix A.
            On exit, A has been overwritten.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    w       COMPLEX array, dimension (N)
            W contains the computed eigenvalues.

    @param[out]
    VL      COMPLEX array, dimension (LDVL,N)
            If JOBVL = MagmaVec, the left eigenvectors u(j) are stored one
            after another in the columns of VL, in the same order
            as their eigenvalues.
            If JOBVL = MagmaNoVec, VL is not referenced.
            u(j) = VL(:,j), the j-th column of VL.

    @param[in]
    ldvl    INTEGER
            The leading dimension of the array VL.  LDVL >= 1; if
            JOBVL = MagmaVec, LDVL >= N.

    @param[out]
    VR      COMPLEX array, dimension (LDVR,N)
            If JOBVR = MagmaVec, the right eigenvectors v(j) are stored one
            after another in the columns of VR, in the same order
            as their eigenvalues.
            If JOBVR = MagmaNoVec, VR is not referenced.
            v(j) = VR(:,j), the j-th column of VR.

    @param[in]
    ldvr    INTEGER
            The leading dimension of the array VR.  LDVR >= 1; if
            JOBVR = MagmaVec, LDVR >= N.

    @param[out]
    work    (workspace) COMPLEX array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= (1+nb)*N.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param
    rwork   (workspace) REAL array, dimension (2*N)

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if INFO = i, the QR algorithm failed to compute all the
                  eigenvalues, and no eigenvectors have been computed;
                  elements and i+1:N of W contain eigenvalues which have
                  converged.

    @ingroup magma_cgeev_driver
    ********************************************************************/
extern "C" magma_int_t
magma_cgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    #ifdef COMPLEX
    magmaFloatComplex *w,
    #else
    float *wr, float *wi,
    #endif
    magmaFloatComplex *VL, magma_int_t ldvl,
    magmaFloatComplex *VR, magma_int_t ldvr,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork,
    #endif
    magma_int_t *info )
{
    #define VL(i,j)  (VL + (i) + (j)*ldvl)
    #define VR(i,j)  (VR + (i) + (j)*ldvr)
    
    const magma_int_t ione  = 1;
    const magma_int_t izero = 0;
    
    float d__1, d__2;
    magmaFloatComplex tmp;
    float scl;
    float dum[1], eps;
    float anrm, cscale, bignum, smlnum;
    magma_int_t i, k, ilo, ihi;
    magma_int_t ibal, ierr, itau, iwrk, nout, liwrk, nb;
    magma_int_t scalea, minwrk, irwork, lquery, wantvl, wantvr, select[1];

    magma_side_t side = MagmaRight;

    irwork = 0;
    *info = 0;
    lquery = (lwork == -1);
    wantvl = (jobvl == MagmaVec);
    wantvr = (jobvr == MagmaVec);
    if (! wantvl && jobvl != MagmaNoVec) {
        *info = -1;
    } else if (! wantvr && jobvr != MagmaNoVec) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < max(1,n)) {
        *info = -5;
    } else if ( (ldvl < 1) || (wantvl && (ldvl < n))) {
        *info = -8;
    } else if ( (ldvr < 1) || (wantvr && (ldvr < n))) {
        *info = -10;
    }

    /* Compute workspace */
    nb = magma_get_cgehrd_nb( n );
    if (*info == 0) {
        minwrk = (1+nb)*n;
        work[0] = MAGMA_C_MAKE( minwrk, 0 );

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
    if (n == 0) {
        return *info;
    }
    
    #if defined(Version3) || defined(Version4) || defined(Version5)
    magmaFloatComplex *dT;
    if (MAGMA_SUCCESS != magma_cmalloc( &dT, nb*n )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    #endif
    #if defined(Version4) || defined(Version5)
    magmaFloatComplex *T;
    if (MAGMA_SUCCESS != magma_cmalloc_cpu( &T, nb*n )) {
        magma_free( dT );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    #endif

    /* Get machine constants */
    eps    = lapackf77_slamch( "P" );
    smlnum = lapackf77_slamch( "S" );
    bignum = 1. / smlnum;
    lapackf77_slabad( &smlnum, &bignum );
    smlnum = magma_ssqrt( smlnum ) / eps;
    bignum = 1. / smlnum;

    /* Scale A if max element outside range [SMLNUM,BIGNUM] */
    anrm = lapackf77_clange( "M", &n, &n, A, &lda, dum );
    scalea = 0;
    if (anrm > 0. && anrm < smlnum) {
        scalea = 1;
        cscale = smlnum;
    } else if (anrm > bignum) {
        scalea = 1;
        cscale = bignum;
    }
    if (scalea) {
        lapackf77_clascl( "G", &izero, &izero, &anrm, &cscale, &n, &n, A, &lda, &ierr );
    }

    /* Balance the matrix
     * (CWorkspace: none)
     * (RWorkspace: need N)
     *  - this space is reserved until after gebak */
    ibal = 0;
    lapackf77_cgebal( "B", &n, A, &lda, &ilo, &ihi, &rwork[ibal], &ierr );

    /* Reduce to upper Hessenberg form
     * (CWorkspace: need 2*N, prefer N + N*NB)
     * (RWorkspace: N)
     *  - including N reserved for gebal/gebak, unused by cgehrd */
    itau = 0;
    iwrk = itau + n;
    liwrk = lwork - iwrk;

    #if defined(Version1)
        // Version 1 - LAPACK
        lapackf77_cgehrd( &n, &ilo, &ihi, A, &lda,
                          &work[itau], &work[iwrk], &liwrk, &ierr );
    #elif defined(Version2)
        // Version 2 - LAPACK consistent HRD
        magma_cgehrd2( n, ilo, ihi, A, lda,
                       &work[itau], &work[iwrk], liwrk, &ierr );
    #elif defined(Version3)
        // Version 3 - LAPACK consistent MAGMA HRD + T matrices stored,
        magma_cgehrd( n, ilo, ihi, A, lda,
                      &work[itau], &work[iwrk], liwrk, dT, &ierr );
    #elif defined(Version4) || defined(Version5)
        // Version 4 - Multi-GPU, T on host
        magma_cgehrd_m( n, ilo, ihi, A, lda,
                        &work[itau], &work[iwrk], liwrk, T, &ierr );
        magma_csetmatrix( nb, n, T, nb, dT, nb );
    #endif

    if (wantvl) {
        /* Want left eigenvectors
         * Copy Householder vectors to VL */
        side = MagmaLeft;
        lapackf77_clacpy( MagmaLowerStr, &n, &n, A, &lda, VL, &ldvl );

        /* Generate unitary matrix in VL
         * (CWorkspace: need 2*N-1, prefer N + (N-1)*NB)
         * (RWorkspace: N)
         *  - including N reserved for gebal/gebak, unused by cunghr */
        #if defined(Version1) || defined(Version2)
            // Version 1 & 2 - LAPACK
            lapackf77_cunghr( &n, &ilo, &ihi, VL, &ldvl, &work[itau],
                              &work[iwrk], &liwrk, &ierr );
        #elif defined(Version3) || defined(Version4)
            // Version 3 - LAPACK consistent MAGMA HRD + T matrices stored
            magma_cunghr( n, ilo, ihi, VL, ldvl, &work[itau], dT, nb, &ierr );
        #elif defined(Version5)
            // Version 5 - Multi-GPU, T on host
            magma_cunghr_m( n, ilo, ihi, VL, ldvl, &work[itau], T, nb, &ierr );
        #endif

        /* Perform QR iteration, accumulating Schur vectors in VL
         * (CWorkspace: need 1, prefer HSWORK (see comments) )
         * (RWorkspace: N)
         *  - including N reserved for gebal/gebak, unused by chseqr */
        iwrk = itau;
        liwrk = lwork - iwrk;
        lapackf77_chseqr( "S", "V", &n, &ilo, &ihi, A, &lda, w,
                          VL, &ldvl, &work[iwrk], &liwrk, info );

        if (wantvr) {
            /* Want left and right eigenvectors
             * Copy Schur vectors to VR */
            side = MagmaBothSides;
            lapackf77_clacpy( "F", &n, &n, VL, &ldvl, VR, &ldvr );
        }
    }
    else if (wantvr) {
        /* Want right eigenvectors
         * Copy Householder vectors to VR */
        side = MagmaRight;
        lapackf77_clacpy( "L", &n, &n, A, &lda, VR, &ldvr );

        /* Generate unitary matrix in VR
         * (CWorkspace: need 2*N-1, prefer N + (N-1)*NB)
         * (RWorkspace: N)
         *  - including N reserved for gebal/gebak, unused by cunghr */
        #if defined(Version1) || defined(Version2)
            // Version 1 & 2 - LAPACK
            lapackf77_cunghr( &n, &ilo, &ihi, VR, &ldvr, &work[itau],
                              &work[iwrk], &liwrk, &ierr );
        #elif defined(Version3) || defined(Version4)
            // Version 3 - LAPACK consistent MAGMA HRD + T matrices stored
            magma_cunghr( n, ilo, ihi, VR, ldvr, &work[itau], dT, nb, &ierr );
        #elif defined(Version5)
            // Version 5 - Multi-GPU, T on host
            magma_cunghr_m( n, ilo, ihi, VR, ldvr, &work[itau], T, nb, &ierr );
        #endif

        /* Perform QR iteration, accumulating Schur vectors in VR
         * (CWorkspace: need 1, prefer HSWORK (see comments) )
         * (RWorkspace: N)
         *  - including N reserved for gebal/gebak, unused by chseqr */
        iwrk = itau;
        liwrk = lwork - iwrk;
        lapackf77_chseqr( "S", "V", &n, &ilo, &ihi, A, &lda, w,
                          VR, &ldvr, &work[iwrk], &liwrk, info );
    }
    else {
        /* Compute eigenvalues only
         * (CWorkspace: need 1, prefer HSWORK (see comments) )
         * (RWorkspace: N)
         *  - including N reserved for gebal/gebak, unused by chseqr */
        iwrk = itau;
        liwrk = lwork - iwrk;
        lapackf77_chseqr( "E", "N", &n, &ilo, &ihi, A, &lda, w,
                          VR, &ldvr, &work[iwrk], &liwrk, info );
    }

    /* If INFO > 0 from CHSEQR, then quit */
    if (*info > 0) {
        goto CLEANUP;
    }

    if (wantvl || wantvr) {
        /* Compute left and/or right eigenvectors
         * (CWorkspace: need 2*N)
         * (RWorkspace: need 2*N)
         *  - including N reserved for gebal/gebak, unused by ctrevc */
        irwork = ibal + n;
        #if TREVC_VERSION == 1
        lapackf77_ctrevc( lapack_side_const(side), "B", select, &n, A, &lda, VL, &ldvl,
                          VR, &ldvr, &n, &nout, &work[iwrk], &rwork[irwork], &ierr );
        #elif TREVC_VERSION == 2
        liwrk = lwork - iwrk;
        lapackf77_ctrevc3( lapack_side_const(side), "B", select, &n, A, &lda, VL, &ldvl,
                           VR, &ldvr, &n, &nout, &work[iwrk], &liwrk, &rwork[irwork], &ierr );
        #elif TREVC_VERSION == 3
        magma_ctrevc3( side, MagmaBacktransVec, select, n, A, lda, VL, ldvl,
                       VR, ldvr, n, &nout, &work[iwrk], liwrk, &rwork[irwork], &ierr );
        #elif TREVC_VERSION == 4
        magma_ctrevc3_mt( side, MagmaBacktransVec, select, n, A, lda, VL, ldvl,
                          VR, ldvr, n, &nout, &work[iwrk], liwrk, &rwork[irwork], &ierr );
        #elif TREVC_VERSION == 5
        magma_ctrevc3_mt_gpu( side, MagmaBacktransVec, select, n, A, lda, VL, ldvl,
                              VR, ldvr, n, &nout, &work[iwrk], liwrk, &rwork[irwork], &ierr );
        #else
        #error Unknown TREVC_VERSION
        #endif
    }

    if (wantvl) {
        /* Undo balancing of left eigenvectors
         * (CWorkspace: none)
         * (RWorkspace: need N) */
        lapackf77_cgebak( "B", "L", &n, &ilo, &ihi, &rwork[ibal], &n,
                          VL, &ldvl, &ierr );

        /* Normalize left eigenvectors and make largest component real */
        for (i = 0; i < n; ++i) {
            scl = 1. / magma_cblas_scnrm2( n, VL(0,i), 1 );
            blasf77_csscal( &n, &scl, VL(0,i), &ione );
            for (k = 0; k < n; ++k) {
                /* Computing 2nd power */
                d__1 = MAGMA_C_REAL( *VL(k,i) );
                d__2 = MAGMA_C_IMAG( *VL(k,i) );
                rwork[irwork + k] = d__1*d__1 + d__2*d__2;
            }
            k = blasf77_isamax( &n, &rwork[irwork], &ione ) - 1;  // subtract 1; k is 0-based
            tmp = MAGMA_C_CNJG( *VL(k,i) ) / magma_ssqrt( rwork[irwork + k] );
            blasf77_cscal( &n, &tmp, VL(0,i), &ione );
            *VL(k,i) = MAGMA_C_MAKE( MAGMA_C_REAL( *VL(k,i) ), 0 );
        }
    }

    if (wantvr) {
        /* Undo balancing of right eigenvectors
         * (CWorkspace: none)
         * (RWorkspace: need N) */
        lapackf77_cgebak( "B", "R", &n, &ilo, &ihi, &rwork[ibal], &n,
                          VR, &ldvr, &ierr );

        /* Normalize right eigenvectors and make largest component real */
        for (i = 0; i < n; ++i) {
            scl = 1. / magma_cblas_scnrm2( n, VR(0,i), 1 );
            blasf77_csscal( &n, &scl, VR(0,i), &ione );
            for (k = 0; k < n; ++k) {
                /* Computing 2nd power */
                d__1 = MAGMA_C_REAL( *VR(k,i) );
                d__2 = MAGMA_C_IMAG( *VR(k,i) );
                rwork[irwork + k] = d__1*d__1 + d__2*d__2;
            }
            k = blasf77_isamax( &n, &rwork[irwork], &ione ) - 1;  // subtract 1; k is 0-based
            tmp = MAGMA_C_CNJG( *VR(k,i) ) / magma_ssqrt( rwork[irwork + k] );
            blasf77_cscal( &n, &tmp, VR(0,i), &ione );
            *VR(k,i) = MAGMA_C_MAKE( MAGMA_C_REAL( *VR(k,i) ), 0 );
        }
    }

CLEANUP:
    /* Undo scaling if necessary */
    if (scalea) {
        // converged eigenvalues, stored in WR[i+1:n] and WI[i+1:n] for i = INFO
        magma_int_t nval = n - (*info);
        magma_int_t ld   = max( nval, 1 );
        lapackf77_clascl( "G", &izero, &izero, &cscale, &anrm, &nval, &ione, w + (*info), &ld, &ierr );
        if (*info > 0) {
            // first ilo columns were already upper triangular,
            // so the corresponding eigenvalues are also valid.
            nval = ilo - 1;
            lapackf77_clascl( "G", &izero, &izero, &cscale, &anrm, &nval, &ione, w, &n, &ierr );
        }
    }

    #if defined(Version3) || defined(Version4) || defined(Version5)
    magma_free( dT );
    #endif
    #if defined(Version4) || defined(Version5)
    magma_free_cpu( T );
    #endif
    
    work[0] = MAGMA_C_MAKE( (float) minwrk, 0. );  // TODO use optwrk as in dgeev

    return *info;
} /* magma_cgeev */
