/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @precisions normal d -> s
       @author Stan Tomov
       @author Mark Gates
*/
#include "common_magma.h"
#include <cblas.h>

#define PRECISION_d

/*
 * Version1 - LAPACK              (lapack_zgehrd and lapack_zunghr)
 * Version2 - MAGMA without dT    (magma_zgehrd2 and lapack_zunghr)
 * Version3 - MAGMA with dT       (magma_zgehrd  and magma_zunghr)
 * Version4 - Multi-GPU magma_zgehrd_m with T on CPU, copied to dT, single-GPU magma_zunghr
 * Version5 - Multi-GPU magma_zgehrd_m with T on CPU, multi-GPU magma_zunghr_m
 */
#define Version5

/**
    Purpose
    -------
    DGEEV computes for an N-by-N real nonsymmetric matrix A, the
    eigenvalues and, optionally, the left and/or right eigenvectors.

    The right eigenvector v(j) of A satisfies
                     A * v(j) = lambda(j) * v(j)
    where lambda(j) is its eigenvalue.
    The left eigenvector u(j) of A satisfies
                  u(j)**T * A = lambda(j) * u(j)**T
    where u(j)**T denotes the transpose of u(j).

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
    A       DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the N-by-N matrix A.
            On exit, A has been overwritten.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    wr      DOUBLE PRECISION array, dimension (N)
    @param[out]
    wi      DOUBLE PRECISION array, dimension (N)
            WR and WI contain the real and imaginary parts,
            respectively, of the computed eigenvalues.  Complex
            conjugate pairs of eigenvalues appear consecutively
            with the eigenvalue having the positive imaginary part
            first.

    @param[out]
    VL      DOUBLE PRECISION array, dimension (LDVL,N)
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
    VR      DOUBLE PRECISION array, dimension (LDVR,N)
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
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= (1+nb)*N.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if INFO = i, the QR algorithm failed to compute all the
                  eigenvalues, and no eigenvectors have been computed;
                  elements and i+1:N of W contain eigenvalues which have
                  converged.

    @ingroup magma_dgeev_driver
    ********************************************************************/
extern "C" magma_int_t
magma_dgeev_m(
    magma_vec_t jobvl, magma_vec_t jobvr, magma_int_t n,
    double *A, magma_int_t lda,
    double *wr, double *wi,
    double *VL, magma_int_t ldvl,
    double *VR, magma_int_t ldvr,
    double *work, magma_int_t lwork,
    magma_int_t *info )
{
    #define VL(i,j)  (VL + (i) + (j)*ldvl)
    #define VR(i,j)  (VR + (i) + (j)*ldvr)
    
    magma_int_t c_one = 1;
    magma_int_t c_zero = 0;
    
    double d__1, d__2;
    double r, cs, sn, scl;
    double dum[1], eps;
    double anrm, cscale, bignum, smlnum;
    magma_int_t i, k, ilo, ihi;
    magma_int_t ibal, ierr, itau, iwrk, nout, liwrk, i__1, i__2, nb;
    magma_int_t scalea, minwrk, lquery, wantvl, wantvr, select[1];
    
    const char* side_  = NULL;
    
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
        *info = -9;
    } else if ( (ldvr < 1) || (wantvr && (ldvr < n))) {
        *info = -11;
    }

    /* Compute workspace */
    nb = magma_get_dgehrd_nb( n );
    if (*info == 0) {
        minwrk = (2+nb)*n;
        work[0] = MAGMA_D_MAKE( (double) minwrk, 0. );
        
        if (lwork < minwrk && ! lquery) {
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

    /* Quick return if possible */
    if (n == 0) {
        return *info;
    }
   
    #if defined(Version3) || defined(Version4) || defined(Version5)
    double *dT;
    if (MAGMA_SUCCESS != magma_dmalloc( &dT, nb*n )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    #endif
    #if defined(Version4) || defined(Version5)
    double *T;
    if (MAGMA_SUCCESS != magma_dmalloc_cpu( &T, nb*n )) {
        magma_free( dT );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    #endif

    /* Get machine constants */
    eps    = lapackf77_dlamch( "P" );
    smlnum = lapackf77_dlamch( "S" );
    bignum = 1. / smlnum;
    lapackf77_dlabad( &smlnum, &bignum );
    smlnum = magma_dsqrt( smlnum ) / eps;
    bignum = 1. / smlnum;

    /* Scale A if max element outside range [SMLNUM,BIGNUM] */
    anrm = lapackf77_dlange( "M", &n, &n, A, &lda, dum );
    scalea = 0;
    if (anrm > 0. && anrm < smlnum) {
        scalea = 1;
        cscale = smlnum;
    } else if (anrm > bignum) {
        scalea = 1;
        cscale = bignum;
    }
    if (scalea) {
        lapackf77_dlascl( "G", &c_zero, &c_zero, &anrm, &cscale, &n, &n, A, &lda, &ierr );
    }

    /* Balance the matrix
     * (Workspace: need N) */
    ibal = 0;
    lapackf77_dgebal( "B", &n, A, &lda, &ilo, &ihi, &work[ibal], &ierr );

    /* Reduce to upper Hessenberg form
     * (Workspace: need 3*N, prefer 2*N + N*NB) */
    itau = ibal + n;
    iwrk = itau + n;
    liwrk = lwork - iwrk;

    #if defined(Version1)
        // Version 1 - LAPACK
        lapackf77_dgehrd( &n, &ilo, &ihi, A, &lda,
                          &work[itau], &work[iwrk], &liwrk, &ierr );
    #elif defined(Version2)
        // Version 2 - LAPACK consistent HRD
        magma_dgehrd2( n, ilo, ihi, A, lda,
                       &work[itau], &work[iwrk], &liwrk, &ierr );
    #elif defined(Version3)
        // Version 3 - LAPACK consistent MAGMA HRD + matrices T stored,
        magma_dgehrd( n, ilo, ihi, A, lda,
                      &work[itau], &work[iwrk], liwrk, dT, &ierr );
    #elif defined(Version4) || defined(Version5)
        // Version 4 - Multi-GPU, T on host
        magma_dgehrd_m( n, ilo, ihi, A, lda,
                        &work[itau], &work[iwrk], liwrk, T, &ierr );
        magma_dsetmatrix( nb, n, T, nb, dT, nb );
    #endif

    if (wantvl) {
        /* Want left eigenvectors
         * Copy Householder vectors to VL */
        side_ = "Left";
        lapackf77_dlacpy( MagmaLowerStr, &n, &n,
                          A, &lda, VL, &ldvl );

        /* Generate orthogonal matrix in VL
         * (Workspace: need 3*N-1, prefer 2*N + (N-1)*NB) */
        #if defined(Version1) || defined(Version2)
            // Version 1 & 2 - LAPACK
            lapackf77_dorghr( &n, &ilo, &ihi, VL, &ldvl, &work[itau],
                              &work[iwrk], &liwrk, &ierr );
        #elif defined(Version3) || defined(Version4)
            // Version 3 - LAPACK consistent MAGMA HRD + matrices T stored
            magma_dorghr( n, ilo, ihi, VL, ldvl, &work[itau], dT, nb, &ierr );
        #elif defined(Version5)
            // Version 5 - Multi-GPU, T on host
            magma_dorghr_m( n, ilo, ihi, VL, ldvl, &work[itau], T, nb, &ierr );
        #endif

        /* Perform QR iteration, accumulating Schur vectors in VL
         * (Workspace: need N+1, prefer N+HSWORK (see comments) ) */
        iwrk = itau;
        liwrk = lwork - iwrk;
        lapackf77_dhseqr( "S", "V", &n, &ilo, &ihi, A, &lda, wr, wi,
                          VL, &ldvl, &work[iwrk], &liwrk, info );

        if (wantvr) {
            /* Want left and right eigenvectors
             * Copy Schur vectors to VR */
            side_ = "Both";
            lapackf77_dlacpy( "F", &n, &n, VL, &ldvl, VR, &ldvr );
        }
    }
    else if (wantvr) {
        /* Want right eigenvectors
         * Copy Householder vectors to VR */
        side_ = "Right";
        lapackf77_dlacpy( "L", &n, &n, A, &lda, VR, &ldvr );

        /* Generate orthogonal matrix in VR
         * (Workspace: need 3*N-1, prefer 2*N + (N-1)*NB) */
        #if defined(Version1) || defined(Version2)
            // Version 1 & 2 - LAPACK
            lapackf77_dorghr( &n, &ilo, &ihi, VR, &ldvr, &work[itau],
                              &work[iwrk], &liwrk, &ierr );
        #elif defined(Version3) || defined(Version4)
            // Version 3 - LAPACK consistent MAGMA HRD + matrices T stored
            magma_dorghr( n, ilo, ihi, VR, ldvr, &work[itau], dT, nb, &ierr );
        #elif defined(Version5)
            // Version 5 - Multi-GPU, T on host
            magma_dorghr_m( n, ilo, ihi, VR, ldvr, &work[itau], T, nb, &ierr );
        #endif

        /* Perform QR iteration, accumulating Schur vectors in VR
         * (Workspace: need N+1, prefer N+HSWORK (see comments) ) */
        iwrk = itau;
        liwrk = lwork - iwrk;
        lapackf77_dhseqr( "S", "V", &n, &ilo, &ihi, A, &lda, wr, wi,
                          VR, &ldvr, &work[iwrk], &liwrk, info );
    }
    else {
        /* Compute eigenvalues only
         * (Workspace: need N+1, prefer N+HSWORK (see comments) ) */
        iwrk = itau;
        liwrk = lwork - iwrk;
        lapackf77_dhseqr( "E", "N", &n, &ilo, &ihi, A, &lda, wr, wi,
                          VR, &ldvr, &work[iwrk], &liwrk, info );
    }

    /* If INFO > 0 from DHSEQR, then quit */
    if (*info > 0) {
        goto CLEANUP;
    }

    if (wantvl || wantvr) {
        /* Compute left and/or right eigenvectors
         * (Workspace: need 4*N) */
        lapackf77_dtrevc( side_, "B", select, &n, A, &lda, VL, &ldvl,
                          VR, &ldvr, &n, &nout, &work[iwrk], &ierr );
    }

    if (wantvl) {
        /* Undo balancing of left eigenvectors
         * (Workspace: need N) */
        lapackf77_dgebak( "B", "L", &n, &ilo, &ihi, &work[ibal], &n,
                          VL, &ldvl, &ierr );

        /* Normalize left eigenvectors and make largest component real */
        for (i = 0; i < n; ++i) {
            if ( wi[i] == 0. ) {
                scl = 1. / cblas_dnrm2( n, VL(0,i), 1 );
                cblas_dscal( n, scl, VL(0,i), 1 );
            }
            else if ( wi[i] > 0. ) {
                d__1 = cblas_dnrm2( n, VL(0,i),   1 );
                d__2 = cblas_dnrm2( n, VL(0,i+1), 1 );
                scl = 1. / lapackf77_dlapy2( &d__1, &d__2 );
                cblas_dscal( n, scl, VL(0,i),   1 );
                cblas_dscal( n, scl, VL(0,i+1), 1 );
                for (k = 0; k < n; ++k) {
                    /* Computing 2nd power */
                    d__1 = *VL(k,i);
                    d__2 = *VL(k,i+1);
                    work[iwrk + k] = d__1*d__1 + d__2*d__2;
                }
                k = cblas_idamax( n, &work[iwrk], 1 );
                lapackf77_dlartg( VL(k,i), VL(k,i+1), &cs, &sn, &r );
                cblas_drot( n, VL(0,i), 1, VL(0,i+1), 1, cs, sn );
                *VL(k,i+1) = 0.;
            }
        }
    }

    if (wantvr) {
        /* Undo balancing of right eigenvectors
         * (Workspace: need N) */
        lapackf77_dgebak( "B", "R", &n, &ilo, &ihi, &work[ibal], &n,
                          VR, &ldvr, &ierr );

        /* Normalize right eigenvectors and make largest component real */
        for (i = 0; i < n; ++i) {
            if ( wi[i] == 0. ) {
                scl = 1. / cblas_dnrm2( n, VR(0,i), 1 );
                cblas_dscal( n, scl, VR(0,i), 1 );
            }
            else if ( wi[i] > 0. ) {
                d__1 = cblas_dnrm2( n, VR(0,i),   1 );
                d__2 = cblas_dnrm2( n, VR(0,i+1), 1 );
                scl = 1. / lapackf77_dlapy2( &d__1, &d__2 );
                cblas_dscal( n, scl, VR(0,i),   1 );
                cblas_dscal( n, scl, VR(0,i+1), 1 );
                for (k = 0; k < n; ++k) {
                    /* Computing 2nd power */
                    d__1 = *VR(k,i);
                    d__2 = *VR(k,i+1);
                    work[iwrk + k] = d__1*d__1 + d__2*d__2;
                }
                k = cblas_idamax( n, &work[iwrk], 1 );
                lapackf77_dlartg( VR(k,i), VR(k,i+1), &cs, &sn, &r );
                cblas_drot( n, VR(0,i), 1, VR(0,i+1), 1, cs, sn );
                *VR(k,i+1) = 0.;
            }
        }
    }

CLEANUP:
    /* Undo scaling if necessary */
    if (scalea) {
        i__1 = n - (*info);
        i__2 = max( n - (*info), 1 );
        lapackf77_dlascl( "G", &c_zero, &c_zero, &cscale, &anrm, &i__1, &c_one,
                          wr + (*info), &i__2, &ierr );
        lapackf77_dlascl( "G", &c_zero, &c_zero, &cscale, &anrm, &i__1, &c_one,
                          wi + (*info), &i__2, &ierr );
        if (*info > 0) {
            i__1 = ilo - 1;
            lapackf77_dlascl( "G", &c_zero, &c_zero, &cscale, &anrm, &i__1, &c_one,
                              wr, &n, &ierr );
            lapackf77_dlascl( "G", &c_zero, &c_zero, &cscale, &anrm, &i__1, &c_one,
                              wi, &n, &ierr );
        }
    }

    #if defined(Version3) || defined(Version4) || defined(Version5)
    magma_free( dT );
    #endif
    #if defined(Version4) || defined(Version5)
    magma_free_cpu( T );
    #endif
    
    return *info;
} /* magma_dgeev */
