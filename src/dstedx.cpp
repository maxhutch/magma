/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
       
       @author Raffaele Solca
       
       @precisions normal d -> s
*/
#include "common_magma.h"

/**
    Purpose
    -------
    DSTEDX computes some eigenvalues and, optionally, eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.  See DLAEX3 for details.

    Arguments
    ---------
    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                             will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.

    @param[in]
    n       INTEGER
            The dimension of the symmetric tridiagonal matrix.  N >= 0.

    @param[in]
    vl      DOUBLE PRECISION
    @param[in]
    vu      DOUBLE PRECISION
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

    @param[in,out]
    d       DOUBLE PRECISION array, dimension (N)
            On entry, the diagonal elements of the tridiagonal matrix.
            On exit, if INFO = 0, the eigenvalues in ascending order.

    @param[in,out]
    e       DOUBLE PRECISION array, dimension (N-1)
            On entry, the subdiagonal elements of the tridiagonal matrix.
            On exit, E has been destroyed.

    @param[in,out]
    Z       DOUBLE PRECISION array, dimension (LDZ,N)
            On exit, if INFO = 0, Z contains the orthonormal eigenvectors
            of the symmetric tridiagonal matrix.

    @param[in]
    ldz     INTEGER
            The leading dimension of the array Z. LDZ >= max(1,N).

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (LWORK)
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If N > 1 then LWORK >= ( 1 + 4*N + N**2 ).
            Note that  if N is less than or
            equal to the minimum divide size, usually 25, then LWORK need
            only be max(1,2*(N-1)).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    @param[out]
    iwork   (workspace) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK[0] returns the optimal LIWORK.

    @param[in]
    liwork  INTEGER
            The dimension of the array IWORK.
            LIWORK >= ( 3 + 5*N ).
            Note that if N is less than or
            equal to the minimum divide size, usually 25, then LIWORK
            need only be 1.
    \n
            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal size of the IWORK array,
            returns this value as the first entry of the IWORK array, and
            no error message related to LIWORK is issued by XERBLA.

    @param
    dwork  (workspace) DOUBLE PRECISION array, dimension (3*N*N/2+3*N)

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  The algorithm failed to compute an eigenvalue while
                  working on the submatrix lying in rows and columns
                  INFO/(N+1) through mod(INFO,N+1).

    Further Details
    ---------------
    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    @ingroup magma_dsyev_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dstedx(
    magma_range_t range, magma_int_t n, double vl, double vu,
    magma_int_t il, magma_int_t iu, double *d, double *e,
    double *Z, magma_int_t ldz,
    double *work, magma_int_t lwork,
    magma_int_t *iwork, magma_int_t liwork,
    magmaDouble_ptr dwork,
    magma_int_t *info)
{
#define Z(i_,j_) (Z + (i_) + (j_)*ldz)

    double d_zero = 0.;
    double d_one  = 1.;
    magma_int_t izero = 0;
    magma_int_t ione = 1;


    magma_int_t alleig, indeig, valeig, lquery;
    magma_int_t i, j, k, m;
    magma_int_t liwmin, lwmin;
    magma_int_t start, end, smlsiz;
    double eps, orgnrm, p, tiny;

    // Test the input parameters.

    alleig = (range == MagmaRangeAll);
    valeig = (range == MagmaRangeV);
    indeig = (range == MagmaRangeI);
    lquery = (lwork == -1 || liwork == -1);

    *info = 0;

    if (! (alleig || valeig || indeig)) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldz < max(1,n)) {
        *info = -10;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -4;
            }
        } else if (indeig) {
            if (il < 1 || il > max(1,n)) {
                *info = -5;
            } else if (iu < min(n,il) || iu > n) {
                *info = -6;
            }
        }
    }

    if (*info == 0) {
        // Compute the workspace requirements

        smlsiz = magma_get_smlsize_divideconquer();
        if ( n <= 1 ) {
            lwmin = 1;
            liwmin = 1;
        } else {
            lwmin = 1 + 4*n + n*n;
            liwmin = 3 + 5*n;
        }

        work[0] = lwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && ! lquery) {
            *info = -12;
        } else if (liwork < liwmin && ! lquery) {
            *info = -14;
        }
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info));
        return *info;
    } else if (lquery) {
        return *info;
    }

    // Quick return if possible
    if (n == 0)
        return *info;
    if (n == 1) {
        *Z = 1.;
        return *info;
    }

    /* determine the number of threads *///not needed here to be checked Azzam
    //magma_int_t threads = magma_get_parallel_numthreads();
    //magma_int_t mklth   = magma_get_lapack_numthreads();
    //magma_set_lapack_numthreads(mklth);

#ifdef ENABLE_DEBUG
    //printf("  D&C is using %d threads\n", threads);
#endif

    // If N is smaller than the minimum divide size (SMLSIZ+1), then
    // solve the problem with another solver.

    if (n < smlsiz) {
        lapackf77_dsteqr("I", &n, d, e, Z, &ldz, work, info);
    } else {
        lapackf77_dlaset("F", &n, &n, &d_zero, &d_one, Z, &ldz);

        //Scale.
        orgnrm = lapackf77_dlanst("M", &n, d, e);

        if (orgnrm == 0) {
            work[0]  = lwmin;
            iwork[0] = liwmin;
            return *info;
        }

        eps = lapackf77_dlamch( "Epsilon" );

        if (alleig) {
            start = 0;
            while ( start < n ) {

                // Let FINISH be the position of the next subdiagonal entry
                // such that E( END ) <= TINY or FINISH = N if no such
                // subdiagonal exists.  The matrix identified by the elements
                // between START and END constitutes an independent
                // sub-problem.

                for (end = start+1; end < n; ++end) {
                    tiny = eps * sqrt( MAGMA_D_ABS(d[end-1]*d[end]));
                    if (MAGMA_D_ABS(e[end-1]) <= tiny)
                        break;
                }

                // (Sub) Problem determined.  Compute its size and solve it.

                m = end - start;
                if (m == 1) {
                    start = end;
                    continue;
                }
                if (m > smlsiz) {
                    // Scale
                    orgnrm = lapackf77_dlanst("M", &m, &d[start], &e[start]);
                    lapackf77_dlascl("G", &izero, &izero, &orgnrm, &d_one, &m, &ione, &d[start], &m, info);
                    magma_int_t mm = m-1;
                    lapackf77_dlascl("G", &izero, &izero, &orgnrm, &d_one, &mm, &ione, &e[start], &mm, info);

                    magma_dlaex0( m, &d[start], &e[start], Z(start, start), ldz, work, iwork, dwork, MagmaRangeAll, vl, vu, il, iu, info);

                    if ( *info != 0) {
                        return *info;
                    }

                    // Scale Back
                    lapackf77_dlascl("G", &izero, &izero, &d_one, &orgnrm, &m, &ione, &d[start], &m, info);
                } else {
                    lapackf77_dsteqr( "I", &m, &d[start], &e[start], Z(start, start), &ldz, work, info);
                    if (*info != 0) {
                        *info = (start+1) *(n+1) + end;
                    }
                }

                start = end;
            }


            // If the problem split any number of times, then the eigenvalues
            // will not be properly ordered.  Here we permute the eigenvalues
            // (and the associated eigenvectors) into ascending order.

            if (m < n) {
                // Use Selection Sort to minimize swaps of eigenvectors
                for (i = 1; i < n; ++i) {
                    k = i-1;
                    p = d[i-1];
                    for (j = i; j < n; ++j) {
                        if (d[j] < p) {
                            k = j;
                            p = d[j];
                        }
                    }
                    if (k != i-1) {
                        d[k] = d[i-1];
                        d[i-1] = p;
                        blasf77_dswap(&n, Z(0,i-1), &ione, Z(0,k), &ione);
                    }
                }
            }
        } else {
            // Scale
            lapackf77_dlascl("G", &izero, &izero, &orgnrm, &d_one, &n, &ione, d, &n, info);
            magma_int_t nm = n-1;
            lapackf77_dlascl("G", &izero, &izero, &orgnrm, &d_one, &nm, &ione, e, &nm, info);

            magma_dlaex0( n, d, e, Z, ldz, work, iwork, dwork, range, vl, vu, il, iu, info);

            if ( *info != 0) {
                return *info;
            }

            // Scale Back
            lapackf77_dlascl("G", &izero, &izero, &d_one, &orgnrm, &n, &ione, d, &n, info);
        }
    }

    work[0]  = lwmin;
    iwork[0] = liwmin;

    return *info;
} /* magma_dstedx */
