/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
    
       @author Raffaele Solca
    
       @generated from zstedx.cpp normal z -> c, Fri Jan 30 19:00:17 2015

*/
#include "common_magma.h"

/**
    Purpose
    -------
    CSTEDX computes some eigenvalues and eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none. See SLAEX3 for details.

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

    @param[in,out]
    d       REAL array, dimension (N)
            On entry, the diagonal elements of the tridiagonal matrix.
            On exit, if INFO = 0, the eigenvalues in ascending order.

    @param[in,out]
    e       REAL array, dimension (N-1)
            On entry, the subdiagonal elements of the tridiagonal matrix.
            On exit, E has been destroyed.

    @param[out]
    Z       COMPLEX array, dimension (LDZ,N)
            On exit, if INFO = 0, Z contains the orthonormal eigenvectors
            of the symmetric tridiagonal matrix.

    @param[in]
    ldz     INTEGER
            The leading dimension of the array Z. LDZ >= max(1,N).

    @param[out]
    rwork   (workspace) REAL array, dimension (LRWORK)
            On exit, if INFO = 0, RWORK[0] returns the optimal LRWORK.

    @param[in]
    lrwork  INTEGER
            The dimension of the array RWORK.
            LRWORK >= 1 + 4*N + 2*N**2.
            Note that if N is less than or
            equal to the minimum divide size, usually 25, then LRWORK
            need only be max(1,2*(N-1)).
    \n
            If LRWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    @param[out]
    iwork   (workspace) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK[0] returns the optimal LIWORK.

    @param[in]
    liwork  INTEGER
            The dimension of the array IWORK.
            LIWORK >= 3 + 5*N .
            Note that if N is less than or
            equal to the minimum divide size, usually 25, then LIWORK
            need only be 1.
    \n
            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    @param
    dwork  (workspace) REAL array, dimension (3*N*N/2+3*N)

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

    @ingroup magma_cheev_comp
    ********************************************************************/
extern "C" magma_int_t
magma_cstedx(
    magma_range_t range, magma_int_t n, float vl, float vu,
    magma_int_t il, magma_int_t iu, float *d, float *e,
    magmaFloatComplex *Z, magma_int_t ldz,
    float *rwork, magma_int_t lrwork,
    magma_int_t *iwork, magma_int_t liwork,
    magmaFloat_ptr dwork,
    magma_int_t *info)
{
    magma_int_t alleig, indeig, valeig, lquery;
    magma_int_t i, j, smlsiz;
    magma_int_t liwmin, lrwmin;

    alleig = (range == MagmaRangeAll);
    valeig = (range == MagmaRangeV);
    indeig = (range == MagmaRangeI);
    lquery = (lrwork == -1 || liwork == -1);

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
            lrwmin = 1;
            liwmin = 1;
        } else {
            lrwmin = 1 + 4*n + 2*n*n;
            liwmin = 3 + 5*n;
        }

        rwork[0] = lrwmin;
        iwork[0] = liwmin;

        if (lrwork < lrwmin && ! lquery) {
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
        *Z = MAGMA_C_MAKE( 1, 0 );
        return *info;
    }

    // If N is smaller than the minimum divide size (SMLSIZ+1), then
    // solve the problem with another solver.

    if (n < smlsiz) {
        lapackf77_csteqr("I", &n, d, e, Z, &ldz, rwork, info);
    } else {
        // We simply call SSTEDX instead.
        magma_sstedx(range, n, vl, vu, il, iu, d, e, rwork, n,
                     rwork+n*n, lrwork-n*n, iwork, liwork, dwork, info);

        for (j=0; j < n; ++j)
            for (i=0; i < n; ++i) {
                *(Z+i+ldz*j) = MAGMA_C_MAKE( *(rwork+i+n*j), 0 );
            }
    }

    rwork[0] = lrwmin;
    iwork[0] = liwmin;

    return *info;
} /* magma_cstedx */
