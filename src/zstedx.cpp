/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
    
       @author Raffaele Solca
    
       @precisions normal z -> c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zstedx(char range, magma_int_t n, double vl, double vu,
             magma_int_t il, magma_int_t iu, double* d, double* e,
             magmaDoubleComplex* z, magma_int_t ldz,
             double* rwork, magma_int_t lrwork,
             magma_int_t* iwork, magma_int_t liwork,
             double* dwork, magma_int_t* info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    ZSTEDX computes some eigenvalues and eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none. See DLAEX3 for details.

    Arguments
    =========
    RANGE   (input) CHARACTER*1
            = 'A': all eigenvalues will be found.
            = 'V': all eigenvalues in the half-open interval (VL,VU]
                   will be found.
            = 'I': the IL-th through IU-th eigenvalues will be found.

    N       (input) INTEGER
            The dimension of the symmetric tridiagonal matrix.  N >= 0.

    VL      (input) DOUBLE PRECISION
    VU      (input) DOUBLE PRECISION
            If RANGE='V', the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = 'A' or 'I'.

    IL      (input) INTEGER
    IU      (input) INTEGER
            If RANGE='I', the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = 'A' or 'V'.

    D       (input/output) DOUBLE PRECISION array, dimension (N)
            On entry, the diagonal elements of the tridiagonal matrix.
            On exit, if INFO = 0, the eigenvalues in ascending order.

    E       (input/output) DOUBLE PRECISION array, dimension (N-1)
            On entry, the subdiagonal elements of the tridiagonal matrix.
            On exit, E has been destroyed.

    Z       (output) COMPLEX_16 array, dimension (LDZ,N)
            On exit, if INFO = 0, Z contains the orthonormal eigenvectors
            of the symmetric tridiagonal matrix.

    LDZ     (input) INTEGER
            The leading dimension of the array Z. LDZ >= max(1,N).

    RWORK   (workspace/output) DOUBLE PRECISION array, dimension (LRWORK)
            On exit, if INFO = 0, RWORK(1) returns the optimal LRWORK.

    LRWORK  (input) INTEGER
            The dimension of the array RWORK.
            LRWORK >= 1 + 4*N + 2*N**2.
            Note that if N is less than or
            equal to the minimum divide size, usually 25, then LRWORK
            need only be max(1,2*(N-1)).

            If LRWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.

    LIWORK  (input) INTEGER
            The dimension of the array IWORK.
            LIWORK >= 3 + 5*N .
            Note that if N is less than or
            equal to the minimum divide size, usually 25, then LIWORK
            need only be 1.

            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    DWORK  (device workspace) DOUBLE PRECISION array, dimension (3*N*N/2+3*N)

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  The algorithm failed to compute an eigenvalue while
                  working on the submatrix lying in rows and columns
                  INFO/(N+1) through mod(INFO,N+1).

    Further Details
    ===============
    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    ===================================================================== */
    char range_[2] = {range, 0};

    magma_int_t alleig, indeig, valeig, lquery;
    magma_int_t i, j, smlsiz;
    magma_int_t liwmin, lrwmin;

    alleig = lapackf77_lsame(range_, "A");
    valeig = lapackf77_lsame(range_, "V");
    indeig = lapackf77_lsame(range_, "I");
    lquery = lrwork == -1 || liwork == -1;

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
        if( n <= 1 ){
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

    if(n==0)
        return *info;
    if(n==1){
        *z = MAGMA_Z_MAKE( 1, 0 );
        return *info;
    }

    // If N is smaller than the minimum divide size (SMLSIZ+1), then
    // solve the problem with another solver.

    if (n < smlsiz){

        char char_I[]= {'I', 0};
        lapackf77_zsteqr(char_I, &n, d, e, z, &ldz, rwork, info);

    } else {
        // We simply call DSTEDX instead.
        magma_dstedx(range, n, vl, vu, il, iu, d, e, rwork, n,
                     rwork+n*n, lrwork-n*n, iwork, liwork, dwork, info);

        for(j=0; j<n; ++j)
            for(i=0; i<n; ++i){
                *(z+i+ldz*j) = MAGMA_Z_MAKE( *(rwork+i+n*j), 0 );
            }
    }

    rwork[0] = lrwmin;
    iwork[0] = liwmin;

    return *info;

} /* zstedx */
