/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
       
       @author Raffaele Solca
       
       @precisions normal d -> s
*/
#include "common_magma.h"

#define Z(ix, iy) (z + (ix) + ldz * (iy))

extern "C"{
    magma_int_t magma_dlaex0_m(magma_int_t nrgpu, magma_int_t n, double* d, double* e, double* q, magma_int_t ldq,
                               double* work, magma_int_t* iwork,
                               char range, double vl, double vu,
                               magma_int_t il, magma_int_t iu, magma_int_t* info);
}

extern "C" magma_int_t
magma_dstedx_m(magma_int_t nrgpu, char range, magma_int_t n, double vl, double vu,
               magma_int_t il, magma_int_t iu, double* d, double* e, double* z, magma_int_t ldz,
               double* work, magma_int_t lwork, magma_int_t* iwork, magma_int_t liwork,
               magma_int_t* info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       .. Scalar Arguments ..
      CHARACTER          RANGE
      INTEGER            IL, IU, INFO, LDZ, LIWORK, LWORK, N
      DOUBLE PRECISION   VL, VU
       ..
       .. Array Arguments ..
      INTEGER            IWORK( * )
      DOUBLE PRECISION   D( * ), E( * ), WORK( * ), Z( LDZ, * ),
     $                   DWORK ( * )
       ..

    Purpose
    =======
    DSTEDX computes some eigenvalues and, optionally, eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.  See DLAEX3 for details.

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

    Z       (input/output) DOUBLE PRECISION array, dimension (LDZ,N)
            On exit, if INFO = 0, Z contains the orthonormal eigenvectors
            of the symmetric tridiagonal matrix.

    LDZ     (input) INTEGER
            The leading dimension of the array Z. LDZ >= max(1,N).

    WORK    (workspace/output) DOUBLE PRECISION array,
                                           dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The dimension of the array WORK.
            If N > 1 then LWORK >= ( 1 + 4*N + N**2 ).
            Note that  if N is less than or
            equal to the minimum divide size, usually 25, then LWORK need
            only be max(1,2*(N-1)).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.

    LIWORK  (input) INTEGER
            The dimension of the array IWORK.
            LIWORK >= ( 3 + 5*N ).
            Note that if N is less than or
            equal to the minimum divide size, usually 25, then LIWORK
            need only be 1.

            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal size of the IWORK array,
            returns this value as the first entry of the IWORK array, and
            no error message related to LIWORK is issued by XERBLA.

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
    Modified by Francoise Tisseur, University of Tennessee.

    =====================================================================
*/
    char range_[2] = {range, 0};

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

    alleig = lapackf77_lsame(range_, "A");
    valeig = lapackf77_lsame(range_, "V");
    indeig = lapackf77_lsame(range_, "I");
    lquery = lwork == -1 || liwork == -1;

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
        return MAGMA_ERR_ILLEGAL_VALUE;
    } else if (lquery) {
        return MAGMA_SUCCESS;
    }

    // Quick return if possible

    if(n==0)
        return MAGMA_SUCCESS;
    if(n==1){
        *z = 1.;
        return MAGMA_SUCCESS;
    }

    /* determine the number of threads */
    magma_int_t threads = magma_get_numthreads();
    magma_setlapack_numthreads(threads);

#ifdef ENABLE_DEBUG
    printf("  D&C_m is using %d threads\n", threads);
#endif

    // If N is smaller than the minimum divide size (SMLSIZ+1), then
    // solve the problem with another solver.

    if (n < smlsiz){
        char char_I[]= {'I', 0};
        lapackf77_dsteqr(char_I, &n, d, e, z, &ldz, work, info);
    } else {
        char char_F[]= {'F', 0};
        lapackf77_dlaset(char_F, &n, &n, &d_zero, &d_one, z, &ldz);

        //Scale.
        char char_M[]= {'M', 0};

        orgnrm = lapackf77_dlanst(char_M, &n, d, e);

        if (orgnrm == 0){
            work[0]  = lwmin;
            iwork[0] = liwmin;
            return MAGMA_SUCCESS;
        }

        eps = lapackf77_dlamch( "Epsilon" );

        if (alleig){
            start = 0;
            while ( start < n ){

                // Let FINISH be the position of the next subdiagonal entry
                // such that E( END ) <= TINY or FINISH = N if no such
                // subdiagonal exists.  The matrix identified by the elements
                // between START and END constitutes an independent
                // sub-problem.

                for(end = start+1; end < n; ++end){
                    tiny = eps * sqrt( MAGMA_D_ABS(d[end-1]*d[end]));
                    if (MAGMA_D_ABS(e[end-1]) <= tiny)
                        break;
                }

                // (Sub) Problem determined.  Compute its size and solve it.

                m = end - start;
                if (m==1){
                    start = end;
                    continue;
                }
                if (m > smlsiz){

                    // Scale
                    char char_G[] = {'G', 0};
                    orgnrm = lapackf77_dlanst(char_M, &m, &d[start], &e[start]);
                    lapackf77_dlascl(char_G, &izero, &izero, &orgnrm, &d_one, &m, &ione, &d[start], &m, info);
                    magma_int_t mm = m-1;
                    lapackf77_dlascl(char_G, &izero, &izero, &orgnrm, &d_one, &mm, &ione, &e[start], &mm, info);

                    magma_dlaex0_m( nrgpu, m, &d[start], &e[start], Z(start, start), ldz, work, iwork, 'A', vl, vu, il, iu, info);

                    if( *info != 0) {
                        return MAGMA_SUCCESS;
                    }

                    // Scale Back
                    lapackf77_dlascl(char_G, &izero, &izero, &d_one, &orgnrm, &m, &ione, &d[start], &m, info);

                } else {

                    char char_I[]= {'I', 0};
                    lapackf77_dsteqr( char_I, &m, &d[start], &e[start], Z(start, start), &ldz, work, info);
                    if (*info != 0){
                        *info = (start+1) *(n+1) + end;
                    }
                }

                start = end;
            }


            // If the problem split any number of times, then the eigenvalues
            // will not be properly ordered.  Here we permute the eigenvalues
            // (and the associated eigenvectors) into ascending order.

            if (m < n){

                // Use Selection Sort to minimize swaps of eigenvectors
                for (i = 1; i < n; ++i){
                    k = i-1;
                    p = d[i-1];
                    for (j = i; j < n; ++j){
                        if (d[j] < p){
                            k = j;
                            p = d[j];
                        }
                    }
                    if(k != i-1) {
                        d[k] = d[i-1];
                        d[i-1] = p;
                        blasf77_dswap(&n, Z(0,i-1), &ione, Z(0,k), &ione);
                    }
                }
            }

        } else {

            // Scale
            char char_G[] = {'G', 0};
            lapackf77_dlascl(char_G, &izero, &izero, &orgnrm, &d_one, &n, &ione, d, &n, info);
            magma_int_t nm = n-1;
            lapackf77_dlascl(char_G, &izero, &izero, &orgnrm, &d_one, &nm, &ione, e, &nm, info);

            magma_dlaex0_m(nrgpu, n, d, e, z, ldz, work, iwork, range, vl, vu, il, iu, info);

            if( *info != 0) {
                return MAGMA_SUCCESS;
            }

            // Scale Back
            lapackf77_dlascl(char_G, &izero, &izero, &d_one, &orgnrm, &n, &ione, d, &n, info);

        }
    }

    work[0]  = lwmin;
    iwork[0] = liwmin;

    return MAGMA_SUCCESS;

} /* magma_dstedx_m */
