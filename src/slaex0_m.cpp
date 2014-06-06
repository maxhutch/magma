/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
       
       @author Raffaele Solca
       
       @generated s Tue Dec 17 13:18:36 2013
*/
#include "common_magma.h"

#define Q(ix, iy) (q + (ix) + ldq * (iy))

extern "C" {
    magma_int_t magma_slaex1_m(magma_int_t nrgpu, magma_int_t n, float* d, float* q, magma_int_t ldq,
                               magma_int_t* indxq, float rho, magma_int_t cutpnt,
                               float* work, magma_int_t* iwork, float** dwork,
                               magma_queue_t stream[MagmaMaxGPUs][2],
                               char range, float vl, float vu,
                               magma_int_t il, magma_int_t iu, magma_int_t* info);

    magma_int_t magma_get_slaex3_m_nb();       // defined in slaex3_m.cpp

}

extern "C" magma_int_t
magma_slaex0_m(magma_int_t nrgpu, magma_int_t n, float* d, float* e, float* q, magma_int_t ldq,
               float* work, magma_int_t* iwork,
               char range, float vl, float vu,
               magma_int_t il, magma_int_t iu, magma_int_t* info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       .. Scalar Arguments ..
      CHARACTER          RANGE
      INTEGER            IL, IU, INFO, LDQ, N
      REAL   VL, VU
       ..
       .. Array Arguments ..
      INTEGER            IWORK( * )
      REAL   D( * ), E( * ), Q( LDQ, * ),
     $                   WORK( * )
       ..

    Purpose
    =======
    SLAEX0 computes all eigenvalues and the choosen eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.

    Arguments
    =========
    N       (input) INTEGER
            The dimension of the symmetric tridiagonal matrix.  N >= 0.
            
    D       (input/output) REAL array, dimension (N)
            On entry, the main diagonal of the tridiagonal matrix.
            On exit, its eigenvalues.
            
    E       (input) REAL array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix.
            On exit, E has been destroyed.
            
    Q       (input/output) REAL array, dimension (LDQ, N)
            On entry, Q will be the identity matrix.
            On exit, Q contains the eigenvectors of the
            tridiagonal matrix.
            
    LDQ     (input) INTEGER
            The leading dimension of the array Q.  If eigenvectors are
            desired, then  LDQ >= max(1,N).  In any case,  LDQ >= 1.
            
    WORK    (workspace) REAL array,
            the dimension of WORK >= 4*N + N**2.
            
    IWORK   (workspace) INTEGER array,
            the dimension of IWORK >= 3 + 5*N.
            
    RANGE   (input) CHARACTER*1
            = 'A': all eigenvalues will be found.
            = 'V': all eigenvalues in the half-open interval (VL,VU]
                   will be found.
            = 'I': the IL-th through IU-th eigenvalues will be found.
            
    VL      (input) REAL
    VU      (input) REAL
            If RANGE='V', the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = 'A' or 'I'.
            
    IL      (input) INTEGER
    IU      (input) INTEGER
            If RANGE='I', the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = 'A' or 'V'.
            
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

    =====================================================================
*/

    magma_int_t ione = 1;
    char range_;
    magma_int_t curlvl, i, indxq;
    magma_int_t igpu, j, k, matsiz, msd2, smlsiz;
    magma_int_t submat, subpbs, tlvls;
    float* dw[MagmaMaxGPUs];
    magma_queue_t stream [MagmaMaxGPUs][2];
    int gpu_b;
    magma_getdevice(&gpu_b);

    // Test the input parameters.

    *info = 0;

    if( n < 0 )
        *info = -1;
    else if( ldq < max(1, n) )
        *info = -5;
    if( *info != 0 ){
        magma_xerbla( __func__, -*info );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }

    // Quick return if possible
    if(n == 0)
        return MAGMA_SUCCESS;

    //workspace dimension for nrgpu > 1
    size_t tmp = (n-1)/2+1;
    if (nrgpu > 1){
        size_t tmp2 = (tmp-1) / (nrgpu/2) + 1;
        tmp = tmp * tmp2 + 2 * magma_get_slaex3_m_nb()*(tmp + tmp2);
    }

    for (igpu = 0; igpu < nrgpu; ++igpu){
        magma_setdevice(igpu);
        if(nrgpu==1){
            if (MAGMA_SUCCESS != magma_smalloc( &dw[igpu], 3*n*(n/2 + 1) )) {
                *info = -15;
                return MAGMA_ERR_DEVICE_ALLOC;
            }
        }
        else {
            if (MAGMA_SUCCESS != magma_smalloc( &dw[igpu], tmp )) {
                *info = -15;
                return MAGMA_ERR_DEVICE_ALLOC;
            }
        }
        magma_queue_create( &stream[igpu][0] );
        magma_queue_create( &stream[igpu][1] );
    }

    smlsiz = magma_get_smlsize_divideconquer();

    // Determine the size and placement of the submatrices, and save in
    // the leading elements of IWORK.

    iwork[0] = n;
    subpbs= 1;
    tlvls = 0;
    while (iwork[subpbs - 1] > smlsiz) {
        for (j = subpbs; j > 0; --j){
            iwork[2*j - 1] = (iwork[j-1]+1)/2;
            iwork[2*j - 2] = iwork[j-1]/2;
        }
        ++tlvls;
        subpbs *= 2;
    }
    for (j=1; j<subpbs; ++j)
        iwork[j] += iwork[j-1];

    // Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
    // using rank-1 modifications (cuts).

    for(i=0; i < subpbs-1; ++i){
        submat = iwork[i];
        d[submat-1] -= MAGMA_S_ABS(e[submat-1]);
        d[submat] -= MAGMA_S_ABS(e[submat-1]);
    }

    indxq = 4*n + 3;

    // Solve each submatrix eigenproblem at the bottom of the divide and
    // conquer tree.

    char char_I[] = {'I', 0};

#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
        magma_timestr_t start, end;
        start = get_current_time();
#endif

    for (i = 0; i < subpbs; ++i){
        if(i == 0){
            submat = 0;
            matsiz = iwork[0];
        } else {
            submat = iwork[i-1];
            matsiz = iwork[i] - iwork[i-1];
        }
        lapackf77_ssteqr(char_I , &matsiz, &d[submat], &e[submat],
                         Q(submat, submat), &ldq, work, info);  // change to edc?
        if(*info != 0){
            printf("info: %d\n, submat: %d\n", (int) *info, (int) submat);
            *info = (submat+1)*(n+1) + submat + matsiz;
            printf("info: %d\n", (int) *info);
            return MAGMA_SUCCESS;
        }
        k = 1;
        for(j = submat; j < iwork[i]; ++j){
            iwork[indxq+j] = k;
            ++k;
        }
    }

#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
    end = get_current_time();
    printf("  for: ssteqr = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif
    // Successively merge eigensystems of adjacent submatrices
    // into eigensystem for the corresponding larger matrix.

    curlvl = 1;
    while (subpbs > 1){
#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
        magma_timestr_t start, end;
        start = get_current_time();
#endif
        for (i=0; i<subpbs-1; i+=2){
            if(i == 0){
                submat = 0;
                matsiz = iwork[1];
                msd2 = iwork[0];
            } else {
                submat = iwork[i-1];
                matsiz = iwork[i+1] - iwork[i-1];
                msd2 = matsiz / 2;
            }

            // Merge lower order eigensystems (of size MSD2 and MATSIZ - MSD2)
            // into an eigensystem of size MATSIZ.
            // SLAEX1 is used only for the full eigensystem of a tridiagonal
            // matrix.

            if (matsiz == n)
                range_=range;
            else
                // We need all the eigenvectors if it is not last step
                range_='A';

            magma_slaex1_m(nrgpu, matsiz, &d[submat], Q(submat, submat), ldq,
                           &iwork[indxq+submat], e[submat+msd2-1], msd2,
                           work, &iwork[subpbs], dw, stream,
                           range_, vl, vu, il, iu, info);

            if(*info != 0){
                *info = (submat+1)*(n+1) + submat + matsiz;
                return MAGMA_SUCCESS;
            }
            iwork[i/2]= iwork[i+1];
        }
        subpbs /= 2;
        ++curlvl;
#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
        end = get_current_time();
        //printf("%d: time: %6.2f\n", curlvl, GetTimerValue(start,end)/1000.);
#endif

    }

    // Re-merge the eigenvalues/vectors which were deflated at the final
    // merge step.

    for(i = 0; i<n; ++i){
        j = iwork[indxq+i] - 1;
        work[i] = d[j];
        blasf77_scopy(&n, Q(0, j), &ione, &work[ n*(i+1) ], &ione);
    }
    blasf77_scopy(&n, work, &ione, d, &ione);
    char char_A[] = {'A',0};
    lapackf77_slacpy ( char_A, &n, &n, &work[n], &n, q, &ldq );

    for (igpu = 0; igpu < nrgpu; ++igpu){
        magma_setdevice(igpu);
        magma_queue_destroy( stream[igpu][0] );
        magma_queue_destroy( stream[igpu][1] );
        magma_free( dw[igpu] );
    }

    magma_setdevice(gpu_b);

    return MAGMA_SUCCESS;

} /* magma_slaex0 */
