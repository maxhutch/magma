/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
       
       @author Raffaele Solca
       
       @generated from dlaex0_m.cpp normal d -> s, Fri Jan 30 19:00:18 2015
*/
#include "common_magma.h"
#include "magma_timer.h"

/**
    Purpose
    -------
    SLAEX0 computes all eigenvalues and the choosen eigenvectors of a
    symmetric tridiagonal matrix using the divide and conquer method.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    n       INTEGER
            The dimension of the symmetric tridiagonal matrix.  N >= 0.
            
    @param[in,out]
    d       REAL array, dimension (N)
            On entry, the main diagonal of the tridiagonal matrix.
            On exit, its eigenvalues.
            
    @param[in]
    e       REAL array, dimension (N-1)
            The off-diagonal elements of the tridiagonal matrix.
            On exit, E has been destroyed.
            
    @param[in,out]
    Q       REAL array, dimension (LDQ, N)
            On entry, Q will be the identity matrix.
            On exit, Q contains the eigenvectors of the
            tridiagonal matrix.
            
    @param[in]
    ldq     INTEGER
            The leading dimension of the array Q.  If eigenvectors are
            desired, then  LDQ >= max(1,N).  In any case,  LDQ >= 1.
            
    @param
    work    (workspace) REAL array,
            the dimension of WORK >= 4*N + N**2.
            
    @param
    iwork   (workspace) INTEGER array,
            the dimension of IWORK >= 3 + 5*N.
            
    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                             will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.
            
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

    @ingroup magma_ssyev_aux
    ********************************************************************/
extern "C" magma_int_t
magma_slaex0_m(
    magma_int_t ngpu,
    magma_int_t n, float *d, float *e,
    float *Q, magma_int_t ldq,
    float *work, magma_int_t *iwork,
    magma_range_t range, float vl, float vu,
    magma_int_t il, magma_int_t iu,
    magma_int_t *info)
{
#define Q(i_,j_) (Q + (i_) + (j_)*ldq)

    magma_int_t ione = 1;
    magma_range_t range2;
    magma_int_t curlvl, i, indxq;
    magma_int_t igpu, j, k, matsiz, msd2, smlsiz;
    magma_int_t submat, subpbs, tlvls;
    float *dw[MagmaMaxGPUs];
    magma_queue_t stream [MagmaMaxGPUs][2];

    // Test the input parameters.
    *info = 0;
    if ( n < 0 )
        *info = -1;
    else if ( ldq < max(1, n) )
        *info = -5;
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return if possible
    if (n == 0)
        return *info;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    // workspace dimension for ngpu > 1
    size_t tmp = (n-1)/2+1;
    if (ngpu > 1) {
        size_t tmp2 = (tmp-1) / (ngpu/2) + 1;
        tmp = tmp * tmp2 + 2 * magma_get_slaex3_m_nb()*(tmp + tmp2);
    }

    for (igpu = 0; igpu < ngpu; ++igpu) {
        magma_setdevice(igpu);
        if (ngpu == 1) {
            if (MAGMA_SUCCESS != magma_smalloc( &dw[igpu], 3*n*(n/2 + 1) )) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
        }
        else {
            if (MAGMA_SUCCESS != magma_smalloc( &dw[igpu], tmp )) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
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
        for (j = subpbs; j > 0; --j) {
            iwork[2*j - 1] = (iwork[j-1]+1)/2;
            iwork[2*j - 2] = iwork[j-1]/2;
        }
        ++tlvls;
        subpbs *= 2;
    }
    for (j=1; j < subpbs; ++j)
        iwork[j] += iwork[j-1];

    // Divide the matrix into SUBPBS submatrices of size at most SMLSIZ+1
    // using rank-1 modifications (cuts).

    for (i=0; i < subpbs-1; ++i) {
        submat = iwork[i];
        d[submat-1] -= MAGMA_S_ABS(e[submat-1]);
        d[submat] -= MAGMA_S_ABS(e[submat-1]);
    }

    indxq = 4*n + 3;

    // Solve each submatrix eigenproblem at the bottom of the divide and
    // conquer tree.
    magma_timer_t time=0;
    timer_start( time );

    for (i = 0; i < subpbs; ++i) {
        if (i == 0) {
            submat = 0;
            matsiz = iwork[0];
        } else {
            submat = iwork[i-1];
            matsiz = iwork[i] - iwork[i-1];
        }
        lapackf77_ssteqr("I", &matsiz, &d[submat], &e[submat],
                         Q(submat, submat), &ldq, work, info);  // change to edc?
        if (*info != 0) {
            printf("info: %d\n, submat: %d\n", (int) *info, (int) submat);
            *info = (submat+1)*(n+1) + submat + matsiz;
            printf("info: %d\n", (int) *info);
            return *info;
        }
        k = 1;
        for (j = submat; j < iwork[i]; ++j) {
            iwork[indxq+j] = k;
            ++k;
        }
    }

    timer_stop( time );
    timer_printf( "  for: ssteqr = %6.2f\n", time );
    
    // Successively merge eigensystems of adjacent submatrices
    // into eigensystem for the corresponding larger matrix.

    curlvl = 1;
    while (subpbs > 1) {
        timer_start( time );
        
        for (i=0; i < subpbs-1; i += 2) {
            if (i == 0) {
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
                range2 = range;
            else
                // We need all the eigenvectors if it is not last step
                range2 = MagmaRangeAll;

            magma_slaex1_m(ngpu, matsiz, &d[submat], Q(submat, submat), ldq,
                           &iwork[indxq+submat], e[submat+msd2-1], msd2,
                           work, &iwork[subpbs], dw, stream,
                           range2, vl, vu, il, iu, info);

            if (*info != 0) {
                *info = (submat+1)*(n+1) + submat + matsiz;
                return *info;
            }
            iwork[i/2]= iwork[i+1];
        }
        subpbs /= 2;
        ++curlvl;
        
        timer_stop( time );
        timer_printf("%d: time: %6.2f\n", (int) curlvl, time );
    }

    // Re-merge the eigenvalues/vectors which were deflated at the final
    // merge step.

    for (i = 0; i < n; ++i) {
        j = iwork[indxq+i] - 1;
        work[i] = d[j];
        blasf77_scopy(&n, Q(0, j), &ione, &work[ n*(i+1) ], &ione);
    }
    blasf77_scopy(&n, work, &ione, d, &ione);
    lapackf77_slacpy( "A", &n, &n, &work[n], &n, Q, &ldq );

    for (igpu = 0; igpu < ngpu; ++igpu) {
        magma_setdevice(igpu);
        magma_queue_destroy( stream[igpu][0] );
        magma_queue_destroy( stream[igpu][1] );
        magma_free( dw[igpu] );
    }

    magma_setdevice( orig_dev );

    return *info;
} /* magma_slaex0 */
