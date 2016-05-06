/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Azzam Haidar
       @author Stan Tomov
       @author Raffaele Solca

       @generated from src/zheevdx_2stage_m.cpp normal z -> c, Mon May  2 23:30:19 2016

*/
#include "magma_internal.h"
#include "magma_timer.h"

#define COMPLEX

/**
    Purpose
    -------
    CHEEVD_2STAGE_M computes all eigenvalues and, optionally, eigenvectors of a
    complex Hermitian matrix A. It uses a two-stage algorithm for the tridiagonalization.
    If eigenvectors are desired, it uses a divide and conquer algorithm.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    jobz    magma_vec_t
      -     = MagmaNoVec:  Compute eigenvalues only;
      -     = MagmaVec:    Compute eigenvalues and eigenvectors.

    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                   will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX array, dimension (LDA, N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = MagmaLower,
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, if JOBZ = MagmaVec, then if INFO = 0, the first m columns
            of A contains the required
            orthonormal eigenvectors of the matrix A.
            If JOBZ = MagmaNoVec, then on exit the lower triangle (if UPLO=MagmaLower)
            or the upper triangle (if UPLO=MagmaUpper) of A, including the
            diagonal, is destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

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
    m       INTEGER
            The total number of eigenvalues found.  0 <= M <= N.
            If RANGE = MagmaRangeAll, M = N, and if RANGE = MagmaRangeI, M = IU-IL+1.

    @param[out]
    W       REAL array, dimension (N)
            If INFO = 0, the required m eigenvalues in ascending order.

    @param[out]
    work    (workspace) COMPLEX array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK.
            If N <= 1,                      LWORK >= 1.
            If JOBZ = MagmaNoVec and N > 1, LWORK >= LWSTG2 + N + N*NB.
            If JOBZ = MagmaVec   and N > 1, LWORK >= LWSTG2 + 2*N + N**2.
            where LWSTG2 is the size needed to store the matrices of stage 2
            and is returned by magma_cbulge_getlwstg2.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal sizes of the WORK, RWORK and
            IWORK arrays, returns these values as the first entries of
            the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    @param[out]
    rwork   (workspace) REAL array,
                                           dimension (LRWORK)
            On exit, if INFO = 0, RWORK[0] returns the optimal LRWORK.

    @param[in]
    lrwork  INTEGER
            The dimension of the array RWORK.
            If N <= 1,                      LRWORK >= 1.
            If JOBZ = MagmaNoVec and N > 1, LRWORK >= N.
            If JOBZ = MagmaVec   and N > 1, LRWORK >= 1 + 5*N + 2*N**2.
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
            If N <= 1,                      LIWORK >= 1.
            If JOBZ = MagmaNoVec and N > 1, LIWORK >= 1.
            If JOBZ = MagmaVec   and N > 1, LIWORK >= 3 + 5*N.
    \n
            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i and JOBZ = MagmaNoVec, then the algorithm failed
                  to converge; i off-diagonal elements of an intermediate
                  tridiagonal form did not converge to zero;
                  if INFO = i and JOBZ = MagmaVec, then the algorithm failed
                  to compute an eigenvalue while working on the submatrix
                  lying in rows and columns INFO/(N+1) through
                  mod(INFO,N+1).

    Further Details
    ---------------
    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    Modified description of INFO. Sven, 16 Feb 05.

    @ingroup magma_cheevd_driver
    ********************************************************************/
extern "C" magma_int_t
magma_cheevdx_2stage_m(
    magma_int_t ngpu,
    magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, float *W,
    magmaFloatComplex *work, magma_int_t lwork,
    #ifdef COMPLEX
    float *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info)
{
    #define A( i_,j_) (A  + (i_) + (j_)*lda)
    #define A2(i_,j_) (A2 + (i_) + (j_)*lda2)
    
    const char* uplo_  = lapack_uplo_const( uplo  );
    const char* jobz_  = lapack_vec_const( jobz  );
    magmaFloatComplex c_one  = MAGMA_C_ONE;
    magma_int_t ione = 1;
    magma_int_t izero = 0;
    float d_one = 1.;

    float d__1;

    float eps;
    float anrm;
    magma_int_t imax;
    float rmin, rmax;
    float sigma;
    #ifdef COMPLEX
    magma_int_t lrwmin;
    #endif
    magma_int_t lwmin, liwmin;
    magma_int_t lower;
    magma_int_t wantz;
    magma_int_t iscale;
    float safmin;
    float bignum;
    float smlnum;
    magma_int_t lquery;
    magma_int_t alleig, valeig, indeig;
    magma_int_t len;

    wantz  = (jobz == MagmaVec);
    lower  = (uplo == MagmaLower);
    alleig = (range == MagmaRangeAll);
    valeig = (range == MagmaRangeV);
    indeig = (range == MagmaRangeI);

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );

    /* determine the number of threads and other parameter */
    magma_int_t Vblksiz, ldv, ldt, blkcnt, sizTAU2, sizT2, sizV2, sizTAU1, ldz, lwstg1, lda2;
    magma_int_t parallel_threads = magma_get_parallel_numthreads();
    magma_int_t nb               = magma_get_cbulge_nb(n, parallel_threads);
    magma_int_t lwstg2           = magma_cbulge_getlwstg2( n, parallel_threads, wantz, 
                                                           &Vblksiz, &ldv, &ldt, &blkcnt, 
                                                           &sizTAU2, &sizT2, &sizV2);
    lwstg1                       = magma_bulge_getlwstg1( n, nb, &lda2 ); // lwstg1=nb*n but since used also to store the band A2 so it is 2nb*n;

    sizTAU1                      = n;
    ldz                          = n;

    //magma_int_t sizZ;
    //sizZ                         = wantz == 0 ? 0 : n*ldz;

    #ifdef COMPLEX
    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);
    #else
    lquery = (lwork == -1 || liwork == -1);
    #endif

    *info = 0;
    if (! (wantz || (jobz == MagmaNoVec))) {
        *info = -1;
    } else if (! (alleig || valeig || indeig)) {
        *info = -2;
    } else if (! (lower || (uplo == MagmaUpper))) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < max(1,n)) {
        *info = -6;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -8;
            }
        } else if (indeig) {
            if (il < 1 || il > max(1,n)) {
                *info = -9;
            } else if (iu < min(n,il) || iu > n) {
                *info = -10;
            }
        }
    }


    #ifdef COMPLEX
    if (wantz) {
        lwmin  = lwstg2 + 2*n + max(lwstg1, n*n);
        lrwmin = 1 + 5*n + 2*n*n;
        liwmin = 5*n + 3;
    } else {
        lwmin  = lwstg2 + n + lwstg1;
        lrwmin = n;
        liwmin = 1;
    }

    work[0]  = magma_cmake_lwork( lwmin );
    rwork[0] = magma_smake_lwork( lrwmin );
    iwork[0] = liwmin;

    if ((lwork < lwmin) && !lquery) {
        *info = -14;
    } else if ((lrwork < lrwmin) && ! lquery) {
        *info = -16;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -18;
    }
    #else
    if (wantz) {
        lwmin  = lwstg2 + 1 + 6*n + max(lwstg1, 2*n*n);
        liwmin = 5*n + 3;
    } else {
        lwmin  = lwstg2 + 2*n + lwstg1;
        liwmin = 1;
    }

    work[0]  = magma_smake_lwork( lwmin );
    iwork[0] = liwmin;

    if ((lwork < lwmin) && !lquery) {
        *info = -14;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -16;
    }
    #endif

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

    if (n == 1) {
        W[0] = MAGMA_C_REAL(A[0]);
        if (wantz) {
            A[0] = MAGMA_C_ONE;
        }
        return *info;
    }


    timer_printf("using %d parallel_threads\n", (int) parallel_threads);

    /* Check if matrix is very small then just call LAPACK on CPU, no need for GPU */
    magma_int_t ntiles = n/nb;
    if ( ( ntiles < 2 ) || ( n <= 128 ) ) {
        #ifdef ENABLE_DEBUG
        printf("--------------------------------------------------------------\n");
        printf("  warning matrix too small N=%d NB=%d, calling lapack on CPU  \n", (int) n, (int) nb);
        printf("--------------------------------------------------------------\n");
        #endif
        lapackf77_cheevd(jobz_, uplo_, &n,
                        A, &lda, W,
                        work, &lwork,
                        #ifdef COMPLEX
                        rwork, &lrwork,
                        #endif
                        iwork, &liwork,
                        info);
        *m = n;
        return *info;
    }

    /* Get machine constants. */
    safmin = lapackf77_slamch("Safe minimum");
    eps = lapackf77_slamch("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = magma_ssqrt(smlnum);
    rmax = magma_ssqrt(bignum);

    /* Scale matrix to allowable range, if necessary. */
    #ifdef COMPLEX
    anrm = lapackf77_clanhe("M", uplo_, &n, A, &lda, rwork);
    #else
    anrm = lapackf77_slansy("M", uplo_, &n, A, &lda, work);
    #endif
    iscale = 0;
    if (anrm > 0. && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        lapackf77_clascl(uplo_, &izero, &izero, &d_one, &sigma, &n, &n, A,
                         &lda, info);
    }


/*
    #ifdef COMPLEX
    magma_int_t indTAU1 = 0;
    #else
    magma_int_t indTAU1 = n;
    #endif
    magma_int_t indTAU2 = indTAU1 + sizTAU1;
    magma_int_t indV2   = indTAU2 + sizTAU2;
    magma_int_t indT2   = indV2   + sizV2;
    magma_int_t indWORK = indT2   + sizT2;
    magma_int_t indA2   = indWORK;
    magma_int_t indZ    = indWORK;
    magma_int_t indWEDC = indZ   + sizZ;
*/

    #ifdef COMPLEX
    float *E                 = rwork;
    magma_int_t sizE_onwork   = 0;
    #else
    float *E                 = work;
    magma_int_t sizE_onwork   = n;
    #endif
    
    magmaFloatComplex *TAU1  = work + sizE_onwork;
    magmaFloatComplex *TAU2  = TAU1 + sizTAU1;
    magmaFloatComplex *V2    = TAU2 + sizTAU2;
    magmaFloatComplex *T2    = V2   + sizV2;
    magmaFloatComplex *Wstg1 = T2   + sizT2;
    magmaFloatComplex *A2    = Wstg1; /* PAY ATTENTION THAT work[indA2] should be able to be of size lda2*n which it should be checked in any future modification of lwork.*/
    magmaFloatComplex *Z     = Wstg1;
    magmaFloatComplex *Wmqr1 = TAU2; // reuse TAU2 as work of unmqr for stage 1 since no need anymor efor TAU2 and V2 and T2 but pay attentionthat does not overlap with Z=Wstg1 so lwmqr1
    magma_int_t lwmqr1        = sizTAU2+sizV2+sizT2;
    #ifdef COMPLEX
    float *Wedc              = E + n; 
    magma_int_t lwedc         = 1 + 4*n + 2*n*n; // lrwork - n; //used only for wantz>0
    #else
    float *Wedc              = Wstg1 + n*n;
    magma_int_t lwedc         = 1 + 4*n + n*n; // lwork - indWEDC; //used only for wantz>0
    #endif
    magma_int_t i;

    magma_timer_t time=0, time_total=0, time_alloc=0, time_dist=0, time_band=0;
    timer_start( time_total );
    timer_start( time );

    //#define SINGLEGPU

#ifdef SINGLEGPU
    magmaFloatComplex *dT1sgpu;
    if (MAGMA_SUCCESS != magma_cmalloc( &dT1sgpu, n*nb)) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    magma_chetrd_he2hb(uplo, n, nb, A, lda, TAU1, Wstg1, lwstg1, dT1sgpu, info);

    timer_stop( time );
    timer_printf( "  N= %10d  nb= %5d time chetrd_he2hb= %6.2f\n", (int)n, (int)nb, time );
    timer_start( time );
#else
    magma_int_t nstream = max(3,ngpu+2);
    magma_queue_t queues[MagmaMaxGPUs][20];
    magmaFloatComplex *dA[MagmaMaxGPUs], *dT1[MagmaMaxGPUs];
    magma_int_t ldda = magma_roundup( n, 32 );

    magma_int_t distblk = max(256, 4*nb);

    #ifdef ENABLE_DEBUG
    printf("voici ngpu %d distblk %d NB %d nstream %d\n ", ngpu, distblk, nb, nstream);
    #endif

    timer_start( time_alloc );
    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_int_t mlocal = ((n / distblk) / ngpu + 1) * distblk;
        magma_setdevice( dev );
        // TODO check malloc
        magma_cmalloc(&dA[dev], ldda*mlocal );
        magma_cmalloc(&dT1[dev], (n*nb) );
        for( i = 0; i < nstream; ++i ) {
            magma_queue_create( dev, &queues[dev][i] );
        }
    }
    timer_stop( time_alloc );
    
    timer_start( time_dist );
    magma_queue_t distqueues[MagmaMaxGPUs];
    for( magma_int_t dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( dev, &distqueues[dev] );
    }
    magma_csetmatrix_1D_col_bcyclic( n, n, A, lda, dA, ldda, ngpu, distblk, distqueues );
    for( magma_int_t dev=0; dev < ngpu; dev++ ) {
        magma_setdevice( dev );
        magma_queue_sync( distqueues[dev] );
        magma_queue_destroy( distqueues[dev] );
    }

    magma_setdevice(0);
    timer_stop( time_dist );

    timer_start( time_band );
    magma_chetrd_he2hb_mgpu( uplo, n, nb, A, lda, TAU1, Wstg1, lwstg1, dA, ldda,
                             dT1, nb, ngpu, distblk, queues, nstream, info );
    timer_stop( time_band );
    timer_printf("    time alloc %7.4f, ditribution %7.4f, chetrd_he2hb_m only = %7.4f\n", time_alloc, time_dist, time_band );

    for( magma_int_t dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_free( dA[dev] );
        magma_free( dT1[dev] );
        for( i = 0; i < nstream; ++i ) {
            magma_queue_sync( queues[dev][i] );
            magma_queue_destroy( queues[dev][i] );
        }
    }
#endif // not SINGLEGPU



    /* copy the input matrix into WORK(INDWRK) with band storage */
    memset(A2, 0, n*lda2*sizeof(magmaFloatComplex));

    for (magma_int_t j = 0; j < n-nb; j++) {
        len = nb+1;
        blasf77_ccopy( &len, A(j,j), &ione, A2(0,j), &ione );
        memset(A(j,j), 0, (nb+1)*sizeof(magmaFloatComplex));
        *A(nb+j,j) = c_one;
    }
    for (magma_int_t j = 0; j < nb; j++) {
        len = nb-j;
        blasf77_ccopy( &len, A(j+n-nb,j+n-nb), &ione, A2(0,j+n-nb), &ione );
        memset(A(j+n-nb,j+n-nb), 0, (nb-j)*sizeof(magmaFloatComplex));
    }

    timer_stop( time );
    timer_printf( "  N= %10d  nb= %5d time chetrd_convert = %6.2f\n", (int)n, (int)nb, time );
    timer_start( time );

    magma_chetrd_hb2st(uplo, n, nb, Vblksiz, A2, lda2, W, E, V2, ldv, TAU2, wantz, T2, ldt);

    timer_stop( time );
    timer_stop( time_total );
    timer_printf( "  N= %10d  nb= %5d time chetrd_hb2st= %6.2f\n", (int)n, (int)nb, time );
    timer_printf( "  N= %10d  nb= %5d time chetrd= %6.2f\n", (int)n, (int)nb, time_total );

    /* For eigenvalues only, call SSTERF.  For eigenvectors, first call
     CSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
     tridiagonal matrix, then call CUNMTR to multiply it to the Householder
     transformations represented as Householder vectors in A. */
    if (! wantz) {
        timer_start( time );

        lapackf77_ssterf(&n, W, E, info);
        magma_smove_eig(range, n, W, &il, &iu, vl, vu, m);

        timer_stop( time );
        timer_printf( "  N= %10d  nb= %5d time dstedc = %6.2f\n", (int)n, (int)nb, time );
    }
    else {
        timer_start( time_total );
        timer_start( time );

#ifdef SINGLEGPU
        float* dwedc;
        if (MAGMA_SUCCESS != magma_smalloc( &dwedc, 3*n*(n/2 + 1) )) {
            // TODO free dT1, etc. --- see goto cleanup in slaex0_m.cpp, etc.
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
        magma_cstedx(range, n, vl, vu, il, iu, W, E,
                     Z, ldz, Wedc, lwedc,
                     iwork, liwork, dwedc, info);
        magma_free( dwedc );
#else
        magma_cstedx_m(ngpu, range, n, vl, vu, il, iu, W, E,
                     Z, ldz, Wedc, lwedc,
                     iwork, liwork, info);
#endif

        timer_stop( time );
        timer_printf( "  N= %10d  nb= %5d time cstedx_m = %6.2f\n", (int)n, (int)nb, time );

        magma_smove_eig(range, n, W, &il, &iu, vl, vu, m);

#ifdef SINGLEGPU
        magmaFloatComplex *dZ;
        magma_int_t lddz = n;

        if (MAGMA_SUCCESS != magma_cmalloc( &dZ, (*m)*lddz)) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }

        timer_start( time );

        magma_cbulge_back(uplo, n, nb, *m, Vblksiz, Z +ldz*(il-1), ldz, dZ, lddz,
                          V2, ldv, TAU2, T2, ldt, info);
#else
        magma_cbulge_back_m(ngpu, uplo, n, nb, *m, Vblksiz, Z +ldz*(il-1), ldz,
                          V2, ldv, TAU2, T2, ldt, info);
#endif // not SINGLEGPU


        timer_stop( time );
        timer_printf( "  N= %10d  nb= %5d time cbulge_back_m = %6.2f\n", (int)n, (int)nb, time );
        timer_start( time );

#ifdef SINGLEGPU
        magma_cunmqr( MagmaLeft, MagmaNoTrans, n-nb, *m, n-nb, A+nb, lda, TAU1,
                       Z +ldz*(il-1)+nb, ldz, Wmqr1, lwmqr1, info);
#else
        magma_cunmqr_m(ngpu, MagmaLeft, MagmaNoTrans, n-nb, *m, n-nb, A+nb, lda, TAU1,
                       Z +ldz*(il-1)+nb, ldz, Wmqr1, lwmqr1, info);

        lapackf77_clacpy("A", &n, m, Z +ldz*(il-1), &ldz, A, &lda);
#endif // not SINGLEGPU

        timer_stop( time );
        timer_printf( "  N= %10d  nb= %5d time cunmqr_m + copy = %6.2f\n", (int)n, (int)nb, time );
#ifdef SINGLEGPU
        magma_free(dT1sgpu);
        magma_free(dZ);
#endif // not SINGLEGPU

        timer_stop( time_total );
        timer_printf( "  N= %10d  nb= %5d time eigenvectors backtransf. = %6.2f\n", (int)n, (int)nb, time_total );
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        d__1 = 1. / sigma;
        blasf77_sscal(&imax, &d__1, W, &ione);
    }

    work[0]  = magma_cmake_lwork( lwmin );
    #ifdef COMPLEX
    rwork[0] = magma_smake_lwork( lrwmin );
    #endif
    iwork[0] = liwmin;

    magma_setdevice( orig_dev );

    return *info;
} /* magma_cheevdx_2stage_m */
