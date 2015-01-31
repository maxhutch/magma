/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015
       
       @author Raffaele Solca
       @generated from dlaex3_m.cpp normal d -> s, Fri Jan 30 19:00:18 2015
*/

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common_magma.h"
#include "magma_timer.h"

extern "C" {

magma_int_t magma_get_slaex3_m_k()  { return  512; }
magma_int_t magma_get_slaex3_m_nb() { return 1024; }

}  // end extern "C"

/**
    Purpose
    -------
    SLAEX3 finds the roots of the secular equation, as defined by the
    values in D, W, and RHO, between 1 and K.  It makes the
    appropriate calls to SLAED4 and then updates the eigenvectors by
    multiplying the matrix of eigenvectors of the pair of eigensystems
    being combined by the matrix of eigenvectors of the K-by-K system
    which is solved here.

    It is used in the last step when only a part of the eigenvectors
    is required.
    It compute only the required part of the eigenvectors and the rest
    is not used.

    This code makes very mild assumptions about floating point
    arithmetic. It will work on machines with a guard digit in
    add/subtract, or on those binary machines without guard digits
    which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
    It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    k       INTEGER
            The number of terms in the rational function to be solved by
            SLAED4.  K >= 0.

    @param[in]
    n       INTEGER
            The number of rows and columns in the Q matrix.
            N >= K (deflation may result in N > K).

    @param[in]
    n1      INTEGER
            The location of the last eigenvalue in the leading submatrix.
            min(1,N) <= N1 <= N/2.

    @param[out]
    d       REAL array, dimension (N)
            D(I) contains the updated eigenvalues for
            1 <= I <= K.

    @param[out]
    Q       REAL array, dimension (LDQ,N)
            Initially the first K columns are used as workspace.
            On output the columns ??? to ??? contain
            the updated eigenvectors.

    @param[in]
    ldq     INTEGER
            The leading dimension of the array Q.  LDQ >= max(1,N).

    @param[in]
    rho     REAL
            The value of the parameter in the rank one update equation.
            RHO >= 0 required.

    @param[in,out]
    dlamda  REAL array, dimension (K)
            The first K elements of this array contain the old roots
            of the deflated updating problem.  These are the poles
            of the secular equation. May be changed on output by
            having lowest order bit set to zero on Cray X-MP, Cray Y-MP,
            Cray-2, or Cray C-90, as described above.

    @param[in]
    Q2      REAL array, dimension (LDQ2, N)
            The first K columns of this matrix contain the non-deflated
            eigenvectors for the split problem.

    @param[in]
    indx    INTEGER array, dimension (N)
            The permutation used to arrange the columns of the deflated
            Q matrix into three groups (see SLAED2).
            The rows of the eigenvectors found by SLAED4 must be likewise
            permuted before the matrix multiply can take place.

    @param[in]
    ctot    INTEGER array, dimension (4)
            A count of the total number of the various types of columns
            in Q, as described in INDX.  The fourth column type is any
            column which has been deflated.

    @param[in,out]
    w       REAL array, dimension (K)
            The first K elements of this array contain the components
            of the deflation-adjusted updating vector. Destroyed on
            output.

    @param
    s       (workspace) REAL array, dimension (N1 + 1)*K
            Will contain the eigenvectors of the repaired matrix which
            will be multiplied by the previously accumulated eigenvectors
            to update the system.

    @param[out]
    indxq   INTEGER array, dimension (N)
            On exit, the permutation which will reintegrate the
            subproblems back into sorted order,
            i.e. D( INDXQ( I = 1, N ) ) will be in ascending order.
    
    @param
    dwork   (devices workspaces) REAL array of arrays,
            dimension NRGPU.
            if NRGPU = 1 the dimension of the first workspace
            should be (3*N*N/2+3*N)
            otherwise the NRGPU workspaces should have the size
            ceil((N-N1) * (N-N1) / floor(ngpu/2)) +
            NB * ((N-N1) + (N-N1) / floor(ngpu/2))
    
    @param
    queues  (device queues) magma_queue_t array,
            dimension (MagmaMaxGPUs,2)

    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                             will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.
            TODO verify range, vl, vu, il, iu -- copied from slaex1.

    @param[in]
    vl      REAL
    @param[in]
    vu      REAL
            if RANGE=MagmaRangeV, the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeI.

    @param[in]
    il      INTEGER
    @param[in]
    iu      INTEGER
            if RANGE=MagmaRangeI, the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeV.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  if INFO = 1, an eigenvalue did not converge

    Further Details
    ---------------
    Based on contributions by
    Jeff Rutter, Computer Science Division, University of California
    at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    @ingroup magma_ssyev_aux
    ********************************************************************/
extern "C" magma_int_t
magma_slaex3_m(
    magma_int_t ngpu,
    magma_int_t k, magma_int_t n, magma_int_t n1, float *d,
    float *Q, magma_int_t ldq, float rho,
    float *dlamda, float *Q2, magma_int_t *indx,
    magma_int_t *ctot, float *w, float *s, magma_int_t *indxq,
    magmaFloat_ptr dwork[],
    magma_queue_t queues[MagmaMaxGPUs][2],
    magma_range_t range, float vl, float vu, magma_int_t il, magma_int_t iu,
    magma_int_t *info )
{
#define Q(i_,j_) (Q + (i_) + (j_)*ldq)

#define dQ2(id)    (dwork[id])
#define dS(id, ii) (dwork[id] + n2*n2_loc + (ii)*(n2*nb))
#define dQ(id, ii) (dwork[id] + n2*n2_loc +    2*(n2*nb) + (ii)*(n2_loc*nb))

    if (ngpu == 1) {
        magma_setdevice(0);
        magma_slaex3(k, n, n1, d, Q, ldq, rho,
                     dlamda, Q2, indx, ctot, w, s, indxq,
                     *dwork, range, vl, vu, il, iu, info );
        return *info;
    }
    float d_one  = 1.;
    float d_zero = 0.;
    magma_int_t ione = 1;
    magma_int_t ineg_one = -1;

    magma_int_t iil, iiu, rk;
    magma_int_t n1_loc, n2_loc, ib, nb, ib2, igpu;
    magma_int_t ni_loc[MagmaMaxGPUs];

    magma_int_t i, ind, iq2, j, n12, n2, n23, tmp;
    float temp;
    magma_int_t alleig, valeig, indeig;

    alleig = (range == MagmaRangeAll);
    valeig = (range == MagmaRangeV);
    indeig = (range == MagmaRangeI);

    *info = 0;

    if (k < 0)
        *info=-1;
    else if (n < k)
        *info=-2;
    else if (ldq < max(1,n))
        *info=-6;
    else if (! (alleig || valeig || indeig))
        *info = -15;
    else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -17;
        }
        else if (indeig) {
            if (il < 1 || il > max(1,n))
                *info = -18;
            else if (iu < min(n,il) || iu > n)
                *info = -19;
        }
    }

    if (*info != 0) {
        magma_xerbla(__func__, -(*info));
        return *info;
    }

    // Quick return if possible
    if (k == 0)
        return *info;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
    /*
     Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can
     be computed with high relative accuracy (barring over/underflow).
     This is a problem on machines without a guard digit in
     add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
     The following code replaces DLAMDA(I) by 2*DLAMDA(I)-DLAMDA(I),
     which on any of these machines zeros out the bottommost
     bit of DLAMDA(I) if it is 1; this makes the subsequent
     subtractions DLAMDA(I)-DLAMDA(J) unproblematic when cancellation
     occurs. On binary machines with a guard digit (almost all
     machines) it does not change DLAMDA(I) at all. On hexadecimal
     and decimal machines with a guard digit, it slightly
     changes the bottommost bits of DLAMDA(I). It does not account
     for hexadecimal or decimal machines without guard digits
     (we know of none). We use a subroutine call to compute
     2*DLAMBDA(I) to prevent optimizing compilers from eliminating
     this code.*/

//#define CHECK_CPU
#ifdef CHECK_CPU
    float *hwS[2][MagmaMaxGPUs], *hwQ[2][MagmaMaxGPUs], *hwQ2[MagmaMaxGPUs];
    #define hQ2(id) (hwQ2[id])
    #define hS(id, ii) (hwS[ii][id])
    #define hQ(id, ii) (hwQ[ii][id])
#endif
    n2 = n - n1;

    n12 = ctot[0] + ctot[1];
    n23 = ctot[1] + ctot[2];

    iq2 = n1 * n12;
    //lq2 = iq2 + n2 * n23;

    n1_loc = (n1-1) / (ngpu/2) + 1;
    n2_loc = (n2-1) / (ngpu/2) + 1;

    nb = magma_get_slaex3_m_nb();

    if (n1 >= magma_get_slaex3_m_k()) {
#ifdef CHECK_CPU
        for (igpu = 0; igpu < ngpu; ++igpu) {
            magma_smalloc_pinned( &(hwS[0][igpu]), n2*nb );
            magma_smalloc_pinned( &(hwS[1][igpu]), n2*nb );
            magma_smalloc_pinned( &(hwQ2[igpu]), n2*n2_loc );
            magma_smalloc_pinned( &(hwQ[0][igpu]), n2_loc*nb );
            magma_smalloc_pinned( &(hwQ[1][igpu]), n2_loc*nb );
        }
#endif
        for (igpu = 0; igpu < ngpu-1; igpu += 2) {
            ni_loc[igpu] = min(n1_loc, n1 - igpu/2 * n1_loc);
#ifdef CHECK_CPU
            lapackf77_slacpy("A", &ni_loc[igpu], &n12, Q2+n1_loc*(igpu/2), &n1, hQ2(igpu), &n1_loc);
#endif
            magma_setdevice(igpu);
            magma_ssetmatrix_async( ni_loc[igpu], n12,
                                    Q2+n1_loc*(igpu/2), n1,
                                    dQ2(igpu),          n1_loc, queues[igpu][0] );
            ni_loc[igpu+1] = min(n2_loc, n2 - igpu/2 * n2_loc);
#ifdef CHECK_CPU
            lapackf77_slacpy("A", &ni_loc[igpu+1], &n23, Q2+iq2+n2_loc*(igpu/2), &n2, hQ2(igpu+1), &n2_loc);
#endif
            magma_setdevice(igpu+1);
            magma_ssetmatrix_async( ni_loc[igpu+1], n23,
                                    Q2+iq2+n2_loc*(igpu/2), n2,
                                    dQ2(igpu+1),            n2_loc, queues[igpu+1][0] );
        }
    }

    //

#ifdef _OPENMP
    /////////////////////////////////////////////////////////////////////////////////
    //openmp implementation
    /////////////////////////////////////////////////////////////////////////////////
    magma_timer_t time=0;
    timer_start( time );

#pragma omp parallel private(i, j, tmp, temp)
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t tot = omp_get_num_threads();

        magma_int_t ib = (  id   * k) / tot; //start index of local loop
        magma_int_t ie = ((id+1) * k) / tot; //end index of local loop
        magma_int_t ik = ie - ib;           //number of local indices

        for (i = ib; i < ie; ++i)
            dlamda[i]=lapackf77_slamc3(&dlamda[i], &dlamda[i]) - dlamda[i];

        for (j = ib; j < ie; ++j) {
            magma_int_t tmpp=j+1;
            magma_int_t iinfo = 0;
            lapackf77_slaed4(&k, &tmpp, dlamda, w, Q(0,j), &rho, &d[j], &iinfo);
            // If the zero finder fails, the computation is terminated.
            if (iinfo != 0) {
#pragma omp critical (info)
                *info = iinfo;
                break;
            }
        }

#pragma omp barrier

        if (*info == 0) {
#pragma omp single
            {
                //Prepare the INDXQ sorting permutation.
                magma_int_t nk = n - k;
                lapackf77_slamrg( &k, &nk, d, &ione, &ineg_one, indxq);

                //compute the lower and upper bound of the non-deflated eigenvectors
                if (valeig)
                    magma_svrange(k, d, &iil, &iiu, vl, vu);
                else if (indeig)
                    magma_sirange(k, indxq, &iil, &iiu, il, iu);
                else {
                    iil = 1;
                    iiu = k;
                }
                rk = iiu - iil + 1;
            }

            if (k == 2) {
#pragma omp single
                {
                    for (j = 0; j < k; ++j) {
                        w[0] = *Q(0,j);
                        w[1] = *Q(1,j);

                        i = indx[0] - 1;
                        *Q(0,j) = w[i];
                        i = indx[1] - 1;
                        *Q(1,j) = w[i];
                    }
                }
            }
            else if (k != 1) {
                // Compute updated W.
                blasf77_scopy( &ik, &w[ib], &ione, &s[ib], &ione);

                // Initialize W(I) = Q(I,I)
                tmp = ldq + 1;
                blasf77_scopy( &ik, Q(ib,ib), &tmp, &w[ib], &ione);

                for (j = 0; j < k; ++j) {
                    magma_int_t i_tmp = min(j, ie);
                    for (i = ib; i < i_tmp; ++i)
                        w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
                    i_tmp = max(j+1, ib);
                    for (i = i_tmp; i < ie; ++i)
                        w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
                }

                for (i = ib; i < ie; ++i)
                    w[i] = copysign( sqrt( -w[i] ), s[i]);

#pragma omp barrier

                //reduce the number of used threads to have enough S workspace
                tot = min(n1, omp_get_num_threads());

                if (id < tot) {
                    ib = (  id   * rk) / tot + iil - 1;
                    ie = ((id+1) * rk) / tot + iil - 1;
                    ik = ie - ib;
                }
                else {
                    ib = -1;
                    ie = -1;
                    ik = -1;
                }

                // Compute eigenvectors of the modified rank-1 modification.
                for (j = ib; j < ie; ++j) {
                    for (i = 0; i < k; ++i)
                        s[id*k + i] = w[i] / *Q(i,j);
                    temp = magma_cblas_snrm2( k, s+id*k, 1 );
                    for (i = 0; i < k; ++i) {
                        magma_int_t iii = indx[i] - 1;
                        *Q(i,j) = s[id*k + iii] / temp;
                    }
                }
            }
        }
    }
    if (*info != 0)
        return *info;

    timer_stop( time );
    timer_printf( "eigenvalues/vector D+zzT = %6.2f\n", time );

#else
    /////////////////////////////////////////////////////////////////////////////////
    // Non openmp implementation
    /////////////////////////////////////////////////////////////////////////////////
    magma_timer_t time=0;
    timer_start( time );

    for (i = 0; i < k; ++i)
        dlamda[i]=lapackf77_slamc3(&dlamda[i], &dlamda[i]) - dlamda[i];

    for (j = 0; j < k; ++j) {
        magma_int_t tmpp=j+1;
        magma_int_t iinfo = 0;
        lapackf77_slaed4(&k, &tmpp, dlamda, w, Q(0,j), &rho, &d[j], &iinfo);
        // If the zero finder fails, the computation is terminated.
        if (iinfo != 0)
            *info=iinfo;
    }
    if (*info != 0)
        return *info;

    //Prepare the INDXQ sorting permutation.
    magma_int_t nk = n - k;
    lapackf77_slamrg( &k, &nk, d, &ione, &ineg_one, indxq);

    //compute the lower and upper bound of the non-deflated eigenvectors
    if (valeig)
        magma_svrange(k, d, &iil, &iiu, vl, vu);
    else if (indeig)
        magma_sirange(k, indxq, &iil, &iiu, il, iu);
    else {
        iil = 1;
        iiu = k;
    }
    rk = iiu - iil + 1;

    if (k == 2) {
        for (j = 0; j < k; ++j) {
            w[0] = *Q(0,j);
            w[1] = *Q(1,j);

            i = indx[0] - 1;
            *Q(0,j) = w[i];
            i = indx[1] - 1;
            *Q(1,j) = w[i];
        }
    }
    else if (k != 1) {
        // Compute updated W.
        blasf77_scopy( &k, w, &ione, s, &ione);

        // Initialize W(I) = Q(I,I)
        tmp = ldq + 1;
        blasf77_scopy( &k, Q, &tmp, w, &ione);

        for (j = 0; j < k; ++j) {
            for (i = 0; i < j; ++i)
                w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
            for (i = j+1; i < k; ++i)
                w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
        }

        for (i = 0; i < k; ++i)
            w[i] = copysign( sqrt( -w[i] ), s[i]);

        // Compute eigenvectors of the modified rank-1 modification.
        for (j = iil-1; j < iiu; ++j) {
            for (i = 0; i < k; ++i)
                s[i] = w[i] / *Q(i,j);
            temp = magma_cblas_snrm2( k, s, 1 );
            for (i = 0; i < k; ++i) {
                magma_int_t iii = indx[i] - 1;
                *Q(i,j) = s[iii] / temp;
            }
        }
    }

    timer_stop( time );
    timer_printf( "eigenvalues/vector D+zzT = %6.2f\n", time );

#endif //_OPENMP

    // Compute the updated eigenvectors.

    timer_start( time );

    if (rk > 0) {
        if (n1 < magma_get_slaex3_m_k()) {
            // stay on the CPU
            if ( n23 != 0 ) {
                lapackf77_slacpy("A", &n23, &rk, Q(ctot[0],iil-1), &ldq, s, &n23);
                blasf77_sgemm("N", "N", &n2, &rk, &n23, &d_one, &Q2[iq2], &n2,
                              s, &n23, &d_zero, Q(n1,iil-1), &ldq );
            }
            else
                lapackf77_slaset("A", &n2, &rk, &d_zero, &d_zero, Q(n1,iil-1), &ldq);

            if ( n12 != 0 ) {
                lapackf77_slacpy("A", &n12, &rk, Q(0,iil-1), &ldq, s, &n12);
                blasf77_sgemm("N", "N", &n1, &rk, &n12, &d_one, Q2, &n1,
                              s, &n12, &d_zero, Q(0,iil-1), &ldq);
            }
            else
                lapackf77_slaset("A", &n1, &rk, &d_zero, &d_zero, Q(0,iil-1), &ldq);
        }
        else {
            //use the gpus
            ib = min(nb, rk);
            for (igpu = 0; igpu < ngpu-1; igpu += 2) {
                if (n23 != 0) {
                    magma_setdevice(igpu+1);
                    magma_ssetmatrix_async( n23, ib,
                                            Q(ctot[0],iil-1), ldq,
                                            dS(igpu+1,0),     n23, queues[igpu+1][0] );
                }
                if (n12 != 0) {
                    magma_setdevice(igpu);
                    magma_ssetmatrix_async( n12, ib,
                                            Q(0,iil-1), ldq,
                                            dS(igpu,0), n12, queues[igpu][0] );
                }
            }

            for (i = 0; i < rk; i += nb) {
                ib = min(nb, rk - i);
                ind = (i/nb)%2;
                if (i+nb < rk) {
                    ib2 = min(nb, rk - i - nb);
                    for (igpu = 0; igpu < ngpu-1; igpu += 2) {
                        if (n23 != 0) {
                            magma_setdevice(igpu+1);
                            magma_ssetmatrix_async( n23, ib2,
                                                    Q(ctot[0],iil-1+i+nb), ldq,
                                                    dS(igpu+1,(ind+1)%2),  n23, queues[igpu+1][(ind+1)%2] );
                        }
                        if (n12 != 0) {
                            magma_setdevice(igpu);
                            magma_ssetmatrix_async( n12, ib2,
                                                    Q(0,iil-1+i+nb),    ldq,
                                                    dS(igpu,(ind+1)%2), n12, queues[igpu][(ind+1)%2] );
                        }
                    }
                }

                // Ensure that the data is copied on gpu since we will overwrite it.
                for (igpu = 0; igpu < ngpu-1; igpu += 2) {
                    if (n23 != 0) {
#ifdef CHECK_CPU
                        lapackf77_slacpy("A", &n23, &ib, Q(ctot[0],iil-1+i), &ldq, hS(igpu+1,ind), &n23);
#endif
                        magma_setdevice(igpu+1);
                        magma_queue_sync( queues[igpu+1][ind] );
                    }
                    if (n12 != 0) {
#ifdef CHECK_CPU
                        lapackf77_slacpy("A", &n12, &ib, Q(0,iil-1+i), &ldq, hS(igpu,ind), &n12);
#endif
                        magma_setdevice(igpu);
                        magma_queue_sync( queues[igpu][ind] );
                    }
                }
                for (igpu = 0; igpu < ngpu-1; igpu += 2) {
                    if (n23 != 0) {
#ifdef CHECK_CPU
                        blasf77_sgemm("N", "N", &ni_loc[igpu+1], &ib, &n23, &d_one, hQ2(igpu+1), &n2_loc,
                                      hS(igpu+1,ind), &n23, &d_zero, hQ(igpu+1, ind), &n2_loc);
#endif
                        magma_setdevice(igpu+1);
                        magmablasSetKernelStream(queues[igpu+1][ind]);
                        magma_sgemm(MagmaNoTrans, MagmaNoTrans, ni_loc[igpu+1], ib, n23, d_one, dQ2(igpu+1), n2_loc,
                                    dS(igpu+1, ind), n23, d_zero, dQ(igpu+1, ind), n2_loc);
#ifdef CHECK_CPU
                        printf("norm Q %d: %f\n", igpu+1, cpu_gpu_sdiff(ni_loc[igpu+1], ib, hQ(igpu+1, ind), n2_loc, dQ(igpu+1, ind), n2_loc));
#endif
                    }
                    if (n12 != 0) {
#ifdef CHECK_CPU
                        blasf77_sgemm("N", "N", &ni_loc[igpu], &ib, &n12, &d_one, hQ2(igpu), &n1_loc,
                                      hS(igpu,ind%2), &n12, &d_zero, hQ(igpu, ind%2), &n1_loc);
#endif
                        magma_setdevice(igpu);
                        magmablasSetKernelStream(queues[igpu][ind]);
                        magma_sgemm(MagmaNoTrans, MagmaNoTrans, ni_loc[igpu], ib, n12, d_one, dQ2(igpu), n1_loc,
                                    dS(igpu, ind), n12, d_zero, dQ(igpu, ind), n1_loc);
#ifdef CHECK_CPU
                        printf("norm Q %d: %f\n", igpu, cpu_gpu_sdiff(ni_loc[igpu], ib, hQ(igpu, ind), n1_loc, dQ(igpu, ind), n1_loc));
#endif
                    }
                }
                for (igpu = 0; igpu < ngpu-1; igpu += 2) {
                    if (n23 != 0) {
                        magma_setdevice(igpu+1);
                        magma_sgetmatrix( ni_loc[igpu+1], ib, dQ(igpu+1, ind), n2_loc,
                                          Q(n1+n2_loc*(igpu/2),iil-1+i), ldq );
//                        magma_sgetmatrix_async( ni_loc[igpu+1], ib, dQ(igpu+1, ind), n2_loc,
//                                                Q(n1+n2_loc*(igpu/2),iil-1+i), ldq, queues[igpu+1][ind] );
                    }
                    if (n12 != 0) {
                        magma_setdevice(igpu);
                        magma_sgetmatrix( ni_loc[igpu], ib, dQ(igpu, ind), n1_loc,
                                          Q(n1_loc*(igpu/2),iil-1+i), ldq );
//                        magma_sgetmatrix_async( ni_loc[igpu], ib, dQ(igpu, ind), n1_loc,
//                                                Q(n1_loc*(igpu/2),iil-1+i), ldq, queues[igpu][ind] );
                    }
                }
            }
            for (igpu = 0; igpu < ngpu; ++igpu) {
#ifdef CHECK_CPU
                magma_free_pinned( hwS[1][igpu] );
                magma_free_pinned( hwS[0][igpu] );
                magma_free_pinned( hwQ2[igpu] );
                magma_free_pinned( hwQ[1][igpu] );
                magma_free_pinned( hwQ[0][igpu] );
#endif
                magma_setdevice(igpu);
                magma_queue_sync( queues[igpu][0] );
                magma_queue_sync( queues[igpu][1] );
            }
            if ( n23 == 0 )
                lapackf77_slaset("A", &n2, &rk, &d_zero, &d_zero, Q(n1,iil-1), &ldq);

            if ( n12 == 0 )
                lapackf77_slaset("A", &n1, &rk, &d_zero, &d_zero, Q(0,iil-1), &ldq);
        }
    }
    timer_stop( time );
    timer_printf( "gemms = %6.2f\n", time );

    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );
    
    return *info;
} /* magma_slaed3_m */
