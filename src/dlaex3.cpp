/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013
       
       @author Raffaele Solca
       
       @precisions normal d -> s
*/

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common_magma.h"
#include <cblas.h>

#define Q(ix, iy) (q + (ix) + ldq * (iy))

extern "C"{
    int magma_get_dlaed3_k() { return 512;}
    
    void magma_dvrange(magma_int_t k, double *d, magma_int_t *il, magma_int_t *iu, double vl, double vu)
    {
        magma_int_t i;

        *il=1;
        *iu=k;
        for (i = 0; i < k; ++i){
            if (d[i] > vu){
                *iu = i;
                break;
            }
            else if (d[i] < vl)
                ++*il;
        }
        return;
    }

    void magma_dirange(magma_int_t k, magma_int_t* indxq, magma_int_t *iil, magma_int_t *iiu, magma_int_t il, magma_int_t iu)
    {
        magma_int_t i;

        *iil = 1;
        *iiu = 0;
        for (i = il; i<=iu; ++i)
            if (indxq[i-1]<=k){
                *iil = indxq[i-1];
                break;
            }
        for (i = iu; i>=il; --i)
            if (indxq[i-1]<=k){
                *iiu = indxq[i-1];
                break;
            }
        return;
    }
}

extern "C" magma_int_t
magma_dlaex3(magma_int_t k, magma_int_t n, magma_int_t n1, double* d,
             double* q, magma_int_t ldq, double rho,
             double* dlamda, double* q2, magma_int_t* indx,
             magma_int_t* ctot, double* w, double* s, magma_int_t* indxq,
             double* dwork,
             char range, double vl, double vu, magma_int_t il, magma_int_t iu,
             magma_int_t* info )
{
/*
    Purpose
    =======
    DLAEX3 finds the roots of the secular equation, as defined by the
    values in D, W, and RHO, between 1 and K.  It makes the
    appropriate calls to DLAED4 and then updates the eigenvectors by
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
    =========
    K       (input) INTEGER
            The number of terms in the rational function to be solved by
            DLAED4.  K >= 0.

    N       (input) INTEGER
            The number of rows and columns in the Q matrix.
            N >= K (deflation may result in N>K).

    N1      (input) INTEGER
            The location of the last eigenvalue in the leading submatrix.
            min(1,N) <= N1 <= N/2.

    D       (output) DOUBLE PRECISION array, dimension (N)
            D(I) contains the updated eigenvalues for
            1 <= I <= K.

    Q       (output) DOUBLE PRECISION array, dimension (LDQ,N)
            Initially the first K columns are used as workspace.
            On output the columns ??? to ??? contain
            the updated eigenvectors.

    LDQ     (input) INTEGER
            The leading dimension of the array Q.  LDQ >= max(1,N).

    RHO     (input) DOUBLE PRECISION
            The value of the parameter in the rank one update equation.
            RHO >= 0 required.

    DLAMDA  (input/output) DOUBLE PRECISION array, dimension (K)
            The first K elements of this array contain the old roots
            of the deflated updating problem.  These are the poles
            of the secular equation. May be changed on output by
            having lowest order bit set to zero on Cray X-MP, Cray Y-MP,
            Cray-2, or Cray C-90, as described above.

    Q2      (input) DOUBLE PRECISION array, dimension (LDQ2, N)
            The first K columns of this matrix contain the non-deflated
            eigenvectors for the split problem.

    INDX    (input) INTEGER array, dimension (N)
            The permutation used to arrange the columns of the deflated
            Q matrix into three groups (see DLAED2).
            The rows of the eigenvectors found by DLAED4 must be likewise
            permuted before the matrix multiply can take place.

    CTOT    (input) INTEGER array, dimension (4)
            A count of the total number of the various types of columns
            in Q, as described in INDX.  The fourth column type is any
            column which has been deflated.

    W       (input/output) DOUBLE PRECISION array, dimension (K)
            The first K elements of this array contain the components
            of the deflation-adjusted updating vector. Destroyed on
            output.

    S       (workspace) DOUBLE PRECISION array, dimension (N1 + 1)*K
            Will contain the eigenvectors of the repaired matrix which
            will be multiplied by the previously accumulated eigenvectors
            to update the system.

    INDXQ   (output) INTEGER array, dimension (N)
            On exit, the permutation which will reintegrate the
            subproblems back into sorted order,
            i.e. D( INDXQ( I = 1, N ) ) will be in ascending order.

    DWORK   (device workspace) DOUBLE PRECISION array, dimension (3*N*N/2+3*N)

    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an eigenvalue did not converge

    Further Details
    ===============
    Based on contributions by
    Jeff Rutter, Computer Science Division, University of California
    at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    ===================================================================== */

    double d_one  = 1.;
    double d_zero = 0.;
    magma_int_t ione = 1;
    magma_int_t ineg_one = -1;
    char range_[] = {range, 0};

    magma_int_t iil, iiu, rk;

    double* dq2= dwork;
    double* ds = dq2  + n*(n/2+1);
    double* dq = ds   + n*(n/2+1);
    magma_int_t lddq = n/2 + 1;

    magma_int_t i,iq2,j,n12,n2,n23,tmp,lq2;
    double temp;
    magma_int_t alleig, valeig, indeig;

    alleig = lapackf77_lsame(range_, "A");
    valeig = lapackf77_lsame(range_, "V");
    indeig = lapackf77_lsame(range_, "I");

    *info = 0;

    if(k < 0)
        *info=-1;
    else if(n < k)
        *info=-2;
    else if(ldq < max(1,n))
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


    if(*info != 0){
        magma_xerbla(__func__, -(*info));
        return MAGMA_ERR_ILLEGAL_VALUE;
    }

    // Quick return if possible
    if(k == 0)
        return MAGMA_SUCCESS;
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

    n2 = n - n1;

    n12 = ctot[0] + ctot[1];
    n23 = ctot[1] + ctot[2];

    iq2 = n1 * n12;
    lq2 = iq2 + n2 * n23;

    magma_dsetvector_async( lq2, q2, 1, dq2, 1, NULL );

#ifdef _OPENMP
    /////////////////////////////////////////////////////////////////////////////////
    //openmp implementation
    /////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
    magma_timestr_t start, end;
    start = get_current_time();
#endif

#pragma omp parallel private(i, j, tmp, temp)
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t tot = omp_get_num_threads();

        magma_int_t ib = (  id   * k) / tot; //start index of local loop
        magma_int_t ie = ((id+1) * k) / tot; //end index of local loop
        magma_int_t ik = ie - ib;           //number of local indices

        for(i = ib; i < ie; ++i)
            dlamda[i]=lapackf77_dlamc3(&dlamda[i], &dlamda[i]) - dlamda[i];

        for(j = ib; j < ie; ++j){
            magma_int_t tmpp=j+1;
            magma_int_t iinfo = 0;
            lapackf77_dlaed4(&k, &tmpp, dlamda, w, Q(0,j), &rho, &d[j], &iinfo);
            // If the zero finder fails, the computation is terminated.
            if(iinfo != 0){
#pragma omp critical (info)
                *info=iinfo;
                break;
            }
        }

#pragma omp barrier

        if(*info == 0){

#pragma omp single
            {
                //Prepare the INDXQ sorting permutation.
                magma_int_t nk = n - k;
                lapackf77_dlamrg( &k, &nk, d, &ione , &ineg_one, indxq);

                //compute the lower and upper bound of the non-deflated eigenvectors
                if (valeig)
                    magma_dvrange(k, d, &iil, &iiu, vl, vu);
                else if (indeig)
                    magma_dirange(k, indxq, &iil, &iiu, il, iu);
                else {
                    iil = 1;
                    iiu = k;
                }
                rk = iiu - iil + 1;
            }

            if (k == 2){
#pragma omp single
                {
                    for(j = 0; j < k; ++j){
                        w[0] = *Q(0,j);
                        w[1] = *Q(1,j);

                        i = indx[0] - 1;
                        *Q(0,j) = w[i];
                        i = indx[1] - 1;
                        *Q(1,j) = w[i];
                    }
                }

            }
            else if(k != 1){

                // Compute updated W.
                blasf77_dcopy( &ik, &w[ib], &ione, &s[ib], &ione);

                // Initialize W(I) = Q(I,I)
                tmp = ldq + 1;
                blasf77_dcopy( &ik, Q(ib,ib), &tmp, &w[ib], &ione);

                for(j = 0; j < k; ++j){
                    magma_int_t i_tmp = min(j, ie);
                    for(i = ib; i < i_tmp; ++i)
                        w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
                    i_tmp = max(j+1, ib);
                    for(i = i_tmp; i < ie; ++i)
                        w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
                }

                for(i = ib; i < ie; ++i)
                    w[i] = copysign( sqrt( -w[i] ), s[i]);

#pragma omp barrier

                //reduce the number of used threads to have enough S workspace
                tot = min(n1, omp_get_num_threads());

                if(id < tot){
                    ib = (  id   * rk) / tot + iil - 1;
                    ie = ((id+1) * rk) / tot + iil - 1;
                    ik = ie - ib;
                }
                else{
                    ib = -1;
                    ie = -1;
                    ik = -1;
                }

                // Compute eigenvectors of the modified rank-1 modification.
                for(j = ib; j < ie; ++j){
                    for(i = 0; i < k; ++i)
                        s[id*k + i] = w[i] / *Q(i,j);
                    temp = cblas_dnrm2( k, s+id*k, 1);
                    for(i = 0; i < k; ++i){
                        magma_int_t iii = indx[i] - 1;
                        *Q(i,j) = s[id*k + iii] / temp;
                    }
                }
            }
        }
    }
    if (*info != 0)
        return MAGMA_SUCCESS; //??????

#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
    end = get_current_time();
    printf("eigenvalues/vector D+zzT = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

#else
    /////////////////////////////////////////////////////////////////////////////////
    // Non openmp implementation
    /////////////////////////////////////////////////////////////////////////////////
#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
    magma_timestr_t start, end;
    start = get_current_time();
#endif

    for(i = 0; i < k; ++i)
        dlamda[i]=lapackf77_dlamc3(&dlamda[i], &dlamda[i]) - dlamda[i];

    for(j = 0; j < k; ++j){
        magma_int_t tmpp=j+1;
        magma_int_t iinfo = 0;
        lapackf77_dlaed4(&k, &tmpp, dlamda, w, Q(0,j), &rho, &d[j], &iinfo);
        // If the zero finder fails, the computation is terminated.
        if(iinfo != 0)
            *info=iinfo;
    }
    if(*info != 0)
        return MAGMA_SUCCESS;

    //Prepare the INDXQ sorting permutation.
    magma_int_t nk = n - k;
    lapackf77_dlamrg( &k, &nk, d, &ione , &ineg_one, indxq);

    //compute the lower and upper bound of the non-deflated eigenvectors
    if (valeig)
        magma_dvrange(k, d, &iil, &iiu, vl, vu);
    else if (indeig)
        magma_dirange(k, indxq, &iil, &iiu, il, iu);
    else {
        iil = 1;
        iiu = k;
    }
    rk = iiu - iil + 1;

    if (k == 2){

        for(j = 0; j < k; ++j){
            w[0] = *Q(0,j);
            w[1] = *Q(1,j);

            i = indx[0] - 1;
            *Q(0,j) = w[i];
            i = indx[1] - 1;
            *Q(1,j) = w[i];
        }

    }
    else if(k != 1){

        // Compute updated W.
        blasf77_dcopy( &k, w, &ione, s, &ione);

        // Initialize W(I) = Q(I,I)
        tmp = ldq + 1;
        blasf77_dcopy( &k, q, &tmp, w, &ione);

        for(j = 0; j < k; ++j){
            for(i = 0; i < j; ++i)
                w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
            for(i = j+1; i < k; ++i)
                w[i] = w[i] * ( *Q(i, j) / ( dlamda[i] - dlamda[j] ) );
        }

        for(i = 0; i < k; ++i)
            w[i] = copysign( sqrt( -w[i] ), s[i]);

        // Compute eigenvectors of the modified rank-1 modification.
        for(j = iil-1; j < iiu; ++j){
            for(i = 0; i < k; ++i)
                s[i] = w[i] / *Q(i,j);
            temp = cblas_dnrm2( k, s, 1);
            for(i = 0; i < k; ++i){
                magma_int_t iii = indx[i] - 1;
                *Q(i,j) = s[iii] / temp;
            }
        }
    }

#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
    end = get_current_time();
    printf("eigenvalues/vector D+zzT = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

#endif //_OPENMP
    // Compute the updated eigenvectors.

#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
    start = get_current_time();
#endif
    magma_queue_sync( NULL );

    if (rk != 0){
        if( n23 != 0 ){
            if (rk < magma_get_dlaed3_k()){
                lapackf77_dlacpy("A", &n23, &rk, Q(ctot[0],iil-1), &ldq, s, &n23);
                blasf77_dgemm("N", "N", &n2, &rk, &n23, &d_one, &q2[iq2], &n2,
                              s, &n23, &d_zero, Q(n1,iil-1), &ldq );
            } else {
                magma_dsetmatrix( n23, rk, Q(ctot[0],iil-1), ldq, ds, n23 );
                magma_dgemm('N', 'N', n2, rk, n23, d_one, &dq2[iq2], n2, ds, n23, d_zero, dq, lddq);
                magma_dgetmatrix( n2, rk, dq, lddq, Q(n1,iil-1), ldq );
            }
        } else
            lapackf77_dlaset("A", &n2, &rk, &d_zero, &d_zero, Q(n1,iil-1), &ldq);

        if( n12 != 0 ) {
            if (rk < magma_get_dlaed3_k()){
                lapackf77_dlacpy("A", &n12, &rk, Q(0,iil-1), &ldq, s, &n12);
                blasf77_dgemm("N", "N", &n1, &rk, &n12, &d_one, q2, &n1,
                              s, &n12, &d_zero, Q(0,iil-1), &ldq);
            } else {
                magma_dsetmatrix( n12, rk, Q(0,iil-1), ldq, ds, n12 );
                magma_dgemm('N', 'N', n1, rk, n12, d_one, dq2, n1, ds, n12, d_zero, dq, lddq);
                magma_dgetmatrix( n1, rk, dq, lddq, Q(0,iil-1), ldq );
            }
        } else
            lapackf77_dlaset("A", &n1, &rk, &d_zero, &d_zero, Q(0,iil-1), &ldq);
    }
#ifdef ENABLE_TIMER_DIVIDE_AND_CONQUER
    end = get_current_time();
    printf("gemms = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

    return MAGMA_SUCCESS;
} /*magma_dlaed3*/
