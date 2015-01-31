/**
 *
 * @file core_zgetrf_reclap.c
 *
 *  PLASMA core_blas kernel
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.5.0
 * @author Hatem Ltaief
 * @author Mathieu Faverge
 * @author Piotr Luszczek
 * @author Simplice Donfack (GPU adaptation)
 * @date January 2015
 *
 * @precisions normal z -> c d s
 *
 **/
#include <math.h>
#if (dbglevel >=1)
#include "ca_dbg_tools.h"
#endif

#ifndef USE_CUDBG
#include <cblas.h>
//#include <mkl.h>
#endif

#include "common_magma.h"

/*Some CBLAS function that need a translation*/
#define cblas_izamax(a,b,c) cblas_idamax((a),(b),(c))
#define cblas_ztrsm  cblas_dtrsm
#define cblas_zgemm  cblas_dgemm
#define cblas_zscal  cblas_dscal
#define cabs(a) ((a)>=0?(a):-(a))

#ifndef LAPACKE_dlamch_work
double LAPACKE_dlamch_work( char cmach )
{
  int info = 0;
  double res;
  /* Call LAPACK function and adjust info */
  res = dlamch_( &cmach );
  return res;
 }
#endif

//void CORE_zgetrf_reclap_init();

static inline void
CORE_zgetrf_reclap_update(int M, int column, int n1, int n2,
                          double *A, int LDA, int *IPIV,
                          int thidx, int thcnt);
static inline void
CORE_zgetrf_reclap_rec(int M, int N,
                       double *A, int LDA,
                       int *IPIV, int *info,
                       int thidx, int thcnt, int column);

/***************************************************************************//**
 *
 * @ingroup CORE_double
 *
 *  CORE_zgetrf_reclap computes a LU factorization of a general M-by-N
 *  matrix A stored in CCRB layout using partial pivoting with row
 *  interchanges.
 *
 *  The factorization has the form
 *
 *    A = P * L * U
 *
 *  where P is a permutation matrix, L is lower triangular with unit
 *  diagonal elements (lower trapezoidal if m > n), and U is upper
 *  triangular (upper trapezoidal if m < n).
 *
 *  This is the recursive version of the algorithm applied on column
 *  major layout.
 *
 *  WARNINGS:
 *     - The function CORE_zgetrf_reclap_init has to be called prior
 *     to any call to this function.
 *     - You cannot call this kernel on different matrices at the same
 *     time.
 *     - The matrix A cannot be more than one tile wide.
 *     - The number of threads calling this function has to be excatly
 *     the number defined by info[2] with each one of them a different
 *     index between 0 included and info[2] excluded.
 *
 *******************************************************************************
 *
 *  @param[in] M
 *          The number of rows of the matrix A.  M >= 0.
 *
 *  @param[in] N
 *          The number of columns of the matrix A.  N >= 0.
 *
 *  @param[in,out] A
 *          On entry, the M-by-N matrix to be factorized.
 *          On exit, the factors L and U from the factorization
 *          A = P*L*U; the unit diagonal elements of L are not stored.
 *
 *  @param[in] LDA
 *          The leading dimension of the array A.  LDA >= max(1,M).
 *
 *  @param[out] IPIV
 *          The pivot indices; for 1 <= i <= min(M,N), row i of the
 *          matrix was interchanged with row IPIV(i).
 *          1 <= IPIV[i] <= M.
 *
 *  @param[in,out] info
 *          Array of 3 integers
 *          - info[0], see returned value
 *          - info[1], is the thread index 0 <= info[0] < info[2]
 *          - info[2], on entry is the number of threads trying to
 *                     participate to the factorization,
 *                     on exit is the real number of threads used to
 *                     perform the factorization.
 *
 *******************************************************************************
 *
 * @return
 *          \retval PLASMA_SUCCESS successful exit
 *          \retval -k, the k-th argument had an illegal value
 *          \retval k if U(k,k) is exactly zero. The factorization
 *                  has been completed, but the factor U is exactly
 *                  singular, and division by zero will occur if it is used
 *                  to solve a system of equations.
 *
 */
#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_zgetrf_reclap = PCORE_zgetrf_reclap
#define CORE_zgetrf_reclap PCORE_zgetrf_reclap
#endif
int CORE_zgetrf_reclap(int M, int N,
                       double *A, int LDA,
                       int *IPIV, int *info)
{
    int thidx = info[1];
    int thcnt = min( info[2], M / N );
    int minMN = min(M, N);

    info[0] = 0;
    info[2] = thcnt;
    
    if( M < 0 ) {
       // coreblas_error(1, "illegal value of M");
        return -1;
    }
    if( N < 0 ) {
        //coreblas_error(2, "illegal value of N");
        return -2;
    }
    if( LDA < max(1, M) ) {
        //coreblas_error(5, "illegal value of LDA");
        return -5;
    }
    
 
    if ( (M == 0) || (N == 0) || (thidx >= thcnt) ){
      return 0; //PLASMA_SUCCESS;
    }
    
    *info = 0;
    CORE_zgetrf_reclap_rec( M, minMN, A, LDA, IPIV, info,
                            thidx, thcnt, 0 );

    if ( N > minMN ) {
        CORE_zgetrf_reclap_update(M, 0, minMN, N-minMN,
                                  A, LDA, IPIV,
                                  thidx, thcnt);
    }

    return info[0];
}



/***************************************************************************//**
 *
 **/
#if defined(PLASMA_HAVE_WEAK)
#pragma weak CORE_zgetrf_reclap_quark = PCORE_zgetrf_reclap_quark
#define CORE_zgetrf_reclap_quark PCORE_zgetrf_reclap_quark
#endif


/*******************************************************************
 *   Additional routines
 */
#define AMAX1BUF_SIZE (48 << 1)
/* 48 threads should be enough for everybody */
static volatile double CORE_zamax1buf[AMAX1BUF_SIZE];
static double sfmin;

void CORE_zgetrf_reclap_init() {
    int i;

    for (i = 0; i < AMAX1BUF_SIZE; ++i) CORE_zamax1buf[i] = -1.0;
    sfmin =  LAPACKE_dlamch_work('S');
}

static inline void
psplit(int n, int pidx, int pcnt, int *poff_p, int *psiz_p)
{
    int q = n / pcnt, r = n % pcnt;

    if (pidx < r) {
        q++;
        *psiz_p = q;
        *poff_p = pidx * q;
    } else {
        *psiz_p = q;
        *poff_p = r * (q + 1) + (pidx - r) * q;
    }
}

static inline void
CORE_zamax1_thread(double localamx,
                   int thidx, int thcnt, int *thwinner,
                   double *globalamx,
                   int pividx, int *ipiv)
{
    if (thidx == 0) {
        int i, j = 0;
        double curval = localamx, tmp;
        double curamx = cabs(localamx);

        /* make sure everybody filled in their value */
        for (i = 1; i < thcnt; ++i) {
            while (CORE_zamax1buf[i << 1] == -1.0) { /* wait for thread i to store its value */
            }
        }

        /* better not fuse the loop above and below to make sure data is sync'd */

        for (i = 1; i < thcnt; ++i) {
            tmp = CORE_zamax1buf[ (i << 1) + 1];
            if (cabs(tmp) > curamx) {
                curamx = cabs(tmp);
                curval = tmp;
                j = i;
            }
        }

        if (0 == j)
            ipiv[0] = pividx;

        /* make sure everybody knows the amax value */
        for (i = 1; i < thcnt; ++i)
            CORE_zamax1buf[ (i << 1) + 1] = curval;

        CORE_zamax1buf[0] = -j - 2.0; /* set the index of the winning thread */

        *thwinner = j;
        *globalamx = curval;

        for (i = 1; i < thcnt; ++i)
            CORE_zamax1buf[i << 1] = -3.0;

        /* make sure everybody read the max value */
        for (i = 1; i < thcnt; ++i) {
            while (CORE_zamax1buf[i << 1] != -1.0) {
            }
        }

        CORE_zamax1buf[0] = -1.0;
    } else {
        CORE_zamax1buf[(thidx << 1) + 1] = localamx;
        CORE_zamax1buf[thidx << 1] = -2.0;  /* announce to thread 0 that local amax was stored */
        while (CORE_zamax1buf[0] == -1.0) { /* wait for thread 0 to finish calculating the global amax */
        }
        while (CORE_zamax1buf[thidx << 1] != -3.0) { /* wait for thread 0 to store amax */
        }
        *globalamx = CORE_zamax1buf[(thidx << 1) + 1]; /* read the amax from the location adjacent to the one in the above loop */
        *thwinner = -CORE_zamax1buf[0] - 2.0;
        CORE_zamax1buf[thidx << 1] = -1.0;  /* signal thread 0 that this thread is done reading */

        if (thidx == *thwinner)
            ipiv[0] = pividx;

        while (CORE_zamax1buf[0] != -1.0) { /* wait for thread 0 to finish */
        }
    }
}

static inline void
CORE_zbarrier_thread(int thidx, int thcnt)
{
    int idum1, idum2;
    double ddum2;
    /* it's probably faster to implement a dedicated barrier */
    CORE_zamax1_thread( 1.0, thidx, thcnt, &idum1, &ddum2, 0, &idum2 );
}

static inline void
CORE_zlaswap1(int ncol, double *a, int lda,
              int idxStart, int idxMax, const int *piv)
{
    int i, j;
    double tmp;

    for (j = 0; j < ncol; ++j) {
        for (i = idxStart; i < idxMax; ++i) {
            tmp = a[j*lda + piv[i] - 1];
            a[j*lda + piv[i] - 1] = a[i + j*lda];
            a[i + j*lda] = tmp;
        }
    }
}

static inline void
CORE_zgetrf_reclap_update(int M, int column, int n1, int n2,
                          double *A, int LDA, int *IPIV,
                          int thidx, int thcnt)
{
    static double posone =  1.0;
    static double negone = -1.0;
    double *Atop  = A    + column*LDA;
    double *Atop2 = Atop + n1    *LDA;
    int coff, ccnt, lm, loff;

    CORE_zbarrier_thread( thidx, thcnt );

    psplit( n2, thidx, thcnt, &coff, &ccnt );

    if (ccnt > 0) {
        CORE_zlaswap1( ccnt, Atop2 + coff*LDA, LDA, column, n1 + column, IPIV ); /* swap to the right */

        cblas_ztrsm( CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                     n1, ccnt, posone, Atop + column, LDA, Atop2 + coff*LDA + column, LDA );
    }

    /* __sync_synchronize(); */ /* hopefully we will not need memory fences */

    /* need to wait for pivoting and triangular solve to finish */
    CORE_zbarrier_thread( thidx, thcnt );

    psplit( M, thidx, thcnt, &loff, &lm );
    if (thidx == 0) {
        loff = column + n1;
        lm  -= column + n1;
    };

    cblas_zgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, lm, n2, n1,
                 negone, Atop+loff, LDA, Atop2 + column, LDA, posone, Atop2+loff, LDA );
}

static void
CORE_zgetrf_reclap_rec(int M, int N,
                       double *A, int LDA,
                       int *IPIV, int *info,
                       int thidx, int thcnt, int column)
{
    int jp, n1, n2, lm, loff;
    double tmp1, tmp2, tmp3;
    double *Atop = A + column*LDA;

    /* Assumption: N = min( M, N ); */
    if (N > 1) {
        int coff, ccnt;

        n1 = N / 2;
        n2 = N - n1;
        //printf("doing recursif m:%d, N:%d, column:%d\n",M,N,column);
        CORE_zgetrf_reclap_rec( M, n1, A, LDA, IPIV, info,
                                thidx, thcnt, column );
        if ( *info != 0 )
            return;

        CORE_zgetrf_reclap_update(M, column, n1, n2,
                                  A, LDA, IPIV,
                                  thidx, thcnt);

        CORE_zgetrf_reclap_rec( M, n2, A, LDA, IPIV, info,
                                thidx, thcnt, column + n1 );
        if ( *info != 0 )
            return;

        psplit( n1, thidx, thcnt, &coff, &ccnt );

        if (ccnt > 0) {
            CORE_zlaswap1( ccnt, Atop+coff*LDA, LDA, n1 + column, N + column, IPIV ); /* swap to the left */
        }

    } else {
        int thrd;

        CORE_zbarrier_thread( thidx, thcnt );

        psplit( M, thidx, thcnt, &loff, &lm );

        if (thidx == 0) {
            loff = column;
            lm -= column;
        }

        tmp2 = Atop[column]; /* all threads read the pivot element in case they need it */

        jp = cblas_izamax( lm, Atop + loff, 1 );
        tmp1 = Atop[loff + jp];

        CORE_zamax1_thread( tmp1, thidx, thcnt, &thrd,
                            &tmp3, loff + jp + 1, IPIV + column );

        Atop[column] = tmp3; /* all threads set the pivot element: no need for synchronization */

        if ( tmp3 != 0.0 ) {
            if ( cabs(tmp3) >= sfmin ) {
                double tmp = (double)1.0 / tmp3;
                n1 = (thidx == 0) ? 1 : 0;
                //cblas_dscal( lm - n1, CBLAS_SADDR(tmp), Atop + loff + n1, 1 );
                cblas_zscal( lm - n1, tmp, Atop + loff + n1, 1 );
            } else {
                int i;
                double *Atop2;
                n1 = (thidx == 0) ? 1 : 0;
                Atop2 = Atop + loff + n1;

                for( i=0; i < lm-n1; i++, Atop2++)
                    *Atop2 = *Atop2 / tmp3;
            }

            if (thrd == thidx) { /* the thread that owns the best pivot */
              if (loff + jp != column) /* if there is a need to exchange the pivot */
                Atop[loff + jp] = tmp2 / tmp3;
            }

        } else {
            *info = column + 1;
            return;
        }

        CORE_zbarrier_thread( thidx, thcnt );
    }
}

