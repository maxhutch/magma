/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c

*/
#include "common_magma.h"

/* ------------------------------------------------------------
 * MAGMA QR params
 * --------------------------------------------------------- */
typedef struct {

  /* Whether or not to restore upper part of matrix */
  int flag;

  /* Number of MAGMA threads */
  int nthreads;

  /* Block size for left side of matrix */
  int nb;

  /* Block size for right side of matrix */
  int ob;

  /* Block size for final factorization */
  int fb;

  /* Block size for multi-core factorization */
  int ib;

  /* Number of panels for left side of matrix */
  int np_gpu;

  /* Number of rows */
  int m;

  /* Number of columns */
  int n;

  /* Leading dimension */
  int lda;

  /* Matrix to be factorized */
  cuDoubleComplex *a;

  /* Storage for every T */
  cuDoubleComplex *t;

  /* Flags to wake up MAGMA threads */
  volatile cuDoubleComplex **p;

  /* Synchronization flag */
  volatile int sync0;

  /* One synchronization flag for each MAGMA thread */
  volatile int *sync1;
  
  /* Synchronization flag */
  volatile int sync2;

  /* Work space */
  cuDoubleComplex *w;

} magma_qr_params;


extern "C" magma_int_t
magma_zgeqrf3(magma_context *cntxt, magma_int_t m, magma_int_t n, 
          cuDoubleComplex *a,    magma_int_t lda, cuDoubleComplex *tau, 
          cuDoubleComplex *work, magma_int_t lwork,
          magma_int_t *info )
{
/*  -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

    Purpose   
    =======
    ZGEQRF computes a QR factorization of a COMPLEX_16 M-by-N matrix A:   
    A = Q * R.   

    Arguments   
    =========   
    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, the elements on and above the diagonal of the array   
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is   
            upper triangular if m >= n); the elements below the diagonal,   
            with the array TAU, represent the orthogonal matrix Q as a   
            product of min(m,n) elementary reflectors (see Further   
            Details).   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= N*NB. 

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   
    The matrix Q is represented as a product of elementary reflectors   

       Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a complex scalar, and v is a complex vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),   
    and tau in TAU(i).
    ====================================================================    */

    cuDoubleComplex c_one = MAGMA_Z_ONE;
    int k, ib;
    magma_qr_params *qr_params = (magma_qr_params *)cntxt->params;

    *info = 0;

    int lwkopt = n * qr_params->nb;
    work[0] = MAGMA_Z_MAKE( (double)lwkopt, 0 );
    long int lquery = (lwork == -1);
    if (m < 0) {
      *info = -1;
    } else if (n < 0) {
      *info = -2;
    } else if (lda < max(1,m)) {
      *info = -4;
    } else if (lwork < max(1,n) && ! lquery) {
      *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    else if (lquery)
      return MAGMA_SUCCESS;

    k = min(m,n);
    if (k == 0) {
      work[0] = c_one;
      return MAGMA_SUCCESS;
    }
    
    int M=qr_params->nthreads*qr_params->ob;
    int N=qr_params->nthreads*qr_params->ob;
    
    if (qr_params->m > qr_params->n) 
      M = qr_params->m - (qr_params->n-qr_params->nthreads*qr_params->ob);
    
    /* Use MAGMA code to factor left portion of matrix, waking up threads 
       along the way to perform updates on the right portion of matrix */
    magma_zgeqrf2(cntxt, m, n - qr_params->nthreads * qr_params->ob, 
          a, lda, tau, work, lwork, info);

    /* Wait for all update threads to finish */
    for (k = 0; k < qr_params->nthreads; k++) {
      while (qr_params->sync1[k] == 0) {
        sched_yield();
      }
    }
    
    /* Unzero upper part of each panel */
    for (k = 0; k < qr_params->np_gpu-1; k++){
      ib = min(qr_params->nb,(n-qr_params->nthreads*qr_params->ob)-qr_params->nb*k);
      zq_to_panel(MagmaUpper, ib, a + k*qr_params->nb*lda + k*qr_params->nb, lda, 
          qr_params->w+qr_params->nb*qr_params->nb*k);
    }

    /* Use final blocking size */
    qr_params->nb = qr_params->fb;

    /* Flag MAGMA code to internally unzero upper part of each panel */
    qr_params->flag = 1;
    
    /* Use MAGMA code to perform final factorization if necessary */
    if (qr_params->m > (qr_params->n - (qr_params->nthreads*qr_params->ob)))

      if (M > (qr_params->m-(qr_params->n-(qr_params->ob*qr_params->nthreads))))
        M = qr_params->m-(qr_params->n-(qr_params->ob*qr_params->nthreads));

      magma_zgeqrf2(cntxt, M, N,
            a + (n-qr_params->nthreads*qr_params->ob)*m+
            (n-qr_params->nthreads*qr_params->ob), lda, 
            &tau[n-qr_params->nthreads*qr_params->ob],
            work, lwork, info);

    /* Prepare for next run */
    for (k = 0; k < qr_params->np_gpu; k++) {
      qr_params->p[k] = NULL;
    }

    for (k = 0; k < qr_params->nthreads; k++) {
      qr_params->sync1[k] = 0;
    }

    /* Infrastructure for next run is not in place yet */
    qr_params->sync0 = 0;

    /* Signal update threads to get in position for next run */
    qr_params->sync2 = 1;
}

