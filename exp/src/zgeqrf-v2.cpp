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
magma_zgeqrf2(magma_context *cntxt, magma_int_t m, magma_int_t n, 
          cuDoubleComplex *a,    magma_int_t lda, cuDoubleComplex *tau, 
          cuDoubleComplex *work, magma_int_t lwork,
          magma_int_t *info)
{
/*  -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

    Purpose
    =======
    ZGEQRF computes a QR factorization of a COMPLEX_16 M-by-N matrix A:
    A = Q * R. This version does not require work space on the GPU
    passed as input. GPU memory is allocated in the routine.

    Arguments
    =========
    CNTXT   (input) MAGMA_CONTEXT
            CNTXT specifies the MAGMA hardware context for this routine.

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

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

        Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using cudaMallocHost.

    LWORK   (input) INTEGER
            The dimension of the array WORK.  LWORK >= N*NB,
            where NB can be obtained through magma_get_zgeqrf_nb(M).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  if INFO = -8, the GPU memory allocation failed

    Further Details
    ===============
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).
    =====================================================================    */

    #define  a_ref(a_1,a_2) ( a+(a_2)*(lda) + (a_1))
    #define da_ref(a_1,a_2) (da+(a_2)*ldda  + (a_1))

    int cnt=-1;
    cuDoubleComplex c_one = MAGMA_Z_ONE;

    int i, k, lddwork, old_i, old_ib;
    int nbmin, nx, ib, ldda;

    *info = 0;

    magma_qr_params *qr_params = (magma_qr_params *)cntxt->params;
    int nb = qr_params->nb;

    int lwkopt = n * nb;
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

    cublasStatus status;
    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    nbmin = 2;
    nx = nb;

    lddwork = ((n+31)/32)*32;
    ldda    = ((m+31)/32)*32;

    cuDoubleComplex *da;
    status = cublasAlloc((n)*ldda + nb*lddwork, sizeof(cuDoubleComplex), (void**)&da);
    if (status != CUBLAS_STATUS_SUCCESS) {
        *info = -8;
        return 0;
    }
    cuDoubleComplex *dwork = da + ldda*(n);

    if (nb >= nbmin && nb < k && nx < k) {
        /* Use blocked code initially */
        cudaMemcpy2DAsync(da_ref(0,nb), ldda*sizeof(cuDoubleComplex),
                           a_ref(0,nb), lda *sizeof(cuDoubleComplex),
                          sizeof(cuDoubleComplex)*(m), (n-nb),
                          cudaMemcpyHostToDevice,stream[0]);

        old_i = 0; old_ib = nb;
        for (i = 0; i < k-nx; i += nb) {
            ib = min(k-i, nb);
            if (i>0){
                cudaMemcpy2DAsync( a_ref(i,i),  lda *sizeof(cuDoubleComplex),
                                   da_ref(i,i), ldda*sizeof(cuDoubleComplex),
                                   sizeof(cuDoubleComplex)*(m-i), ib,
                                   cudaMemcpyDeviceToHost,stream[1]);

                cudaMemcpy2DAsync( a_ref(0,i),  lda *sizeof(cuDoubleComplex),
                                   da_ref(0,i), ldda*sizeof(cuDoubleComplex),
                                   sizeof(cuDoubleComplex)*i, ib,
                                   cudaMemcpyDeviceToHost,stream[0]);

                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise, 
                  m-old_i, n-old_i-2*old_ib, old_ib,
                  da_ref(old_i, old_i),          ldda, dwork,        lddwork,
                  da_ref(old_i, old_i+2*old_ib), ldda, dwork+old_ib, lddwork);
            }

            cudaStreamSynchronize(stream[1]);
            int rows = m-i;

        cnt++;
        cntxt->nb = qr_params->ib;
        magma_zgeqrf_mc(cntxt, &rows, &ib, a_ref(i,i), &lda, 
                tau+i, work, &lwork, info);
        cntxt->nb = nb;

            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr, 
                              &rows, &ib, a_ref(i,i), &lda, tau+i, qr_params->t+cnt*nb*nb, &ib);
        if (cnt < qr_params->np_gpu) {
          qr_params->p[cnt]=a;
        }
        zpanel_to_q(MagmaUpper, ib, a_ref(i,i), lda, qr_params->w+cnt*qr_params->nb*qr_params->nb);
            cublasSetMatrix(rows, ib, sizeof(cuDoubleComplex),
                            a_ref(i,i), lda, da_ref(i,i), ldda);
        if (qr_params->flag == 1)
          zq_to_panel(MagmaUpper, ib, a_ref(i,i), lda, qr_params->w+cnt*qr_params->nb*qr_params->nb);
        
            if (i + ib < n) { 
          cublasSetMatrix(ib, ib, sizeof(cuDoubleComplex), qr_params->t+cnt*nb*nb, ib, dwork, lddwork);

          if (i+ib < k-nx)
        /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
        magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise, 
                  rows, ib, ib, 
                  da_ref(i, i   ), ldda, dwork,    lddwork, 
                  da_ref(i, i+ib), ldda, dwork+ib, lddwork);
          else
        magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise, 
                  rows, n-i-ib, ib, 
                  da_ref(i, i   ), ldda, dwork,    lddwork, 
                  da_ref(i, i+ib), ldda, dwork+ib, lddwork);

          old_i  = i;
          old_ib = ib;
            }
        }
    } else {
      i = 0;
    }
    
    /* Use unblocked code to factor the last or only block. */
    if (i < k) 
      {
    ib = n-i;
    if (i!=0)
      cublasGetMatrix(m, ib, sizeof(cuDoubleComplex),
              da_ref(0,i), ldda, a_ref(0,i), lda);
        int rows = m-i;
    
        cnt++;
        lapackf77_zgeqrf(&rows, &ib, a_ref(i,i), &lda, tau+i, work, &lwork, info);
    
    if (cnt < qr_params->np_gpu) 
      {
        int ib2=min(ib,nb);
        
        lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr, 
                              &rows, &ib2, a_ref(i,i), &lda, tau+i, qr_params->t+cnt*nb*nb, &ib2);
        
        qr_params->p[cnt]=a;
      }
      }
    
    cudaStreamDestroy( stream[0] );
    cudaStreamDestroy( stream[1] );
    cublasFree(da);
    return MAGMA_SUCCESS;
} /* magma_zgeqrf */

