/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"

#define PRECISION_d


extern "C" magma_int_t
magma_dgegqr_gpu( magma_int_t m, magma_int_t n,
                  double *dA,   magma_int_t ldda,
                  double *dwork, double *work,
                  magma_int_t *info )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    ZGEGQR orthogonalizes the N vectors given by a real M-by-N matrix A:
           
            A = Q * R. 

    On exit, if successful, the orthogonal vectors Q overwrite A
    and R is given in work (on the CPU memory).
    
    This version uses normal equations and SVD in an iterative process that 
    makes the computation numerically accurate.
    
    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    dA      (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the M-by-N matrix Q with orthogonal columns.

    LDDA     (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            dividable by 16.

    dwork   (GPU workspace) DOUBLE_PRECISION array, dimension (N,N)
 
    work    (CPU workspace/output) DOUBLE_PRECISION array, dimension 3n^2.
            On exit, work(1:n^2) holds the rectangular matrix R.
            Preferably, for higher performance, work must be in pinned memory.
 
    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ===============



    =====================================================================    */

    magma_int_t i = 0, j, k, n2 = n*n, ione = 1;
    double zero = MAGMA_D_ZERO, one = MAGMA_D_ONE;
    double cn = 200., mins, maxs;

    /* check arguments */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    double *U, *VT, *vt, *R, *G, *hwork, *tau;
    double *S;

    R    = work;             // Size n * n
    G    = R    + n*n;       // Size n * n
    VT   = G    + n*n;       // Size n * n 

    magma_dmalloc_cpu( &hwork, 2*n*n + 2*n);
    if ( hwork == NULL ) {
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
    }

    magma_int_t lwork = n*n; // First part f hwork; used as workspace in svd 

    U    = hwork + n*n;      // Size n*n  
    S    = (double *)(U+n*n);// Size n
    tau  = U + n*n + n  ;    // Size n

    #if defined(PRECISION_c) || defined(PRECISION_z)
    double *rwork;
    magma_dmalloc_cpu( &rwork, 5*n);
    if ( rwork == NULL ) {
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
    }
    #endif

    do {
      i++;

      magma_dgemm(MagmaTrans, MagmaNoTrans, n, n, m, one, dA, ldda, dA, ldda, zero, dwork, n );
      // magmablas_dgemm_reduce(n, n, m, one, dA, ldda, dA, ldda, zero, dwork, n );
      magma_dgetmatrix(n, n, dwork, n, G, n);
      
      #if defined(PRECISION_s) || defined(PRECISION_d) 
         lapackf77_dgesvd("n", "a", &n, &n, G, &n, S, U, &n, VT, &n, 
                          hwork, &lwork, info);
      #else
         lapackf77_dgesvd("n", "a", &n, &n, G, &n, S, U, &n, VT, &n,
                          hwork, &lwork, rwork, info);
      #endif

      mins = 100.f, maxs = 0.f;
      for(k=0; k<n; k++){
        S[k] = magma_dsqrt( S[k] );

        if (S[k] < mins)  mins = S[k];
        if (S[k] > maxs)  maxs = S[k];
      }

      for(k=0; k<n;k++){
        vt = VT + k*n;
        for(j=0; j<n; j++)
          vt[j]*=S[j];
      }
      lapackf77_dgeqrf(&n, &n, VT, &n, tau, hwork, &lwork, info);

      if (i==1)
        blasf77_dcopy(&n2, VT, &ione, R, &ione);
      else
        blasf77_dtrmm("l", "u", "n", "n", &n, &n, &one, VT, &n, R, &n);

      magma_dsetmatrix(n, n, VT, n, G, n);
      magma_dtrsm('r', 'u', 'n', 'n', m, n, one, G, n, dA, ldda);
      if (mins > 0.00001f)
        cn = maxs/mins;

      //fprintf(stderr, "Iteration %d, cond num = %f \n", i, cn);
    } while (cn > 10.f);

    magma_free_cpu( hwork );
    #if defined(PRECISION_c) || defined(PRECISION_z)
    magma_free_cpu( rwork );
    #endif
    
    return *info;
}   /* magma_dgegqr_gpu */
