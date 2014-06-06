/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"

#define PRECISION_c


extern "C" magma_int_t
magma_cgegqr_gpu( magma_int_t m, magma_int_t n,
                  magmaFloatComplex *dA,   magma_int_t ldda,
                  magmaFloatComplex *dwork, magmaFloatComplex *work,
                  magma_int_t *info )
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    ZGEGQR orthogonalizes the N vectors given by a complex M-by-N matrix A:
           
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

    dA      (input/output) COMPLEX array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the M-by-N matrix Q with orthogonal columns.

    LDDA     (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            dividable by 16.

    dwork   (GPU workspace) COMPLEX array, dimension (N,N)
 
    work    (CPU workspace/output) COMPLEX array, dimension 3n^2.
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
    magmaFloatComplex zero = MAGMA_C_ZERO, one = MAGMA_C_ONE;
    float cn = 200., mins, maxs;

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

    magmaFloatComplex *U, *VT, *vt, *R, *G, *hwork, *tau;
    float *S;

    R    = work;             // Size n * n
    G    = R    + n*n;       // Size n * n
    VT   = G    + n*n;       // Size n * n 

    magma_cmalloc_cpu( &hwork, 2*n*n + 2*n);
    if ( hwork == NULL ) {
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
    }

    magma_int_t lwork = n*n; // First part f hwork; used as workspace in svd 

    U    = hwork + n*n;      // Size n*n  
    S    = (float *)(U+n*n);// Size n
    tau  = U + n*n + n  ;    // Size n

    #if defined(PRECISION_c) || defined(PRECISION_z)
    float *rwork;
    magma_smalloc_cpu( &rwork, 5*n);
    if ( rwork == NULL ) {
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
    }
    #endif

    do {
      i++;

      magma_cgemm(MagmaConjTrans, MagmaNoTrans, n, n, m, one, dA, ldda, dA, ldda, zero, dwork, n );
      // magmablas_cgemm_reduce(n, n, m, one, dA, ldda, dA, ldda, zero, dwork, n );
      magma_cgetmatrix(n, n, dwork, n, G, n);
      
      #if defined(PRECISION_s) || defined(PRECISION_d) 
         lapackf77_cgesvd("n", "a", &n, &n, G, &n, S, U, &n, VT, &n, 
                          hwork, &lwork, info);
      #else
         lapackf77_cgesvd("n", "a", &n, &n, G, &n, S, U, &n, VT, &n,
                          hwork, &lwork, rwork, info);
      #endif

      mins = 100.f, maxs = 0.f;
      for(k=0; k<n; k++){
        S[k] = magma_ssqrt( S[k] );

        if (S[k] < mins)  mins = S[k];
        if (S[k] > maxs)  maxs = S[k];
      }

      for(k=0; k<n;k++){
        vt = VT + k*n;
        for(j=0; j<n; j++)
          vt[j]*=S[j];
      }
      lapackf77_cgeqrf(&n, &n, VT, &n, tau, hwork, &lwork, info);

      if (i==1)
        blasf77_ccopy(&n2, VT, &ione, R, &ione);
      else
        blasf77_ctrmm("l", "u", "n", "n", &n, &n, &one, VT, &n, R, &n);

      magma_csetmatrix(n, n, VT, n, G, n);
      magma_ctrsm('r', 'u', 'n', 'n', m, n, one, G, n, dA, ldda);
      if (mins > 0.00001f)
        cn = maxs/mins;

      //fprintf(stderr, "Iteration %d, cond num = %f \n", i, cn);
    } while (cn > 10.f);

    magma_free_cpu( hwork );
    #if defined(PRECISION_c) || defined(PRECISION_z)
    magma_free_cpu( rwork );
    #endif
    
    return *info;
}   /* magma_cgegqr_gpu */
