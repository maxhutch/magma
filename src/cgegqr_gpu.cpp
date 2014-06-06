/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zgegqr_gpu.cpp normal z -> c, Fri May 30 10:40:57 2014

*/
#include "common_magma.h"

#define PRECISION_c


/**
    Purpose
    -------
    CGEGQR orthogonalizes the N vectors given by a complex M-by-N matrix A:
           
            A = Q * R.

    On exit, if successful, the orthogonal vectors Q overwrite A
    and R is given in work (on the CPU memory).
    
    This version uses normal equations and SVD in an iterative process that
    makes the computation numerically accurate.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the M-by-N matrix Q with orthogonal columns.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param
    dwork   (GPU workspace) COMPLEX array, dimension (N,N)

    @param[out]
    work    (CPU workspace) COMPLEX array, dimension 3n^2.
            On exit, work(1:n^2) holds the rectangular matrix R.
            Preferably, for higher performance, work should be in pinned memory.
 
    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ---------------

    @ingroup magma_cgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_cgegqr_gpu( magma_int_t ikind, magma_int_t m, magma_int_t n,
                  magmaFloatComplex *dA,   magma_int_t ldda,
                  magmaFloatComplex *dwork, magmaFloatComplex *work,
                  magma_int_t *info )
{
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

    if (ikind == 1) {
        // === Iterative, based on SVD ============================================================
        magmaFloatComplex *U, *VT, *vt, *R, *G, *hwork, *tau;
        float *S;

        R    = work;             // Size n * n
        G    = R    + n*n;       // Size n * n
        VT   = G    + n*n;       // Size n * n
        
        magma_cmalloc_cpu( &hwork, 32 + 2*n*n + 2*n);
        if ( hwork == NULL ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        
        magma_int_t lwork=n*n+32; // First part f hwork; used as workspace in svd
        
        U    = hwork + n*n + 32;  // Size n*n
        S    = (float *)(U+n*n); // Size n
        tau  = U + n*n + n;       // Size n
        
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
            magma_cgetmatrix(n, n, dwork, n, G, n);
            
#if defined(PRECISION_s) || defined(PRECISION_d)
            lapackf77_cgesvd("n", "a", &n, &n, G, &n, S, U, &n, VT, &n,
                             hwork, &lwork, info);
#else
            lapackf77_cgesvd("n", "a", &n, &n, G, &n, S, U, &n, VT, &n,
                             hwork, &lwork, rwork, info);
#endif
            
            mins = 100.f, maxs = 0.f;
            for (k=0; k < n; k++) {
                S[k] = magma_ssqrt( S[k] );
                
                if (S[k] < mins)  mins = S[k];
                if (S[k] > maxs)  maxs = S[k];
            }
            
            for (k=0; k < n; k++) {
                vt = VT + k*n;
                for (j=0; j < n; j++)
                    vt[j] *= S[j];
            }
            lapackf77_cgeqrf(&n, &n, VT, &n, tau, hwork, &lwork, info);
            
            if (i == 1)
                blasf77_ccopy(&n2, VT, &ione, R, &ione);
            else
                blasf77_ctrmm("l", "u", "n", "n", &n, &n, &one, VT, &n, R, &n);
            
            magma_csetmatrix(n, n, VT, n, dwork, n);
            magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, m, n, one, dwork, n, dA, ldda);
            if (mins > 0.00001f)
                cn = maxs/mins;
            
            //fprintf(stderr, "Iteration %d, cond num = %f \n", i, cn);
        } while (cn > 10.f);
        
        magma_free_cpu( hwork );
#if defined(PRECISION_c) || defined(PRECISION_z)
        magma_free_cpu( rwork );
#endif
        // ================== end of ikind == 1 ===================================================
    }
    else if (ikind == 2) {
        // ================== LAPACK based      ===================================================
        magma_int_t min_mn = min(m, n);
        int             nb = n;

        magmaFloatComplex *dtau = dwork + 2*n*n, *d_T = dwork, *ddA = dwork + n*n;
        magmaFloatComplex *tau  = work;

        magma_cgeqr2x3_gpu(&m, &n, dA, &ldda, dtau, d_T, ddA,
                           (float *)(dwork+min_mn+2*n*n), info);
        magma_cgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn);
        magma_cungqr_gpu( m, n, n, dA, ldda, tau, d_T, nb, info );
        // ================== end of ikind == 2 ===================================================       
    }
    else if (ikind == 3) {
        // ================== MGS               ===================================================
        #define work(i,j) (work + (i) + (j)*n)
        #define dA(  i,j) (dA   + (i) + (j)*ldda)
        for(magma_int_t j = 0; j<n; j++){
            for(magma_int_t i = 0; i<j; i++){
                *work(i, j) = magma_cdotc(m, dA(0,i), 1, dA(0,j), 1);
                magma_caxpy(m, -(*work(i,j)),  dA(0,i), 1, dA(0,j), 1);
            }
            for(magma_int_t i = j; i<n; i++)
                *work(i, j) = MAGMA_C_ZERO;
            //*work(j,j) = MAGMA_C_MAKE( magma_scnrm2(m, dA(0,j), 1), 0. );
            *work(j,j) = magma_cdotc(m, dA(0,j), 1, dA(0,j), 1);
            *work(j,j) = MAGMA_C_MAKE( sqrt(MAGMA_C_REAL( *work(j,j) )), 0.);
            magma_cscal(m, 1./ *work(j,j), dA(0,j), 1);
        }
        // ================== end of ikind == 3 ===================================================
    }
    else if (ikind == 4) {
        // ================== Cholesky QR       ===================================================
        magma_cgemm(MagmaConjTrans, MagmaNoTrans, n, n, m, one, dA, ldda, dA, ldda, zero, dwork, n );
        magma_cgetmatrix(n, n, dwork, n, work, n);
        lapackf77_cpotrf("u", &n, work, &n, info);
        magma_csetmatrix(n, n, work, n, dwork, n);
        magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, m, n, one, dwork, n, dA, ldda);
        // ================== end of ikind == 4 ===================================================
    }
             
    return *info;
} /* magma_cgegqr_gpu */
