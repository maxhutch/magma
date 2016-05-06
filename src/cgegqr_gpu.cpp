/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Stan Tomov
       @generated from src/zgegqr_gpu.cpp normal z -> c, Mon May  2 23:30:06 2016

*/
#include "magma_internal.h"

#define COMPLEX

// === Define what BLAS to use ============================================
#undef  magma_ctrsm
#define magma_ctrsm magmablas_ctrsm
// === End defining what BLAS to use ======================================

/**
    Purpose
    -------
    CGEGQR orthogonalizes the N vectors given by a complex M-by-N matrix A:
           
        A = Q * R.

    On exit, if successful, the orthogonal vectors Q overwrite A
    and R is given in work (on the CPU memory).
    The routine is designed for tall-and-skinny matrices: M >> N, N <= 128.
    
    This version uses normal equations and SVD in an iterative process that
    makes the computation numerically accurate.
    
    Arguments
    ---------
    @param[in]
    ikind   INTEGER
            Several versions are implemented indiceted by the ikind value:
            1:  This version uses normal equations and SVD in an iterative process
                that makes the computation numerically accurate.
            2:  This version uses a standard LAPACK-based orthogonalization through
                MAGMA's QR panel factorization (magma_cgeqr2x3_gpu) and magma_cungqr
            3:  Modified Gram-Schmidt (MGS)
            4.  Cholesky QR [ Note: this method uses the normal equations which
                                    squares the condition number of A, therefore
                                    ||I - Q'Q|| < O(eps cond(A)^2)               ]

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  m >= n >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. 128 >= n >= 0.

    @param[in,out]
    dA      COMPLEX array on the GPU, dimension (ldda,n)
            On entry, the m-by-n matrix A.
            On exit, the m-by-n matrix Q with orthogonal columns.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,m).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param
    dwork   (GPU workspace) COMPLEX array, dimension:
            n^2                    for ikind = 1
            3 n^2 + min(m, n) + 2  for ikind = 2
            0 (not used)           for ikind = 3
            n^2                    for ikind = 4

    @param[out]
    work    (CPU workspace) COMPLEX array, dimension 3 n^2.
            On exit, work(1:n^2) holds the rectangular matrix R.
            Preferably, for higher performance, work should be in pinned memory.
 
    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  for ikind = 4, the normal equations were not
                  positive definite, so the factorization could not be
                  completed, and the solution has not been computed.

    @ingroup magma_cgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_cgegqr_gpu(
    magma_int_t ikind, magma_int_t m, magma_int_t n,
    magmaFloatComplex_ptr dA,   magma_int_t ldda,
    magmaFloatComplex_ptr dwork, magmaFloatComplex *work,
    magma_int_t *info )
{
    #define work(i_,j_) (work + (i_) + (j_)*n)
    #define dA(i_,j_)   (dA   + (i_) + (j_)*ldda)
    
    magma_int_t i = 0, j, k, n2 = n*n;
    magma_int_t ione = 1;
    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    magmaFloatComplex c_one  = MAGMA_C_ONE;
    float cn = 200., mins, maxs;

    /* check arguments */
    *info = 0;
    if (ikind < 1 || ikind > 4) {
        *info = -1;
    } else if (m < 0 || m < n) {
        *info = -2;
    } else if (n < 0 || n > 128) {
        *info = -3;
    } else if (ldda < max(1,m)) {
        *info = -5;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    if (ikind == 1) {
        // === Iterative, based on SVD ============================================================
        magmaFloatComplex *U, *VT, *vt, *R, *G, *hwork, *tau;
        float *S;

        R    = work;             // Size n * n
        G    = R    + n*n;       // Size n * n
        VT   = G    + n*n;       // Size n * n
        
        magma_cmalloc_cpu( &hwork, 32 + 2*n*n + 2*n );
        if ( hwork == NULL ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        
        magma_int_t lwork=n*n+32; // First part f hwork; used as workspace in svd
        
        U    = hwork + n*n + 32;   // Size n*n
        S    = (float*)(U + n*n); // Size n
        tau  = U + n*n + n;        // Size n
        
        #ifdef COMPLEX
        float *rwork;
        magma_smalloc_cpu( &rwork, 5*n );
        if ( rwork == NULL ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        #endif
        
        do {
            i++;
            
            magma_cgemm( MagmaConjTrans, MagmaNoTrans, n, n, m, c_one,
                         dA, ldda, dA, ldda, c_zero, dwork, n, queue );
            magma_cgetmatrix( n, n, dwork, n, G, n, queue );
            
            lapackf77_cgesvd( "n", "a", &n, &n, G, &n, S, U, &n, VT, &n,
                              hwork, &lwork,
                              #ifdef COMPLEX
                              rwork,
                              #endif
                              info );
            
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
            lapackf77_cgeqrf( &n, &n, VT, &n, tau, hwork, &lwork, info );
            
            if (i == 1)
                blasf77_ccopy( &n2, VT, &ione, R, &ione );
            else
                blasf77_ctrmm( "l", "u", "n", "n", &n, &n, &c_one, VT, &n, R, &n );
            
            magma_csetmatrix( n, n, VT, n, dwork, n, queue );
            magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                         m, n, c_one, dwork, n, dA, ldda, queue );
            if (mins > 0.00001f)
                cn = maxs/mins;
            
            //fprintf( stderr, "Iteration %d, cond num = %f \n", i, cn );
        } while (cn > 10.f);
        
        magma_free_cpu( hwork );
        #ifdef COMPLEX
        magma_free_cpu( rwork );
        #endif
        // ================== end of ikind == 1 ===================================================
    }
    else if (ikind == 2) {
        // ================== LAPACK based      ===================================================
        magma_int_t min_mn = min(m, n);
        magma_int_t nb = n;

        magmaFloatComplex_ptr dtau = dwork + 2*n*n;
        magmaFloatComplex_ptr d_T  = dwork;
        magmaFloatComplex_ptr ddA  = dwork + n*n;
        magmaFloatComplex *tau  = work+n*n;

        magmablas_claset( MagmaFull, n, n, c_zero, c_zero, d_T, n, queue );
        magma_cgeqr2x3_gpu( m, n, dA, ldda, dtau, d_T, ddA,
                            (float*)(dwork+min_mn+2*n*n), info );
        magma_cgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn, queue );
        magma_cgetmatrix( n, n, ddA, n, work, n, queue );
        magma_cungqr_gpu( m, n, n, dA, ldda, tau, d_T, nb, info );
        // ================== end of ikind == 2 ===================================================
    }
    else if (ikind == 3) {
        // ================== MGS               ===================================================
        for (j = 0; j < n; j++) {
            for (i = 0; i < j; i++) {
                *work(i, j) = magma_cdotc( m, dA(0,i), 1, dA(0,j), 1, queue );
                magma_caxpy( m, -(*work(i,j)),  dA(0,i), 1, dA(0,j), 1, queue );
            }
            for (i = j; i < n; i++) {
                *work(i, j) = MAGMA_C_ZERO;
            }
            //*work(j,j) = MAGMA_C_MAKE( magma_scnrm2( m, dA(0,j), 1), 0., queue );
            *work(j,j) = magma_cdotc( m, dA(0,j), 1, dA(0,j), 1, queue );
            *work(j,j) = MAGMA_C_MAKE( sqrt(MAGMA_C_REAL( *work(j,j) )), 0. );
            magma_cscal( m, 1./ *work(j,j), dA(0,j), 1, queue );
        }
        // ================== end of ikind == 3 ===================================================
    }
    else if (ikind == 4) {
        // ================== Cholesky QR       ===================================================
        magma_cgemm( MagmaConjTrans, MagmaNoTrans, n, n, m, c_one,
                     dA, ldda, dA, ldda, c_zero, dwork, n, queue );
        magma_cgetmatrix( n, n, dwork, n, work, n, queue );
        lapackf77_cpotrf( "u", &n, work, &n, info );
        magma_csetmatrix( n, n, work, n, dwork, n, queue );
        magma_ctrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                     m, n, c_one, dwork, n, dA, ldda, queue );
        // ================== end of ikind == 4 ===================================================
    }
             
    magma_queue_destroy( queue );

    return *info;
} /* magma_cgegqr_gpu */
