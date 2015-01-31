/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Stan Tomov
       @generated from zgegqr_gpu.cpp normal z -> s, Fri Jan 30 19:00:15 2015

*/
#include "common_magma.h"

#define PRECISION_s

// === Define what BLAS to use ============================================
//#if defined(PRECISION_s) || defined(PRECISION_d)
    #define magma_strsm magmablas_strsm
//#endif
// === End defining what BLAS to use ======================================

/**
    Purpose
    -------
    SGEGQR orthogonalizes the N vectors given by a real M-by-N matrix A:
           
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
                MAGMA's QR panel factorization (magma_sgeqr2x3_gpu) and magma_sorgqr
            3:  MGS
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
    dA      REAL array on the GPU, dimension (ldda,n)
            On entry, the m-by-n matrix A.
            On exit, the m-by-n matrix Q with orthogonal columns.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,m).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param
    dwork   (GPU workspace) REAL array, dimension:
            n^2                    for ikind = 1
            3 n^2 + min(m, n) + 2  for ikind = 2
            0 (not used)           for ikind = 3
            n^2                    for ikind = 4

    @param[out]
    work    (CPU workspace) REAL array, dimension 3 n^2.
            On exit, work(1:n^2) holds the rectangular matrix R.
            Preferably, for higher performance, work should be in pinned memory.
 
    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.


    @ingroup magma_sgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_sgegqr_gpu(
    magma_int_t ikind, magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA,   magma_int_t ldda,
    magmaFloat_ptr dwork, float *work,
    magma_int_t *info )
{
    #define work(i_,j_) (work + (i_) + (j_)*n)
    #define dA(i_,j_)   (dA   + (i_) + (j_)*ldda)
    
    magma_int_t i = 0, j, k, n2 = n*n;
    magma_int_t ione = 1;
    float c_zero = MAGMA_S_ZERO;
    float c_one  = MAGMA_S_ONE;
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

    if (ikind == 1) {
        // === Iterative, based on SVD ============================================================
        float *U, *VT, *vt, *R, *G, *hwork, *tau;
        float *S;

        R    = work;             // Size n * n
        G    = R    + n*n;       // Size n * n
        VT   = G    + n*n;       // Size n * n
        
        magma_smalloc_cpu( &hwork, 32 + 2*n*n + 2*n);
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
            
            magma_sgemm(MagmaConjTrans, MagmaNoTrans, n, n, m, c_one, dA, ldda, dA, ldda, c_zero, dwork, n );
            magma_sgetmatrix(n, n, dwork, n, G, n);
            
#if defined(PRECISION_s) || defined(PRECISION_d)
            lapackf77_sgesvd("n", "a", &n, &n, G, &n, S, U, &n, VT, &n,
                             hwork, &lwork, info);
#else
            lapackf77_sgesvd("n", "a", &n, &n, G, &n, S, U, &n, VT, &n,
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
            lapackf77_sgeqrf(&n, &n, VT, &n, tau, hwork, &lwork, info);
            
            if (i == 1)
                blasf77_scopy(&n2, VT, &ione, R, &ione);
            else
                blasf77_strmm("l", "u", "n", "n", &n, &n, &c_one, VT, &n, R, &n);
            
            magma_ssetmatrix(n, n, VT, n, dwork, n);
            magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, m, n, c_one, dwork, n, dA, ldda);
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
        magma_int_t nb = n;

        magmaFloat_ptr dtau = dwork + 2*n*n;
        magmaFloat_ptr d_T  = dwork;
        magmaFloat_ptr ddA  = dwork + n*n;
        float *tau  = work+n*n;

        magmablas_slaset( MagmaFull, n, n, c_zero, c_zero, d_T, n );
        magma_sgeqr2x3_gpu(m, n, dA, ldda, dtau, d_T, ddA,
                           (float *)(dwork+min_mn+2*n*n), info);
        magma_sgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn);
        magma_sgetmatrix( n, n, ddA, n, work, n);
        magma_sorgqr_gpu( m, n, n, dA, ldda, tau, d_T, nb, info );
        // ================== end of ikind == 2 ===================================================
    }
    else if (ikind == 3) {
        // ================== MGS               ===================================================
        for (magma_int_t j = 0; j < n; j++) {
            for (magma_int_t i = 0; i < j; i++) {
                *work(i, j) = magma_sdot(m, dA(0,i), 1, dA(0,j), 1);
                magma_saxpy(m, -(*work(i,j)),  dA(0,i), 1, dA(0,j), 1);
            }
            for (magma_int_t i = j; i < n; i++)
                *work(i, j) = MAGMA_S_ZERO;
            //*work(j,j) = MAGMA_S_MAKE( magma_snrm2(m, dA(0,j), 1), 0. );
            *work(j,j) = magma_sdot(m, dA(0,j), 1, dA(0,j), 1);
            *work(j,j) = MAGMA_S_MAKE( sqrt(MAGMA_S_REAL( *work(j,j) )), 0.);
            magma_sscal(m, 1./ *work(j,j), dA(0,j), 1);
        }
        // ================== end of ikind == 3 ===================================================
    }
    else if (ikind == 4) {
        // ================== Cholesky QR       ===================================================
        magma_sgemm(MagmaConjTrans, MagmaNoTrans, n, n, m, c_one, dA, ldda, dA, ldda, c_zero, dwork, n );
        magma_sgetmatrix(n, n, dwork, n, work, n);
        lapackf77_spotrf("u", &n, work, &n, info);
        magma_ssetmatrix(n, n, work, n, dwork, n);
        magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, m, n, c_one, dwork, n, dA, ldda);
        // ================== end of ikind == 4 ===================================================
    }
             
    return *info;
} /* magma_sgegqr_gpu */
