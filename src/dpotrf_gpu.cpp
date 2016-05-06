/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Stan Tomov
       @author Mark Gates
       
       @generated from src/zpotrf_gpu.cpp normal z -> d, Mon May  2 23:29:59 2016
*/
#include "magma_internal.h"

// === Define what BLAS to use ============================================
    #undef  magma_dtrsm
    #define magma_dtrsm magmablas_dtrsm
// === End defining what BLAS to use =======================================

/**
    Purpose
    -------
    DPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the symmetric matrix dA.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_dposv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dpotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda + dA_offset)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif

    /* Constants */
    const double c_one     = MAGMA_D_ONE;
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const double d_one     =  1.0;
    const double d_neg_one = -1.0;
    
    /* Local variables */
    const char* uplo_ = lapack_uplo_const( uplo );
    bool upper = (uplo == MagmaUpper);
    
    magma_int_t j, jb, nb;
    double *work;

    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    nb = magma_get_dpotrf_nb( n );
    
    if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    
    if (nb <= 1 || nb >= n) {
        /* Use unblocked code. */
        magma_dgetmatrix( n, n, dA(0,0), ldda, work, n, queues[0] );
        lapackf77_dpotrf( uplo_, &n, work, &n, info );
        magma_dsetmatrix( n, n, work, n, dA(0,0), ldda, queues[0] );
    }
    else {
        /* Use blocked code. */
        if (upper) {
            //=========================================================
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j < n; j += nb) {
                // apply all previous updates to diagonal block,
                // then transfer it to CPU
                jb = min( nb, n-j );
                magma_dsyrk( MagmaUpper, MagmaConjTrans, jb, j,
                             d_neg_one, dA(0, j), ldda,
                             d_one,     dA(j, j), ldda, queues[1] );
                
                magma_queue_sync( queues[1] );
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
                
                // apply all previous updates to block row right of diagonal block
                if (j+jb < n) {
                    magma_dgemm( MagmaConjTrans, MagmaNoTrans,
                                 jb, n-j-jb, j,
                                 c_neg_one, dA(0, j   ), ldda,
                                            dA(0, j+jb), ldda,
                                 c_one,     dA(j, j+jb), ldda, queues[1] );
                }
                
                // simultaneous with above dgemm, transfer diagonal block,
                // factor it on CPU, and test for positive definiteness
                magma_queue_sync( queues[0] );
                lapackf77_dpotrf( MagmaUpperStr, &jb, work, &jb, info );
                magma_dsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[1] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                
                // apply diagonal block to block row right of diagonal block
                if (j+jb < n) {
                    magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                 jb, n-j-jb,
                                 c_one, dA(j, j),    ldda,
                                        dA(j, j+jb), ldda, queues[1] );
                }
            }
        }
        else {
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j < n; j += nb) {
                // apply all previous updates to diagonal block,
                // then transfer it to CPU
                jb = min( nb, n-j );
                magma_dsyrk( MagmaLower, MagmaNoTrans, jb, j,
                             d_neg_one, dA(j, 0), ldda,
                             d_one,     dA(j, j), ldda, queues[1] );
                
                magma_queue_sync( queues[1] );
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
                
                // apply all previous updates to block column below diagonal block
                if (j+jb < n) {
                    magma_dgemm( MagmaNoTrans, MagmaConjTrans,
                                 n-j-jb, jb, j,
                                 c_neg_one, dA(j+jb, 0), ldda,
                                            dA(j,    0), ldda,
                                 c_one,     dA(j+jb, j), ldda, queues[1] );
                }
                
                // simultaneous with above dgemm, transfer diagonal block,
                // factor it on CPU, and test for positive definiteness
                magma_queue_sync( queues[0] );
                lapackf77_dpotrf( MagmaLowerStr, &jb, work, &jb, info );
                magma_dsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[1] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                
                // apply diagonal block to block column below diagonal
                if (j+jb < n) {
                    magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                 n-j-jb, jb,
                                 c_one, dA(j,    j), ldda,
                                        dA(j+jb, j), ldda, queues[1] );
                }
            }
        }
    }
    
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    magma_free_pinned( work );
    
    return *info;
} /* magma_dpotrf_gpu */
