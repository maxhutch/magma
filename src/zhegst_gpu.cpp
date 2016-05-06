/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @precisions normal z -> s d c
*/

#include "magma_internal.h"

/**
    Purpose
    -------
    ZHEGST_GPU reduces a complex Hermitian-definite generalized
    eigenproblem to standard form.
    
    If ITYPE = 1, the problem is A*x = lambda*B*x,
    and A is overwritten by inv(U^H)*A*inv(U) or inv(L)*A*inv(L^H)
    
    If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
    B*A*x = lambda*x, and A is overwritten by U*A*U^H or L^H*A*L.
    
    B must have been previously factorized as U^H*U or L*L^H by ZPOTRF.
    
    Arguments
    ---------
    @param[in]
    itype   INTEGER
            = 1: compute inv(U^H)*A*inv(U) or inv(L)*A*inv(L^H);
            = 2 or 3: compute U*A*U^H or L^H*A*L.
    
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored and B is factored as U^H*U;
      -     = MagmaLower:  Lower triangle of A is stored and B is factored as L*L^H.
    
    @param[in]
    n       INTEGER
            The order of the matrices A and B.  N >= 0.
    
    @param[in,out]
    dA      COMPLEX_16 array, on the GPU device, dimension (LDDA,N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the transformed matrix, stored in the
            same format as A.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).
    
    @param[in]
    dB      COMPLEX_16 array, on the GPU device, dimension (LDDB,N)
            The triangular factor from the Cholesky factorization of B,
            as returned by ZPOTRF.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDDB >= max(1,N).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_zheev_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zhegst_gpu(
    magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magma_int_t *info)
{
    #define A(i_, j_) (work + (i_) + (j_)*lda         )
    #define B(i_, j_) (work + (i_) + (j_)*ldb + nb*ldb)
    
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    
    /* Constants */
    const magmaDoubleComplex c_one      = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one  = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_half     = MAGMA_Z_HALF;
    const magmaDoubleComplex c_neg_half = MAGMA_Z_NEG_HALF;
    const double             d_one      = 1.0;
    
    /* Local variables */
    const char* uplo_ = lapack_uplo_const( uplo );
    magma_int_t k, kb, kb2, nb;
    magma_int_t lda;
    magma_int_t ldb;
    magmaDoubleComplex *work;
    bool upper = (uplo == MagmaUpper);
    
    /* Test the input parameters. */
    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (! upper && uplo != MagmaLower) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldda < max(1,n)) {
        *info = -5;
    } else if (lddb < max(1,n)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    /* Quick return */
    if ( n == 0 )
        return *info;
    
    nb = magma_get_zhegst_nb( n );
    
    lda = nb;
    ldb = nb;
    
    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, 2*nb*nb )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    
    /* Use hybrid blocked code */
    if (itype == 1) {
        if (upper) {
            kb = min( n, nb );
            
            /* Compute inv(U^H)*A*inv(U) */
            magma_zgetmatrix_async( kb, kb,
                                    dA(0, 0), ldda,
                                     A(0, 0), lda, queues[0] );
            
            magma_zgetmatrix_async( kb, kb,
                                    dB(0, 0), lddb,
                                     B(0, 0), ldb, queues[0] );
            
            for (k = 0; k < n; k += nb) {
                kb  = min( n-k,    nb );
                kb2 = min( n-k-nb, nb );
                
                magma_queue_sync( queues[0] );  // finish get dA(k,k) -> A(0,0) and dB(k,k) -> B(0,0)
                
                /* Update the upper triangle of A(k:n,k:n) */
                lapackf77_zhegst( &itype, uplo_, &kb, A(0,0), &lda, B(0,0), &ldb, info );
                
                magma_zsetmatrix_async( kb, kb,
                                         A(0, 0), lda,
                                        dA(k, k), ldda, queues[0] );
                
                if (k+kb < n) {
                    magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                 kb, n-k-kb,
                                 c_one, dB(k,k),    lddb,
                                        dA(k,k+kb), ldda, queues[1] );
                    
                    magma_queue_sync( queues[0] );  // finish set dA(k,k)
                    
                    // Start copying next B block
                    magma_zgetmatrix_async( kb2, kb2,
                                            dB(k+kb, k+kb), lddb,
                                             B(0, 0),       ldb, queues[0] );
                    
                    magma_zhemm( MagmaLeft, MagmaUpper,
                                 kb, n-k-kb,
                                 c_neg_half, dA(k,k),    ldda,
                                             dB(k,k+kb), lddb,
                                 c_one,      dA(k,k+kb), ldda, queues[1] );
                    
                    magma_zher2k( MagmaUpper, MagmaConjTrans,
                                  n-k-kb, kb,
                                  c_neg_one, dA(k,k+kb),    ldda,
                                             dB(k,k+kb),    lddb,
                                  d_one,     dA(k+kb,k+kb), ldda, queues[1] );
                    
                    // Start copying next A block
                    magma_queue_sync( queues[1] );
                    magma_zgetmatrix_async( kb2, kb2,
                                            dA(k+kb, k+kb), ldda,
                                             A(0, 0),       lda, queues[0] );
                    
                    magma_zhemm( MagmaLeft, MagmaUpper,
                                 kb, n-k-kb,
                                 c_neg_half, dA(k,k),    ldda,
                                             dB(k,k+kb), lddb,
                                 c_one,      dA(k,k+kb), ldda, queues[1] );
                    
                    magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                                 kb, n-k-kb,
                                 c_one, dB(k+kb,k+kb), lddb,
                                        dA(k,k+kb),    ldda, queues[1] );
                }
            }
        }
        else {
            kb = min( n, nb );
            
            /* Compute inv(L)*A*inv(L^H) */
            magma_zgetmatrix_async( kb, kb,
                                    dA(0, 0), ldda,
                                     A(0, 0), lda, queues[0] );
            
            magma_zgetmatrix_async( kb, kb,
                                    dB(0, 0), lddb,
                                     B(0, 0), ldb, queues[0] );
            
            for (k = 0; k < n; k += nb) {
                kb  = min( n-k,    nb );
                kb2 = min( n-k-nb, nb );
                
                magma_queue_sync( queues[0] );  // finish get dA(k,k) -> A(0,0) and dB(k,k) -> B(0,0)
                
                /* Update the lower triangle of A(k:n,k:n) */
                lapackf77_zhegst( &itype, uplo_, &kb, A(0, 0), &lda, B(0, 0), &ldb, info );
                
                magma_zsetmatrix_async( kb, kb,
                                         A(0, 0), lda,
                                        dA(k, k), ldda, queues[0] );
                
                if (k+kb < n) {
                    magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                 n-k-kb, kb,
                                 c_one, dB(k,k),    lddb,
                                        dA(k+kb,k), ldda, queues[1] );
                    
                    magma_queue_sync( queues[0] );  // finish set dA(k,k)
                    
                    // Start copying next B block
                    magma_zgetmatrix_async( kb2, kb2,
                                            dB(k+kb, k+kb), lddb,
                                             B(0, 0),       ldb, queues[0] );
                    
                    magma_zhemm( MagmaRight, MagmaLower,
                                 n-k-kb, kb,
                                 c_neg_half, dA(k,k),     ldda,
                                             dB(k+kb,k),  lddb,
                                 c_one,      dA(k+kb, k), ldda, queues[1] );
                    
                    magma_zher2k( MagmaLower, MagmaNoTrans,
                                  n-k-kb, kb,
                                  c_neg_one, dA(k+kb,k),    ldda,
                                             dB(k+kb,k),    lddb,
                                  d_one,     dA(k+kb,k+kb), ldda, queues[1] );
                    
                    // Start copying next A block
                    magma_queue_sync( queues[1] );
                    magma_zgetmatrix_async( kb2, kb2,
                                            dA(k+kb, k+kb), ldda,
                                             A(0, 0),       lda, queues[0] );
                    
                    magma_zhemm( MagmaRight, MagmaLower,
                                 n-k-kb, kb,
                                 c_neg_half, dA(k,k),    ldda,
                                             dB(k+kb,k), lddb,
                                 c_one,      dA(k+kb,k), ldda, queues[1] );
                    
                    magma_ztrsm( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaNonUnit,
                                 n-k-kb, kb,
                                 c_one, dB(k+kb,k+kb), lddb,
                                        dA(k+kb,k),    ldda, queues[1] );
                }
            }
        }
    }
    else {  // itype == 2 or 3
        if (upper) {
            /* Compute U*A*U^H */
            for (k = 0; k < n; k += nb) {
                kb = min( n-k, nb );
                
                magma_zgetmatrix_async( kb, kb,
                                        dA(k, k), ldda,
                                         A(0, 0), lda, queues[0] );
                
                magma_zgetmatrix_async( kb, kb,
                                        dB(k, k), lddb,
                                         B(0, 0), ldb, queues[0] );
                
                /* Update the upper triangle of A(1:k+kb-1,1:k+kb-1) */
                if (k > 0) {
                    magma_ztrmm( MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                                 k, kb,
                                 c_one, dB(0,0), lddb,
                                        dA(0,k), ldda, queues[1] );
                    
                    magma_zhemm( MagmaRight, MagmaUpper,
                                 k, kb,
                                 c_half, dA(k,k), ldda,
                                         dB(0,k), lddb,
                                 c_one,  dA(0,k), ldda, queues[1] );
                    
                    magma_zher2k( MagmaUpper, MagmaNoTrans,
                                  k, kb,
                                  c_one, dA(0,k), ldda,
                                         dB(0,k), lddb,
                                  d_one, dA(0,0), ldda, queues[1] );
                    
                    magma_zhemm( MagmaRight, MagmaUpper,
                                 k, kb,
                                 c_half, dA(k,k), ldda,
                                         dB(0,k), lddb,
                                 c_one,  dA(0,k), ldda, queues[1] );
                    
                    magma_ztrmm( MagmaRight, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                 k, kb,
                                 c_one, dB(k,k), lddb,
                                        dA(0,k), ldda, queues[1] );
                }
                
                magma_queue_sync( queues[0] );  // finish get dA(k,k) -> A(0,0) and dB(k,k) -> B(0,0)
                
                lapackf77_zhegst( &itype, uplo_, &kb, A(0, 0), &lda, B(0, 0), &ldb, info );
                
                magma_zsetmatrix_async( kb, kb,
                                         A(0, 0), lda,
                                        dA(k, k), ldda, queues[1] );
                magma_queue_sync( queues[1] );  // wait for A(0,0) before getting next panel
            }
        }
        else {
            /* Compute L^H*A*L */
            for (k = 0; k < n; k += nb) {
                kb = min( n-k, nb );
                
                magma_zgetmatrix_async( kb, kb,
                                        dA(k, k), ldda,
                                         A(0, 0), lda, queues[0] );
                
                magma_zgetmatrix_async( kb, kb,
                                        dB(k, k), lddb,
                                         B(0, 0), ldb, queues[0] );
                
                /* Update the lower triangle of A(1:k+kb-1,1:k+kb-1) */
                if (k > 0) {
                    magma_ztrmm( MagmaRight, MagmaLower, MagmaNoTrans, MagmaNonUnit,
                                 kb, k,
                                 c_one, dB(0,0), lddb,
                                        dA(k,0), ldda, queues[1] );
                    
                    magma_zhemm( MagmaLeft, MagmaLower,
                                 kb, k,
                                 c_half, dA(k,k), ldda,
                                         dB(k,0), lddb,
                                 c_one,  dA(k,0), ldda, queues[1] );
                    
                    magma_queue_sync( queues[1] );
                    
                    magma_zher2k( MagmaLower, MagmaConjTrans,
                                  k, kb,
                                  c_one, dA(k,0), ldda,
                                         dB(k,0), lddb,
                                  d_one, dA(0,0), ldda, queues[1] );
                    
                    magma_zhemm( MagmaLeft, MagmaLower,
                                 kb, k,
                                 c_half, dA(k,k), ldda,
                                         dB(k,0), lddb,
                                 c_one,  dA(k,0), ldda, queues[1] );
                    
                    magma_ztrmm( MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                 kb, k,
                                 c_one, dB(k,k), lddb,
                                        dA(k,0), ldda, queues[1] );
                }
                
                magma_queue_sync( queues[0] );  // finish get dA(k,k) -> A(0,0) and dB(k,k) -> B(0,0)
                
                lapackf77_zhegst( &itype, uplo_, &kb, A(0, 0), &lda, B(0, 0), &ldb, info );
                
                magma_zsetmatrix_async( kb, kb,
                                         A(0, 0), lda,
                                        dA(k, k), ldda, queues[1] );
                magma_queue_sync( queues[1] );  // wait for A(0,0) before getting next panel
            }
        }
    }
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    magma_free_pinned( work );
    
    return *info;
} /* magma_zhegst_gpu */

#undef A
#undef B
#undef dA
#undef dB
