/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hatem Ltaief
       @author Mark Gates
       
       @generated from src/ztrtri.cpp normal z -> s, Mon May  2 23:30:01 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    STRTRI computes the inverse of a real upper or lower triangular
    matrix A.

    This is the Level 3 BLAS version of the algorithm.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  A is upper triangular;
      -     = MagmaLower:  A is lower triangular.

    @param[in]
    diag    magma_diag_t
      -     = MagmaNonUnit:  A is non-unit triangular;
      -     = MagmaUnit:     A is unit triangular.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    A       REAL array, dimension (LDA,N)
            On entry, the triangular matrix A.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of the array A contains
            the upper triangular matrix, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of the array A contains
            the lower triangular matrix, and the strictly upper
            triangular part of A is not referenced.  If DIAG = MagmaUnit, the
            diagonal elements of A are also not referenced and are
            assumed to be 1.
            On exit, the (triangular) inverse of the original matrix, in
            the same storage format.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0: successful exit
      -     < 0: if INFO = -i, the i-th argument had an illegal value
      -     > 0: if INFO = i, A(i,i) is exactly zero.  The triangular
                    matrix is singular and its inverse cannot be computed.

    @ingroup magma_sgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_strtri(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    float *A, magma_int_t lda,
    magma_int_t *info)
{
    #define  A(i_, j_) ( A + (i_) + (j_)*lda )
    
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif

    // Constants
    const float c_zero     = MAGMA_S_ZERO;
    const float c_one      = MAGMA_S_ONE;
    const float c_neg_one  = MAGMA_S_NEG_ONE;
    const char* uplo_ = lapack_uplo_const( uplo );
    const char* diag_ = lapack_diag_const( diag );
    
    // Local variables
    magma_int_t ldda, nb, nn, j, jb;
    magmaFloat_ptr dA;

    bool upper  = (uplo == MagmaUpper);
    bool nounit = (diag == MagmaNonUnit);

    *info = 0;

    if (! upper && uplo != MagmaLower)
        *info = -1;
    else if (! nounit && diag != MagmaUnit)
        *info = -2;
    else if (n < 0)
        *info = -3;
    else if (lda < max(1,n))
        *info = -5;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Quick return
    if ( n == 0 )
        return *info;

    // Check for singularity if non-unit
    if (nounit) {
        for (j=0; j < n; ++j) {
            if ( MAGMA_S_EQUAL( *A(j,j), c_zero )) {
                *info = j+1;  // Fortran index
                return *info;
            }
        }
    }

    // Determine the block size for this environment
    nb = magma_get_spotrf_nb( n );

    ldda = magma_roundup( n, 32 );
    if (MAGMA_SUCCESS != magma_smalloc( &dA, (n)*ldda )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    //magma_queue_create( cdev, &queues[1] );  // unused

    if (nb <= 1 || nb >= n) {
        lapackf77_strtri( uplo_, diag_, &n, A, &lda, info );
    }
    else if (upper) {
        // Compute inverse of upper triangular matrix
        for (j=0; j < n; j += nb) {
            jb = min( nb, n-j );
            
            if (j > 0) {
                // Send current block column (with diagonal) to device
                // This must finish before trtri below
                magma_ssetmatrix( j+jb, jb,
                                  A(0,j),  lda,
                                  dA(0,j), ldda, queues[0] );
                
                // Compute rows 0:j of current block column
                magma_strmm( MagmaLeft, MagmaUpper,
                             MagmaNoTrans, diag, j, jb, c_one,
                             dA(0,0), ldda,
                             dA(0,j), ldda, queues[0] );
    
                magma_strsm( MagmaRight, MagmaUpper,
                             MagmaNoTrans, diag, j, jb, c_neg_one,
                             dA(j,j), ldda,
                             dA(0,j), ldda, queues[0] );
                
                // Get above diagonal from device
                // TODO: could be on another queue, after trmm/trsm finish
                magma_sgetmatrix_async( j, jb,
                                        dA(0,j), ldda,
                                        A(0,j),  lda, queues[0] );
            }

            // Compute inverse of current diagonal block
            // TODO: problem if diagonal has not finished sending yet?
            lapackf77_strtri( MagmaUpperStr, diag_, &jb, A(j,j), &lda, info );

            if (j+jb < n) {
                // Send inverted diagonal block to device
                magma_ssetmatrix( jb, jb,
                                  A(j,j),  lda,
                                  dA(j,j), ldda, queues[0] );
            }
        }
    }
    else {
        // Compute inverse of lower triangular matrix
        nn = ((n-1)/nb)*nb;

        for (j=nn; j >= 0; j -= nb) {
            jb = min( nb, n-j );

            if (j+jb < n) {
                // Send current block row (with diagonal) to device
                // This must finish before trtri below
                magma_ssetmatrix( n-j, jb,
                                  A(j,j),  lda,
                                  dA(j,j), ldda, queues[0] );
                
                // Compute rows j+jb:n of current block column
                magma_strmm( MagmaLeft, MagmaLower,
                             MagmaNoTrans, diag, n-j-jb, jb, c_one,
                             dA(j+jb,j+jb), ldda,
                             dA(j+jb,j),    ldda, queues[0] );

                magma_strsm( MagmaRight, MagmaLower,
                             MagmaNoTrans, diag, n-j-jb, jb, c_neg_one,
                             dA(j,j),    ldda,
                             dA(j+jb,j), ldda, queues[0] );

                // Get below diagonal block from device
                magma_sgetmatrix_async( n-j-jb, jb,
                                        dA(j+jb,j), ldda,
                                        A(j+jb,j),  lda, queues[0] );
            }
            
            // Compute inverse of current diagonal block
            lapackf77_strtri( MagmaLowerStr, diag_, &jb, A(j,j), &lda, info );
            
            if (j > 0) {
                // Send inverted diagonal block to device
                magma_ssetmatrix( jb, jb,
                                  A(j,j),  lda,
                                  dA(j,j), ldda, queues[0] );
            }
        }
    }

    magma_queue_destroy( queues[0] );
    //magma_queue_destroy( queues[1] );  // unused
    magma_free( dA );

    return *info;
}
