/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       @author Adrien REMY

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    ZHETRS solves a system of linear equations A*X = B with a complex
    Hermitian matrix A using the factorization A = U * D * U**H or
    A = L * D * L**H computed by ZHETRF_NOPIV_GPU.
    
    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            The block diagonal matrix D and the multipliers used to
            obtain the factor U or L as computed by ZHETRF_NOPIV_GPU.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in,out]
    dB      COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDDB >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_zhesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zhetrs_nopiv_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info)
{
    /* Constants */
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;

    /* Local variables */
    bool upper = (uplo == MagmaUpper);
    
    /* Check input arguments */
    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
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

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    if (upper) {
        magma_ztrsm( MagmaLeft, MagmaUpper,
                     MagmaConjTrans, MagmaUnit,
                     n, nrhs, c_one,
                     dA, ldda, dB, lddb, queue );
        magmablas_zlascl_diag( MagmaUpper, n, nrhs, dA, ldda, dB, lddb, queue, info );
        //for (i = 0; i < nrhs; i++)
        //    magmablas_zlascl_diag( MagmaUpper, 1, n, dA, ldda, dB+(lddb*i), 1, info );
        magma_ztrsm( MagmaLeft, MagmaUpper,
                     MagmaNoTrans, MagmaUnit,
                     n, nrhs, c_one,
                     dA, ldda, dB, lddb, queue );
    } else {
        magma_ztrsm( MagmaLeft, MagmaLower,
                     MagmaNoTrans, MagmaUnit,
                     n, nrhs, c_one,
                     dA, ldda, dB, lddb, queue );
        magmablas_zlascl_diag( MagmaUpper, n, nrhs, dA, ldda, dB, lddb, queue, info );
        //for (i = 0; i < nrhs; i++)
        //    magmablas_zlascl_diag( MagmaLower, 1, n, dA, ldda, dB+(lddb*i), 1, info );
        magma_ztrsm( MagmaLeft, MagmaLower,
                     MagmaConjTrans, MagmaUnit,
                     n, nrhs, c_one,
                     dA, ldda, dB, lddb, queue );
    }
    
    magma_queue_destroy( queue );
    
    return *info;
}
