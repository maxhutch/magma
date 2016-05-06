/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zpotrs_gpu.cpp normal z -> d, Mon May  2 23:30:00 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    DPOTRS solves a system of linear equations A*X = B with a symmetric
    positive definite matrix A using the Cholesky factorization
    A = U**H*U or A = L*L**H computed by DPOTRF.

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
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            The triangular factor U or L from the Cholesky factorization
            A = U**H*U or A = L*L**H, as computed by DPOTRF.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in,out]
    dB      DOUBLE PRECISION array on the GPU, dimension (LDDB,NRHS)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  LDDB >= max(1,N).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_dposv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dpotrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    magma_int_t *info)
{
    // Constants
    const double c_one = MAGMA_D_ONE;

    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower )
        *info = -1;
    if ( n < 0 )
        *info = -2;
    if ( nrhs < 0)
        *info = -3;
    if ( ldda < max(1, n) )
        *info = -5;
    if ( lddb < max(1, n) )
        *info = -7;
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if ( (n == 0) || (nrhs == 0) ) {
        return *info;
    }

    magma_queue_t queue = NULL;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );
    
    if ( uplo == MagmaUpper ) {
        if ( nrhs == 1) {
            magma_dtrsv( MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, dA, ldda, dB, 1, queue );
            magma_dtrsv( MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, dA, ldda, dB, 1, queue );
        } else {
            magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
            magma_dtrsm( MagmaLeft, MagmaUpper, MagmaNoTrans,   MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
        }
    }
    else {
        if ( nrhs == 1) {
            magma_dtrsv( MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, dA, ldda, dB, 1, queue );
            magma_dtrsv( MagmaLower, MagmaConjTrans, MagmaNonUnit, n, dA, ldda, dB, 1, queue );
        } else {
            magma_dtrsm( MagmaLeft, MagmaLower, MagmaNoTrans,   MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
            magma_dtrsm( MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit, n, nrhs, c_one, dA, ldda, dB, lddb, queue );
        }
    }

    magma_queue_destroy( queue );
    
    return *info;
}
