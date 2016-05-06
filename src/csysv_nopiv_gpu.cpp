/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       @author Adrien REMY

       @generated from src/zsysv_nopiv_gpu.cpp normal z -> c, Mon May  2 23:30:13 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    CSYSV solves a system of linear equations
        A * X = B
    where A is an n-by-n symmetric matrix and X and B are n-by-nrhs matrices.
    The LU decomposition with no pivoting is
    used to factor A as
    The factorization has the form
        A = U^T * D * U,  if UPLO = MagmaUpper, or
        A = L  * D * L^T, if UPLO = MagmaLower,
    where U is an upper triangular matrix, L is lower triangular, and
    D is a diagonal matrix.
    The factored form of A is then
    used to solve the system of equations A * X = B.
    
    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The order of the matrix A.  n >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  nrhs >= 0.

    @param[in,out]
    dA      COMPLEX array, dimension (ldda,n).
            On entry, the n-by-n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  ldda >= max(1,n).

    @param[in,out]
    dB      COMPLEX array, dimension (lddb,nrhs)
            On entry, the right hand side matrix B.
            On exit, the solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array B.  lddb >= max(1,n).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_csysv_driver
    ********************************************************************/
extern "C" magma_int_t
magma_csysv_nopiv_gpu(
    magma_uplo_t uplo,  magma_int_t n, magma_int_t nrhs,
    magmaFloatComplex_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr dB, magma_int_t lddb,
    magma_int_t *info)
{
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

    magma_csytrf_nopiv_gpu(uplo, n, dA, ldda, info);
    if (*info == 0) {
        magma_csytrs_nopiv_gpu( uplo, n, nrhs, dA, ldda, dB, lddb, info );
    }
    return *info;
}
