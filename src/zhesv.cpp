/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @precisions normal z -> s d c
*/

#include "magma_internal.h"

/**
    Purpose
    -------
    ZHESV computes the solution to a complex system of linear equations
       A * X = B,
    where A is an n-by-n Hermitian matrix and X and B are n-by-nrhs
    matrices.

    The diagonal pivoting method is used to factor A as
       A = U * D * U**H,  if uplo = MagmaUpper, or
       A = L * D * L**H,  if uplo = MagmaLower,
    where U (or L) is a product of permutation and unit upper (lower)
    triangular matrices, and D is Hermitian and block diagonal with
    1-by-1 and 2-by-2 diagonal blocks.  The factored form of A is then
    used to solve the system of equations A * X = B.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            = MagmaUpper:  Upper triangle of A is stored;
            = MagmaLower:  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The number of linear equations, i.e., the order of the
            matrix A.  n >= 0.
 
    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  nrhs >= 0.

    @param[in,out]
    A       COMPLEX*16 array, dimension (lda,n)
            On entry, the Hermitian matrix A.  If uplo = MagmaUpper, the leading
            n-by-n upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If uplo = MagmaLower, the
            leading n-by-n lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.

            On exit, if info = 0, the block diagonal matrix D and the
            multipliers used to obtain the factor U or L from the
            factorization A = U*D*U**H or A = L*D*L**H as computed by
            ZHETRF.
 
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  lda >= max(1,n).
 
    @param[out]
    ipiv    INTEGER array, dimension (n)
            Details of the interchanges and the block structure of D, as
            determined by ZHETRF.  If ipiv(k) > 0, then rows and columns
            k and ipiv(k) were interchanged, and D(k,k) is a 1-by-1
            diagonal block.  If uplo = MagmaUpper and ipiv(k) = ipiv(k-1) < 0,
            then rows and columns k-1 and -ipiv(k) were interchanged and
            D(k-1:k,k-1:k) is a 2-by-2 diagonal block.  If uplo = MagmaLower and
            ipiv(k) = ipiv(k+1) < 0, then rows and columns k+1 and
            -ipiv(k) were interchanged and D(k:k+1,k:k+1) is a 2-by-2
            diagonal block.

    @param[in,out]
    B       (input/output) COMPLEX*16 array, dimension (ldb,nrhs)
            On entry, the n-by-nrhs right hand side matrix B.
            On exit, if info = 0, the n-by-nrhs solution matrix X.

    @param[in]
    ldb     INTEGER
            The leading dimension of the array B.  ldb >= max(1,n).

    @param[out]
    info    INTEGER
            = 0: successful exit
            < 0: if info = -i, the i-th argument had an illegal value
            > 0: if info = i, D(i,i) is exactly zero.  The factorization
                 has been completed, but the block diagonal matrix D is
                 exactly singular, so the solution could not be computed.

    @ingroup magma_zhesv_driver
    ********************************************************************/
extern "C" magma_int_t
magma_zhesv(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    magmaDoubleComplex *A, magma_int_t lda, magma_int_t *ipiv,
    magmaDoubleComplex *B, magma_int_t ldb,
    magma_int_t *info )
{
    /* Local variables */
    bool upper = (uplo == MagmaUpper);

    /* Test the input parameters. */
    *info = 0;
    if ( !upper && uplo != MagmaLower ) {
        *info = -1;
    } else if ( n < 0 ) {
        *info = -2;
    } else if ( nrhs < 0 ) {
        *info = -3;
    } else if ( lda < max( 1, n ) ) {
        *info = -5;
    } else if ( ldb < max( 1, n ) ) {
        *info = -8;
    }

    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Compute the factorization A = U*D*U' or A = L*D*L'. */
    magma_zhetrf( uplo, n, A, lda, ipiv, info );
    if ( *info == 0 ) {
        /* Solve the system A*X = B, overwriting B with X. */
        lapackf77_zhetrs( (upper ? MagmaUpperStr: MagmaLowerStr),
                           &n, &nrhs, A, &lda, ipiv, B, &ldb, info );
    }

    return *info;
}
