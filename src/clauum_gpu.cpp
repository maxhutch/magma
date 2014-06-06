/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:34 2013

*/
#include "common_magma.h"

#define dA(i, j) (dA+(j)*ldda + (i))

extern "C" magma_int_t
magma_clauum_gpu(char uplo, magma_int_t n,
                 magmaFloatComplex  *dA, magma_int_t ldda, magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    CLAUUM computes the product U * U' or L' * L, where the triangular
    factor U or L is stored in the upper or lower triangular part of
    the array dA.

    If UPLO = 'U' or 'u' then the upper triangle of the result is stored,
    overwriting the factor U in dA.
    If UPLO = 'L' or 'l' then the lower triangle of the result is stored,
    overwriting the factor L in dA.
    This is the blocked form of the algorithm, calling Level 3 BLAS.

    Arguments
    =========
    UPLO    (input) CHARACTER*1
            Specifies whether the triangular factor stored in the array dA
            is upper or lower triangular:
            = 'U':  Upper triangular
            = 'L':  Lower triangular

    N       (input) INTEGER
            The order of the triangular factor U or L.  N >= 0.

    dA      (input/output) REAL array on the GPU, dimension (LDDA,N)
            On entry, the triangular factor U or L.
            On exit, if UPLO = 'U', the upper triangle of dA is
            overwritten with the upper triangle of the product U * U';
            if UPLO = 'L', the lower triangle of dA is overwritten with
            the lower triangle of the product L' * L.

    LDDA    (input) INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value

    ===================================================================== */

    /* Local variables */
    char uplo_[2] = {uplo, 0};
    magma_int_t         nb, i, ib;
    float              d_one = MAGMA_D_ONE;
    magmaFloatComplex  c_one = MAGMA_C_ONE;
    magmaFloatComplex  *work;

    int upper  = lapackf77_lsame(uplo_, "U");

    *info = 0;

    if ((! upper) && (! lapackf77_lsame(uplo_, "L")))
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < max(1,n))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    nb = magma_get_cpotrf_nb(n);

    if (MAGMA_SUCCESS != magma_cmalloc_pinned( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    magma_queue_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    if (nb <= 1 || nb >= n)
    {
        magma_cgetmatrix( n, n, dA, ldda, work, n );
        lapackf77_clauum(uplo_, &n, work, &n, info);
        magma_csetmatrix( n, n, work, n, dA, ldda );
    }
    else
    {
        if (upper)
        {
            /* Compute inverse of upper triangular matrix */
            for (i=0; i < n; i += nb)
            {
                ib = min(nb, (n-i));

                /* Compute the product U * U'. */
                magma_ctrmm( MagmaRight, MagmaUpper,
                         MagmaConjTrans, MagmaNonUnit, i, ib,
                         c_one, dA(i,i), ldda, dA(0, i),ldda);

                magma_cgetmatrix( ib, ib,
                                  dA(i, i), ldda,
                                  work,     ib );

                lapackf77_clauum(MagmaUpperStr, &ib, work, &ib, info);

                magma_csetmatrix( ib, ib,
                                  work,     ib,
                                  dA(i, i), ldda );

                if(i+ib < n)
                {
                    magma_cgemm( MagmaNoTrans, MagmaConjTrans,
                                 i, ib, (n-i-ib), c_one, dA(0,i+ib),
                                 ldda, dA(i, i+ib), ldda, c_one,
                                 dA(0,i), ldda);

                    magma_cherk( MagmaUpper, MagmaNoTrans, ib,(n-i-ib),
                                 d_one, dA(i, i+ib), ldda,
                                 d_one, dA(i, i),    ldda);
                }
            }
        }
        else
        {
            /* Compute the product L' * L. */
            for(i=0; i<n; i=i+nb)
            {
                ib=min(nb,(n-i));

                magma_ctrmm( MagmaLeft, MagmaLower,
                             MagmaConjTrans, MagmaNonUnit, ib,
                             i, c_one, dA(i,i), ldda,
                             dA(i, 0),ldda);

                magma_cgetmatrix( ib, ib,
                                  dA(i, i), ldda,
                                  work,     ib );

                lapackf77_clauum(MagmaLowerStr, &ib, work, &ib, info);

                magma_csetmatrix( ib, ib,
                                  work,     ib,
                                  dA(i, i), ldda );

                if((i+ib) < n)
                {
                    magma_cgemm( MagmaConjTrans, MagmaNoTrans,
                                 ib, i, (n-i-ib), c_one, dA( i+ib,i),
                                 ldda, dA(i+ib, 0),ldda, c_one,
                                 dA(i,0), ldda);
                    magma_cherk( MagmaLower, MagmaConjTrans, ib, (n-i-ib),
                                 d_one, dA(i+ib, i), ldda,
                                 d_one, dA(i, i),    ldda);
                }
            }
        }
    }

    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );

    magma_free_pinned( work );

    return *info;
}
