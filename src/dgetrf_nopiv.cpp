/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zgetrf_nopiv.cpp normal z -> d, Fri Jan 30 19:00:14 2015

*/
#include "common_magma.h"

#define PRECISION_d

/**
    Purpose
    -------
    DGETRF_NOPIV computes an LU factorization of a general M-by-N
    matrix A without pivoting.

    The factorization has the form
       A = L * U
    where L is lower triangular with unit diagonal elements (lower
    trapezoidal if m > n), and U is upper triangular (upper
    trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       DOUBLE_PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @ingroup magma_dgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dgetrf_nopiv(
    magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    magma_int_t *info)
{
    #define A(i_,j_) (A + (i_) + (j_)*lda)
    
    double c_one = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    
    magma_int_t min_mn, i__3, i__4;
    magma_int_t j, jb, nb, iinfo;

    A -= 1 + lda;

    /* Function Body */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return *info;
    }

    /* Determine the block size for this environment. */
    nb = 128;
    min_mn = min(m,n);
    if (nb <= 1 || nb >= min_mn) {
        /* Use unblocked code. */
        magma_dgetf2_nopiv( m, n, A(1,1), lda, info );
    }
    else {
        /* Use blocked code. */
        for (j = 1; j <= min_mn; j += nb) {
            jb = min( min_mn - j + 1, nb );
            
            /* Factor diagonal and subdiagonal blocks and test for exact
               singularity. */
            i__3 = m - j + 1;
            //magma_dgetf2_nopiv( i__3, jb, A(j,j), lda, &iinfo );

            i__3 -= jb;
            magma_dgetf2_nopiv( jb, jb, A(j,j), lda, &iinfo );
            blasf77_dtrsm( "R", "U", "N", "N", &i__3, &jb, &c_one,
                           A(j,j),    &lda,
                           A(j+jb,j), &lda );
            
            /* Adjust INFO */
            if (*info == 0 && iinfo > 0)
                *info = iinfo + j - 1;

            if (j + jb <= n) {
                /* Compute block row of U. */
                i__3 = n - j - jb + 1;
                blasf77_dtrsm( "Left", "Lower", "No transpose", "Unit",
                               &jb, &i__3, &c_one,
                               A(j,j),    &lda,
                               A(j,j+jb), &lda );
                if (j + jb <= m) {
                    /* Update trailing submatrix. */
                    i__3 = m - j - jb + 1;
                    i__4 = n - j - jb + 1;
                    blasf77_dgemm( "No transpose", "No transpose",
                                   &i__3, &i__4, &jb, &c_neg_one,
                                   A(j+jb,j),    &lda,
                                   A(j,j+jb),    &lda, &c_one,
                                   A(j+jb,j+jb), &lda );
                }
            }
        }
    }
    
    return *info;
} /* magma_dgetrf_nopiv */
