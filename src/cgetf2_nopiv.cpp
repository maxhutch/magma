/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated c Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"

#define PRECISION_c

extern "C" magma_int_t
magma_cgetf2_nopiv(magma_int_t *m, magma_int_t *n, magmaFloatComplex *a,
                   magma_int_t *lda, magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    CGETF2_NOPIV computes an LU factorization of a general m-by-n
    matrix A without pivoting.

    The factorization has the form
       A = L * U
    where L is lower triangular with unit diagonal elements (lower
    trapezoidal if m > n), and U is upper triangular (upper
    trapezoidal if m < n).

    This is the right-looking Level 2 BLAS version of the algorithm.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX array, dimension (LDA,N)
            On entry, the m by n matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    INFO    (output) INTEGER
            = 0: successful exit
            < 0: if INFO = -k, the k-th argument had an illegal value
            > 0: if INFO = k, U(k,k) is exactly zero. The factorization
                 has been completed, but the factor U is exactly
                 singular, and division by zero will occur if it is used
                 to solve a system of equations.
    =====================================================================   */

    magmaFloatComplex c_one = MAGMA_C_ONE;
    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t ione = 1;

    magma_int_t a_dim1, a_offset, i__1, i__2, i__3;
    magmaFloatComplex z__1;
    magma_int_t i__, j;
    float sfmin;

    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    if (*m < 0) {
        *info = -1;
    } else if (*n < 0) {
        *info = -2;
    } else if (*lda < max(1,*m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (*m == 0 || *n == 0)
        return *info;

    /* Compute machine safe minimum */
    sfmin = lapackf77_slamch("S");

    i__1 = min(*m,*n);
    for (j = 1; j <= i__1; ++j) {
        /* Test for singularity. */
        i__2 = j + j * a_dim1;
        if (!MAGMA_C_EQUAL(a[i__2], c_zero)) {

            /* Compute elements J+1:M of J-th column. */
            if (j < *m) {
                if (MAGMA_C_ABS(a[j + j * a_dim1]) >= sfmin) {
                    i__2 = *m - j;
                    z__1 = MAGMA_C_DIV(c_one, a[j + j * a_dim1]);
                    blasf77_cscal(&i__2, &z__1, &a[j + 1 + j * a_dim1], &ione);
                }
                else {
                    i__2 = *m - j;
                    for (i__ = 1; i__ <= i__2; ++i__) {
                        i__3 = j + i__ + j * a_dim1;
                        a[i__3] = MAGMA_C_DIV(a[j + i__ + j * a_dim1], a[j + j*a_dim1]);
                    }
                }
            }
        }
        else if (*info == 0) {
            *info = j;
        }

        if (j < min(*m,*n)) {
            /* Update trailing submatrix. */
            i__2 = *m - j;
            i__3 = *n - j;
            blasf77_cgeru( &i__2, &i__3, &c_neg_one,
                           &a[j + 1 + j * a_dim1], &ione,
                           &a[j + (j+1) * a_dim1], lda,
                           &a[j + 1 + (j+1) * a_dim1], lda);
        }
    }

    return *info;
} /* magma_cgetf2_nopiv */
