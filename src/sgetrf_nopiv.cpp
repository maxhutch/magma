/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:36 2013

*/
#include "common_magma.h"

#define PRECISION_s

extern "C" magma_int_t
magma_sgetf2_nopiv(magma_int_t *m, magma_int_t *n, float *a,
                   magma_int_t *lda, magma_int_t *info);


extern "C" magma_int_t
magma_sgetrf_nopiv(magma_int_t *m, magma_int_t *n, float *a,
                   magma_int_t *lda, magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    SGETRF_NOPIV computes an LU factorization of a general M-by-N
    matrix A without pivoting.

    The factorization has the form
       A = L * U
    where L is lower triangular with unit diagonal elements (lower
    trapezoidal if m > n), and U is upper triangular (upper
    trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) REAL array, dimension (LDA,N)
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.
    =====================================================================   */
    
    float c_one = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;
    
    magma_int_t a_dim1, a_offset, min_mn, i__3, i__4;
    magma_int_t j, jb, nb, iinfo;

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
    if (*m == 0 || *n == 0) {
        return *info;
    }

    /* Determine the block size for this environment. */
    nb = 128;
    min_mn = min(*m,*n);
    if (nb <= 1 || nb >= min_mn) {
        /* Use unblocked code. */
        magma_sgetf2_nopiv(m, n, &a[a_offset], lda, info);
    }
    else {
        /* Use blocked code. */
        for (j = 1; j <= min_mn; j += nb) {
            /* Computing MIN */
            i__3 = min_mn - j + 1;
            jb = min(i__3,nb);
            
            /* Factor diagonal and subdiagonal blocks and test for exact
               singularity. */
            i__3 = *m - j + 1;
            //magma_sgetf2_nopiv(&i__3, &jb, &a[j + j * a_dim1], lda, &iinfo);

            i__3 -= jb;
            magma_sgetf2_nopiv(&jb, &jb, &a[j + j * a_dim1], lda, &iinfo);
            blasf77_strsm("R", "U", "N", "N", &i__3, &jb, &c_one,
                          &a[j + j * a_dim1], lda,
                          &a[j + jb + j * a_dim1], lda);
            
            /* Adjust INFO */
            if (*info == 0 && iinfo > 0)
                *info = iinfo + j - 1;

            if (j + jb <= *n) {
                /* Compute block row of U. */
                i__3 = *n - j - jb + 1;
                blasf77_strsm("Left", "Lower", "No transpose", "Unit", &jb, &i__3,
                       &c_one, &a[j + j * a_dim1], lda, &a[j + (j+jb)*a_dim1], lda);
                if (j + jb <= *m) {
                    /* Update trailing submatrix. */
                    i__3 = *m - j - jb + 1;
                    i__4 = *n - j - jb + 1;
                    blasf77_sgemm("No transpose", "No transpose", &i__3, &i__4, &jb,
                           &c_neg_one, &a[j + jb + j * a_dim1], lda,
                           &a[j + (j + jb) * a_dim1], lda, &c_one,
                           &a[j + jb + (j + jb) * a_dim1], lda);
                }
            }
        }
    }
    
    return *info;
} /* magma_sgetrf_nopiv */
