/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zgetrf_nopiv.cpp normal z -> c, Mon May  2 23:30:05 2016

*/
#include "magma_internal.h"

/**
    Purpose
    -------
    CGETRF_NOPIV computes an LU factorization of a general M-by-N
    matrix A without pivoting.

    The factorization has the form
       A = L * U
    where L is lower triangular with unit diagonal elements (lower
    trapezoidal if m > n), and U is upper triangular (upper
    trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    
    This is a CPU-only (not accelerated) version.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX array, dimension (LDA,N)
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

    @ingroup magma_cgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_cgetrf_nopiv(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info)
{
    #define A(i_,j_) (A + (i_) + (j_)*lda)
    
    magmaFloatComplex c_one = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    
    magma_int_t min_mn, m_j_jb, n_j_jb;
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
        magma_cgetf2_nopiv( m, n, A(1,1), lda, info );
    }
    else {
        /* Use blocked code. */
        for (j = 1; j <= min_mn; j += nb) {
            jb = min( min_mn - j + 1, nb );
            
            /* Factor diagonal and subdiagonal blocks and test for exact
               singularity. */
            m_j_jb = m - j - jb + 1;
            magma_cgetf2_nopiv( jb, jb, A(j,j), lda, &iinfo );
            blasf77_ctrsm( "R", "U", "N", "N", &m_j_jb, &jb, &c_one,
                           A(j,j),    &lda,
                           A(j+jb,j), &lda );
            
            /* Adjust INFO */
            if (*info == 0 && iinfo > 0)
                *info = iinfo + j - 1;

            if (j + jb <= n) {
                /* Compute block row of U. */
                n_j_jb = n - j - jb + 1;
                blasf77_ctrsm( "Left", "Lower", "No transpose", "Unit",
                               &jb, &n_j_jb, &c_one,
                               A(j,j),    &lda,
                               A(j,j+jb), &lda );
                if (j + jb <= m) {
                    /* Update trailing submatrix. */
                    m_j_jb = m - j - jb + 1;
                    n_j_jb = n - j - jb + 1;
                    blasf77_cgemm( "No transpose", "No transpose",
                                   &m_j_jb, &n_j_jb, &jb, &c_neg_one,
                                   A(j+jb,j),    &lda,
                                   A(j,j+jb),    &lda, &c_one,
                                   A(j+jb,j+jb), &lda );
                }
            }
        }
    }
    
    return *info;
} /* magma_cgetrf_nopiv */
