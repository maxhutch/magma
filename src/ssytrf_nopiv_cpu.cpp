/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zhetrf_nopiv_cpu.cpp normal z -> s, Mon May  2 23:30:12 2016
 
*/
#include "magma_internal.h"

#define REAL

// TODO convert to usual (A + (i) + (j)*lda), i.e., returns pointer?
#define  A(i, j) ( A[(j)*lda  + (i)])
#define  C(i, j) ( C[(j)*ldc  + (i)])
#define  D(i)    ( D[(i)*incD] )

// TODO: change alpha and beta to be float, per BLAS, instead of float-real
// trailing submatrix update with inner-blocking
magma_int_t ssyrk_d(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    float alpha, float *A, magma_int_t lda,
    float beta,  float *C, magma_int_t ldc,
    float *D, magma_int_t incD)
{
    magma_int_t i, j, k;
    float *Aik;
    float *Dkk;
    float *Akj;

    /* Check input arguments */
    if ((uplo != MagmaLower) && (uplo != MagmaUpper)) {
        return -1;
    }
    if (m < 0) {
        return -3;
    }
    if (n < 0) {
        return -4;
    }
    if ((lda < max(1, m)) && (m > 0)) {
        return -7;
    }
    if ((ldc < max(1, m)) && (m > 0)) {
        return -10;
    }
    if ( incD < 0 ) {
        return -12;
    }

    /* Quick return */
    if (m == 0 || n == 0 ||
        ((alpha == 0.0 || m == 0) && beta == 1.0) ) {
        return MAGMA_SUCCESS;
    }

    if ( uplo == MagmaLower ) {
        for (j=0; j < m; j++) {
            for (i=j; i < m; i++) {
                float tmp = MAGMA_S_ZERO;
                Aik = A+i;
                Dkk = D;
                Akj = A+j;
                for (k=0; k < n; k++) {
                    tmp += (*Aik) * (*Dkk) * conj( *Akj );
                    Aik += lda;
                    Dkk += incD;
                    Akj += lda;
                }
                C(i, j) = beta * C(i, j) + alpha * tmp;
            }
        }
    }
    else {
        for (j=0; j < m; j++) {
            for (i=0; i <= j; i++) {
                float tmp = MAGMA_S_ZERO;
                for (k=0; k < n; k++) {
                    tmp += A(i, k) * D( k ) * conj( A(k, j) );
                }
                C(i, j) = beta * C(i, j) + alpha * tmp;
            }
        }
    }
    return MAGMA_SUCCESS;
}


// TODO: change alpha and beta to be float, per BLAS, instead of float-real
// trailing submatrix update with inner-blocking, using workshpace that
// stores D*L'
magma_int_t ssyrk_d_workspace(
    magma_uplo_t uplo, magma_int_t n, magma_int_t k,
    float alpha, float *A, magma_int_t lda,
    float beta,  float *C, magma_int_t ldc,
    float *work, magma_int_t ldw)
{
    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;

    /* Check input arguments */
    if ((uplo != MagmaLower) && (uplo != MagmaUpper)) {
        return -1;
    }
    if (n < 0) {
        return -2;
    }
    if (k < 0) {
        return -3;
    }
    if ((lda < max(1,n)) && (n > 0)) {
        return -6;
    }
    if ((ldc < max(1,n)) && (n > 0)) {
        return -9;
    }

    /* Quick return */
    if (n == 0 || k == 0 ||
        ((alpha == 0.0 || k == 0) && beta == 1.0) ) {
        return MAGMA_SUCCESS;
    }

    if ( uplo == MagmaLower ) {
        blasf77_sgemm( MagmaNoTransStr, MagmaNoTransStr,
                       &n, &n, &k,
                       &c_neg_one, A,    &lda,
                                   work, &ldw,
                       &c_one,     C,    &ldc );
    }
    else {
        blasf77_sgemm( MagmaNoTransStr, MagmaNoTransStr,
                       &n, &n, &k,
                       &c_neg_one, work, &ldw,
                                   A,    &lda,
                       &c_one,     C,    &ldc );
    }
    return MAGMA_SUCCESS;
}


// diagonal factorization with inner-block
magma_int_t ssytrf_diag_nopiv(
    magma_uplo_t uplo, magma_int_t n,
    float *A, magma_int_t lda)
{
    /* Quick return */
    if (n == 1)
        return 0;
    if (lda < n)
        return -1;

    /**/
    magma_int_t info = 0, ione = 1;
    float *Ak1k = NULL;
    float *Akk = NULL;
    float d_one = 1.0;
    float alpha;

    if ( uplo == MagmaLower ) {
        /* Diagonal element */
        Akk  = A;

        /* Pointer on first extra diagonal element */
        Ak1k = A + 1;

        for (magma_int_t k=n-1; k > 0; k--) {
            alpha = MAGMA_S_REAL( *Akk );
            if ( fabs(alpha) < lapackf77_slamch("Epsilon") ) {
                info = k;
                return info;
            }
            *Akk = MAGMA_S_MAKE(alpha, 0.0);

            // scale off-diagonals
            alpha = d_one / alpha;
            blasf77_sscal(&k, &alpha, Ak1k, &ione);

            // update remaining
            alpha = - MAGMA_S_REAL( *Akk );
            blasf77_ssyr(MagmaLowerStr, &k,
                         &alpha, Ak1k, &ione, Ak1k + lda, &lda);

            /* Move to next diagonal element */
            if (k > 1) { 
                Ak1k += lda;
                Akk = Ak1k;
                Ak1k++;
            }
        }
    } else {
        /* Diagonal element */
        Akk  = A;

        /* Pointer on first extra diagonal element */
        Ak1k = A + lda;

        for (magma_int_t k=n-1; k > 0; k--) {
            alpha = MAGMA_S_REAL( *Akk );
            if ( fabs(alpha) < lapackf77_slamch("Epsilon") ) {
                info = k;
                return info;
            }
            *Akk = MAGMA_S_MAKE(alpha, 0.0);

            // scale off-diagonals
            alpha = d_one / alpha;
            blasf77_sscal(&k, &alpha, Ak1k, &lda);

            // update remaining
            alpha = - MAGMA_S_REAL( *Akk );

            #ifdef COMPLEX
            lapackf77_slacgv(&k, Ak1k, &lda);
            #endif
            blasf77_ssyr(MagmaUpperStr, &k,
                         &alpha, Ak1k, &lda, Ak1k + 1, &lda);
            #ifdef COMPLEX
            lapackf77_slacgv(&k, Ak1k, &lda);
            #endif

            /* Move to next diagonal element */
            if (k > 1) {
                Ak1k ++;
                Akk = Ak1k;
                Ak1k += lda;
            }
        }
    }
    return info;
}


// main routine
extern "C" magma_int_t
magma_ssytrf_nopiv_cpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t ib,
    float *A, magma_int_t lda,
    magma_int_t *info)
{
    magma_int_t ione = 1;
    float alpha;
    float d_one = 1.0;
    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;

    /* Check input arguments */
    if (lda < n) {
        *info = -1;
        return *info;
    }

    *info = 0;
    /* Quick return */
    if (n == 1) {
        return *info;
    }

    if ( uplo == MagmaLower ) {
        for (magma_int_t i = 0; i < n; i += ib) {
            magma_int_t sb = min(n-i, ib);

            /* Factorize the diagonal block */
            *info = ssytrf_diag_nopiv(uplo, sb, &A(i, i), lda);
            if (*info != 0) return *info;

            if ( i + sb < n ) {
                magma_int_t height = n - i - sb;

                /* Solve the lower panel ( L21*D11 )*/
                blasf77_strsm(
                    MagmaRightStr, MagmaLowerStr,
                    MagmaConjTransStr, MagmaUnitStr,
                    &height, &sb,
                    &c_one, &A(i, i),    &lda,
                            &A(i+sb, i), &lda);

                /* Scale the block to divide by D */
                for (magma_int_t k=0; k < sb; k++) {
                    #define SSYRK_D_WORKSPACE
                    #ifdef SSYRK_D_WORKSPACE
                    for (magma_int_t ii=i+sb; ii < n; ii++) {
                        A(i+k, ii) = MAGMA_S_CONJ( A(ii, i+k) );
                    }
                    #endif
                    alpha = d_one / MAGMA_S_REAL(A(i+k, i+k));
                    blasf77_sscal(&height, &alpha, &A(i+sb, i+k), &ione);
                    A(i+k, i+k) = MAGMA_S_MAKE(MAGMA_S_REAL(A(i+k, i+k)), 0.0);
                }

                /* Update the trailing submatrix A22 = A22 - A21 * D11 * A21' */
                #ifdef SSYRK_D_WORKSPACE
                ssyrk_d_workspace( MagmaLower, height, sb,
                                   c_neg_one, &A(i+sb, i),    lda,    // A21
                                   c_one,     &A(i+sb, i+sb), lda,    // A22
                                              &A(i, i+sb),    lda );  // workspace, I am writing on upper part :)
                #else
                ssyrk_d( MagmaLower, height, sb,
                         c_neg_one, &A(i+sb, i),    lda,      // A21
                         c_one,     &A(i+sb, i+sb), lda,      // A22
                                    &A(i, i),       lda+1 );  // D11
                #endif
            }
        }
    } else {
        for (magma_int_t i = 0; i < n; i += ib) {
            magma_int_t sb = min(n-i, ib);

            /* Factorize the diagonal block */
            *info = ssytrf_diag_nopiv(uplo, sb, &A(i, i), lda);
            if (*info != 0) return *info;

            if ( i + sb < n ) {
                magma_int_t height = n - i - sb;

                /* Solve the lower panel ( L21*D11 )*/
                blasf77_strsm(
                    MagmaLeftStr, MagmaUpperStr,
                    MagmaConjTransStr, MagmaUnitStr,
                    &sb, &height,
                    &c_one, &A(i, i),    &lda,
                            &A(i, i+sb), &lda);

                /* Scale the block to divide by D */
                for (magma_int_t k=0; k < sb; k++) {
                    #define SSYRK_D_WORKSPACE
                    #ifdef SSYRK_D_WORKSPACE
                    for (magma_int_t ii=i+sb; ii < n; ii++) {
                        A(ii, i+k) = MAGMA_S_CONJ( A(i+k, ii) );
                    }
                    #endif
                    alpha = d_one / MAGMA_S_REAL(A(i+k, i+k));
                    blasf77_sscal(&height, &alpha, &A(i+k, i+sb), &lda);
                    A(i+k, i+k) = MAGMA_S_MAKE(MAGMA_S_REAL(A(i+k, i+k)), 0.0);
                }

                /* Update the trailing submatrix A22 = A22 - A21 * D11 * A21' */
                #ifdef SSYRK_D_WORKSPACE
                ssyrk_d_workspace( MagmaUpper, height, sb,
                                   c_neg_one, &A(i, i+sb),    lda,    // A21
                                   c_one,     &A(i+sb, i+sb), lda,    // A22
                                              &A(i+sb, i),    lda );  // workspace, I am writing on upper part :)
                #else
                ssyrk_d( MagmaUpper, height, sb,
                         c_neg_one, &A(i, i+sb),    lda,      // A21
                         c_one,     &A(i+sb, i+sb), lda,      // A22
                                    &A(i, i),       lda+1 );  // D11
                #endif
            }
        }
    }

    return *info;
}
