/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Ichitaro Yamazaki                                                                   
       @author Adrien Remy
       
       @generated from src/zsytrf_nopiv_cpu.cpp normal z -> c, Mon May  2 23:30:12 2016
       
 
*/
#include "magma_internal.h"

#define  A(i, j) ( A[(j)*lda  + (i)])
#define  C(i, j) ( C[(j)*ldc  + (i)])
#define  D(i)    ( D[(i)*incD] )

// trailing submatrix update with inner-blocking 
magma_int_t csyrk_d(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha, magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex beta,  magmaFloatComplex *C, magma_int_t ldc,
    magmaFloatComplex *D, magma_int_t incD)
{
    magmaFloatComplex *Aik;
    magmaFloatComplex *Dkk;
    magmaFloatComplex *Akj;

    /* Check input arguments */
    magma_int_t i, j, k;
    magma_int_t info = 0;
    if ((uplo != MagmaLower) && (uplo != MagmaUpper)) {
        info = -1;
    }
    else if (m < 0) {
        info = -3;
    }
    else if (n < 0) {
        info = -4;
    }
    else if ((lda < max(1, m)) && (m > 0)) {
        info = -7;
    }
    else if ((ldc < max(1, m)) && (m > 0)) {
        info = -10;
    }
    else if ( incD < 0 ) {
        info = -12;
    }
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;        
    }

    /* Quick return */
    if (m == 0 || n == 0 ||
        ((alpha == 0.0 || m == 0) && beta == 1.0) ) {
        return info;
    }

    if ( uplo == MagmaLower ) {
        for (j=0; j < m; j++) {
            for (i=j; i < m; i++) {
                magmaFloatComplex tmp = MAGMA_C_ZERO;
                Aik = A+i;
                Dkk = D;
                Akj = A+j;
                for (k=0; k < n; k++) {
                    tmp += (*Aik) * (*Dkk) * ( *Akj );
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
                magmaFloatComplex tmp = MAGMA_C_ZERO;
                for (k=0; k < n; k++) {
                    tmp += A(i, k) * D( k ) * A(k, j);
                }
                C(i, j) = beta * C(i, j) + alpha * tmp;
            }
        }
    }
    return info;
}


// trailing submatrix update with inner-blocking, using workshpace that
// stores D*L'
magma_int_t csyrk_d_workspace(
    magma_uplo_t uplo, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha, magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex beta,  magmaFloatComplex *C, magma_int_t ldc,
    magmaFloatComplex *work, magma_int_t ldw)
{
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;

    /* Check input arguments */
    magma_int_t info = 0;
    if ((uplo != MagmaLower) && (uplo != MagmaUpper)) {
        info = -1;
    }
    else if (n < 0) {
        info = -2;
    }
    else if (k < 0) {
        info = -3;
    }
    else if ((lda < max(1,n)) && (n > 0)) {
        info = -6;
    }
    else if ((ldc < max(1,n)) && (n > 0)) {
        info = -9;
    }
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;        
    }

    /* Quick return */
    if (n == 0 || k == 0 ||
        ((alpha == 0.0 || k == 0) && beta == 1.0) ) {
        return info;
    }

    if ( uplo == MagmaLower ) {
        blasf77_cgemm( MagmaNoTransStr, MagmaNoTransStr, 
                       &n, &n, &k,
                       &c_neg_one, A,    &lda,
                                   work, &ldw,
                       &c_one,     C,    &ldc );
    }
    else {
        blasf77_cgemm( MagmaNoTransStr, MagmaNoTransStr, 
                       &n, &n, &k,
                       &c_neg_one, work, &ldw,
                                   A,    &lda,
                       &c_one,     C,    &ldc );
    }
    return info;
}


// diagonal factorization with inner-block
magma_int_t csytrf_diag_nopiv(
    magma_uplo_t uplo, magma_int_t n, 
    magmaFloatComplex *A, magma_int_t lda)
{
    /* Constants */
    const magma_int_t ione = 1;
    const magmaFloatComplex c_one = MAGMA_C_ONE;
    
    /* Local variables */
    magmaFloatComplex *Ak1k = NULL;
    magmaFloatComplex Akk;
    magmaFloatComplex alpha;
    
    /* Check input arguments */
    magma_int_t info = 0;
    if (lda < n) {
        info = -4;
    }
    /* TODO: need to check all other arguments */
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;        
    }
    
    /* Quick return */
    if (n <= 1)
        return info;

    if ( uplo == MagmaLower ) {
        /* Diagonal element */
        Akk  = *A;

        /* Pointer on first extra diagonal element */
        Ak1k = A + 1;

        for (magma_int_t k=n-1; k > 0; k--) {
            if ( MAGMA_C_ABS(Akk) < lapackf77_slamch("Epsilon") ) {
                info = k;
                return info;
            }

            // scale off-diagonals
            alpha = MAGMA_C_DIV( c_one, Akk );
            blasf77_cscal(&k, &alpha, Ak1k, &ione);

            // update remaining
            alpha = -( Akk );
            lapackf77_csyr(MagmaLowerStr, &k, 
                           &alpha, Ak1k, &ione, Ak1k + lda, &lda);

            /* Move to next diagonal element */
            Ak1k += lda;
            Akk = *Ak1k;
            Ak1k++;
        }
    } else {
        /* Diagonal element */
        Akk  = *A;

        /* Pointer on first extra diagonal element */
        Ak1k = A + lda;

        for (magma_int_t k=n-1; k > 0; k--) {
            if ( MAGMA_C_ABS(Akk) < lapackf77_slamch("Epsilon") ) {
                info = k;
                return info;
            }

            // scale off-diagonals
            alpha = MAGMA_C_DIV( c_one, Akk );
            blasf77_cscal(&k, &alpha, Ak1k, &lda);

            // update remaining
            alpha = - ( Akk );

            lapackf77_csyr(MagmaUpperStr, &k, 
                         &alpha, Ak1k, &lda, Ak1k + 1, &lda);

            /* Move to next diagonal element */
            Ak1k ++;
            Akk = *Ak1k;
            Ak1k += lda;
        }
    }
    return info;
}


// main routine
extern "C" magma_int_t
magma_csytrf_nopiv_cpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t ib,
    magmaFloatComplex *A, magma_int_t lda,
    magma_int_t *info)
{
    /* Constants */
    const magma_int_t ione = 1;
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;

    /* Local variables */
    magmaFloatComplex alpha;
    
    /* Check input arguments */
    *info = 0;
    if (lda < n) {
        *info = -5;
    }
    /* TODO: need to check all other arguments */
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;        
    }

    /* Quick return */
    if (n == 1) {
        return *info;
    }

    if ( uplo == MagmaLower ) {
        for (magma_int_t i = 0; i < n; i += ib) {
            magma_int_t sb = min(n-i, ib);

            /* Factorize the diagonal block */
            *info = csytrf_diag_nopiv(uplo, sb, &A(i, i), lda);
            if (*info != 0) return *info;

            if ( i + sb < n ) {
                magma_int_t height = n - i - sb;

                /* Solve the lower panel ( L21*D11 )*/
                blasf77_ctrsm(
                    MagmaRightStr, MagmaLowerStr, 
                    MagmaTransStr, MagmaUnitStr,
                    &height, &sb, 
                    &c_one, &A(i, i),    &lda,
                            &A(i+sb, i), &lda);

                /* Scale the block to divide by D */
                for (magma_int_t k=0; k < sb; k++) {
                    #define CSYRK_D_WORKSPACE
                    #ifdef CSYRK_D_WORKSPACE
                    for (magma_int_t ii=i+sb; ii < n; ii++) {
                        A(i+k, ii) = A(ii, i+k);
                    }
                    #endif
                    alpha = MAGMA_C_DIV( c_one, A(i+k, i+k));
                    blasf77_cscal(&height, &alpha, &A(i+sb, i+k), &ione);
                }

                /* Update the trailing submatrix A22 = A22 - A21 * D11 * A21' */
                #ifdef CSYRK_D_WORKSPACE
                csyrk_d_workspace( MagmaLower, height, sb,
                                   c_neg_one, &A(i+sb, i),    lda,    // A21
                                   c_one,     &A(i+sb, i+sb), lda,    // A22
                                              &A(i, i+sb),    lda );  // workspace, I am writing on upper part :)
                #else
                csyrk_d( MagmaLower, height, sb,
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
            *info = csytrf_diag_nopiv(uplo, sb, &A(i, i), lda);
            if (*info != 0) return *info;

            if ( i + sb < n ) {
                magma_int_t height = n - i - sb;

                /* Solve the lower panel ( L21*D11 )*/
                blasf77_ctrsm(
                    MagmaLeftStr, MagmaUpperStr, 
                    MagmaTransStr, MagmaUnitStr,
                    &sb, &height, 
                    &c_one, &A(i, i),    &lda,
                            &A(i, i+sb), &lda);

                /* Scale the block to divide by D */
                for (magma_int_t k=0; k < sb; k++) {
                    #define CSYRK_D_WORKSPACE
                    #ifdef CSYRK_D_WORKSPACE
                    for (magma_int_t ii=i+sb; ii < n; ii++) {
                        A(ii, i+k) = A(i+k, ii);
                    }
                    #endif
                    alpha = MAGMA_C_DIV( c_one, A(i+k, i+k) );
                    blasf77_cscal(&height, &alpha, &A(i+k, i+sb), &lda);
                }

                /* Update the trailing submatrix A22 = A22 - A21 * D11 * A21' */
                #ifdef CSYRK_D_WORKSPACE
                csyrk_d_workspace( MagmaUpper, height, sb,
                                   c_neg_one, &A(i, i+sb),    lda,    // A21
                                   c_one,     &A(i+sb, i+sb), lda,    // A22
                                              &A(i+sb, i),    lda );  // workspace, I am writing on upper part :)
                #else
                csyrk_d( MagmaUpper, height, sb,
                         c_neg_one, &A(i, i+sb),    lda,      // A21
                         c_one,     &A(i+sb, i+sb), lda,      // A22
                                    &A(i, i),       lda+1 );  // D11
                #endif
            }
        }
    }

    return *info;
}
