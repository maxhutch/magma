/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from ztrsm.cu normal z -> d, Fri May 30 10:40:44 2014

       @author Peng Du
       @author Tingxing Dong
*/
#include "common_magma.h"

#define BLOCK_SIZE 16 // inner blocking size, <=32
#define nb 128        // outer blocking size, >BLOCK_SIZE

__global__ void
dtrsm_copy_kernel (int m, int n, double *b, int ldb, double *d_x, int ldx)
{
    int by = blockIdx.y;
    int gx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gx < m)
        b[by*ldb+gx] = d_x[by*ldx+gx];
}


#define MAX_THREAD_PER_BLOCK 512
#define WARP_SIZE 32


#define dtrsm_copy() \
    do { \
        dim3 dimBlock( (m >= MAX_THREAD_PER_BLOCK) ? MAX_THREAD_PER_BLOCK : (WARP_SIZE*((m/WARP_SIZE)+(m % WARP_SIZE != 0))), 1 ); \
        dim3 dimGrid( (m - 1)/dimBlock.x + 1, n ); \
        dtrsm_copy_kernel<<< dimGrid, dimBlock, 0, magma_stream >>>(m, n, b, ldb, d_x, m); \
        magma_device_sync(); \
    } while(0)


/*
 * magmablas_dtrsm
 */

extern "C"
void diag_dtrtri (magma_int_t m, magma_uplo_t uplo, magma_diag_t diag, const double *A, double *d_dinvA, magma_int_t lda);

/**
    Purpose
    -------

    dtrsm solves one of the matrix equations on gpu

        op( A )*x = alpha*b,   or   x*op( A ) = alpha*b,

    where alpha is a scalar, x and b are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op( A ) is one of

        op( A ) = A   or   op( A ) = A^T.

    The matrix X is overwritten on B.


    Arguments
    ----------

    @param[in]
    side    magma_side_t.
            On entry, side specifies whether op( A ) appears on the left
            or right of X as follows:
      -     = MagmaLeft:       op( A )*X = alpha*B.
      -     = MagmaRight:      X*op( A ) = alpha*B.

    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A^T.
      -     = MagmaTrans:  op( A ) = A^T.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    m       INTEGER.
            On entry, m specifies the number of rows of B. m must be at
            least zero.

    @param[in]
    n       INTEGER.
            On entry, n specifies the number of columns of B. n must be
            at least zero.

    @param[in]
    alpha   COMPLEX.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    A       COMPLEX array of DIMENSION ( lda, k ), where k is m
            when side = MagmaLeft and is n when side = MagmaRight.
            Before entry with uplo = MagmaUpper, the leading k by k
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with uplo = MagmaLower, the leading k by k
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when diag = MagmaUnit, the diagonal elements of
            A are not referenced either, but are assumed to be unity.

    @param[in]
    lda     INTEGER.
            On entry, lda specifies the first dimension of A as declared
            in the calling (sub) program. When side = MagmaLeft then
            lda must be at least max( 1, m ), when side = MagmaRight
            then lda must be at least max( 1, n ).

    @param[in,out]
    b       COMPLEX array of DIMENSION ( ldb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B, and on exit is
            overwritten by the solution matrix X.

    @param[in]
    ldb     INTEGER.
            On entry, ldb specifies the first dimension of B as declared
            in the calling (sub) program. ldb must be at least
            max( 1, m ).

    Level 3 Blas routine.

    @ingroup magma_dblas3
    ********************************************************************/
extern "C"
void magmablas_dtrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, magma_int_t m, magma_int_t n,
    double alpha,
    const double* A, magma_int_t lda,
    double* b, magma_int_t ldb )
{
    int i;
    double *d_dinvA, *d_x;

    /* quick return on wrong size */
    if (m <= 0 || n <= 0)
        return;
    
    magma_trans_t Notrans   = MagmaNoTrans;
    magma_trans_t Trans     = MagmaTrans;
    magma_trans_t Conjtrans = MagmaTrans;

    double neg_one = MAGMA_D_NEG_ONE;
    double one = MAGMA_D_ONE;
    double zero = MAGMA_D_ZERO;

    if (side == MagmaLeft) {
        // side=L
        /* invert the diagonals
         * Allocate device memory for the inverted diagonal blocks, size=m*nb
         */
        magma_dmalloc( &d_dinvA, nb*((m/nb)+(m % nb != 0))*nb );
        magma_dmalloc( &d_x,     n*m );

        cudaMemset(d_x,     0, n*m*sizeof(double));
        cudaMemset(d_dinvA, 0, nb*((m/nb)+(m % nb != 0))*nb*sizeof(double));
        diag_dtrtri (m, uplo, diag, A, d_dinvA, lda);

        if (transA == MagmaNoTrans) {
            /* the non-transpose case */
            if (uplo == MagmaLower) {

                /* the lower case */
                /* handle the first block seperately with alpha */
                int mm = min(nb, m);
                magma_dgemm(Notrans, Notrans, mm, n, mm, alpha, d_dinvA, nb, b, ldb, zero, d_x, m);

                if (nb >= m) {
                    dtrsm_copy();
                    magma_free( d_dinvA );
                    magma_free( d_x );
                    return;
                }

                magma_dgemm(Notrans, Notrans, m-nb, n, nb, neg_one, A+nb, lda, d_x, m, alpha, b+nb, ldb);

                /* the rest blocks */
                for( i=nb; i < m; i += nb ) {
                    mm = min(m-i, nb);
                    magma_dgemm(Notrans, Notrans, mm, n, mm, one, d_dinvA+i*nb, nb, b+i, ldb, zero, d_x+i, m);

                    if (i+nb >= m)
                        break;

                    magma_dgemm(Notrans, Notrans, m-i-nb, n, nb, neg_one, A+i*lda+i+nb, lda, d_x+i, m, one, b+i+nb, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int mm = (m % nb == 0) ? nb : (m % nb);
                i = m-mm;
                magma_dgemm(Notrans, Notrans, mm, n, mm, alpha, d_dinvA+i*nb, nb, b+i, ldb, zero, d_x+i, m);

                if (i-nb < 0) {
                    dtrsm_copy();
                    magma_free( d_dinvA );
                    magma_free( d_x );
                    return;
                }

                magma_dgemm(Notrans, Notrans, i, n, mm, neg_one, A+i*lda, lda, d_x+i, m, alpha, b, ldb);

                /* the rest blocks */
                for( i=m-mm-nb; i >= 0; i -= nb ) {
                    magma_dgemm(Notrans, Notrans, nb, n, nb, one, d_dinvA+i*nb, nb, b+i, ldb, zero, d_x+i, m);

                    if (i-nb < 0)
                        break;

                    magma_dgemm(Notrans, Notrans, i, n, nb, neg_one, A+i*lda, lda, d_x+i, m, one, b, ldb);
                }
            }
        }
        else if( transA == MagmaTrans) {
            /* the transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int mm = (m % nb == 0) ? nb : (m % nb);
                i = m-mm;
                magma_dgemm(Trans, Notrans, mm, n, mm, alpha, d_dinvA+i*nb, nb, b+i, ldb, zero, d_x+i, m);

                if (i-nb < 0) {
                    dtrsm_copy();
                    magma_free( d_dinvA );
                    magma_free( d_x );
                    return;
                }

                magma_dgemm(Trans, Notrans, i, n, mm, neg_one, A+i, lda, d_x+i, m, alpha, b, ldb);

                /* the rest blocks */
                for( i=m-mm-nb; i >= 0; i -= nb ) {
                    magma_dgemm(Trans, Notrans, nb, n, nb, one, d_dinvA+i*nb, nb, b+i, ldb, zero, d_x+i, m);

                    if (i-nb < 0)
                        break;

                    magma_dgemm(Trans, Notrans, i, n, nb, neg_one, A+i, lda, d_x+i, m, one, b, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int mm = min(nb, m);
                magma_dgemm(Trans, Notrans, mm, n, mm, alpha, d_dinvA, nb, b, ldb, zero, d_x, m);

                if (nb >= m) {
                    dtrsm_copy();
                    magma_free( d_dinvA );
                    magma_free( d_x );
                    return;
                }

                magma_dgemm(Trans, Notrans, m-nb, n, nb, neg_one, A+(nb)*lda, lda, d_x, m, alpha, b+nb, ldb);

                /* the rest blocks */
                for( i=nb; i < m; i += nb ) {
                    mm = min(m-i, nb);
                    magma_dgemm(Trans, Notrans, mm, n, mm, one, d_dinvA+i*nb, nb, b+i, ldb, zero, d_x+i, m);

                    if (i+nb >= m)
                        break;

                    magma_dgemm(Trans, Notrans, m-i-nb, n, nb, neg_one, A+(i+nb)*lda+i, lda, d_x+i, m, one, b+i+nb, ldb);
                }
            }
        }
        else{
            /* the  transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int mm = (m % nb == 0) ? nb : (m % nb);
                i = m-mm;
                magma_dgemm(Conjtrans, Notrans, mm, n, mm, alpha, d_dinvA+i*nb, nb, b+i, ldb, zero, d_x+i, m);

                if (i-nb < 0) {
                    dtrsm_copy();
                    magma_free( d_dinvA );
                    magma_free( d_x );
                    return;
                }

                magma_dgemm(Conjtrans, Notrans, i, n, mm, neg_one, A+i, lda, d_x+i, m, alpha, b, ldb);

                /* the rest blocks */
                for( i=m-mm-nb; i >= 0; i -= nb ) {
                    magma_dgemm(Conjtrans, Notrans, nb, n, nb, one, d_dinvA+i*nb, nb, b+i, ldb, zero, d_x+i, m);

                    if (i-nb < 0)
                        break;

                    magma_dgemm(Conjtrans, Notrans, i, n, nb, neg_one, A+i, lda, d_x+i, m, one, b, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int mm = min(nb, m);
                magma_dgemm(Conjtrans, Notrans, mm, n, mm, alpha, d_dinvA, nb, b, ldb, zero, d_x, m);

                if (nb >= m) {
                    dtrsm_copy();
                    magma_free( d_dinvA );
                    magma_free( d_x );
                    return;
                }

                magma_dgemm(Conjtrans, Notrans, m-nb, n, nb, neg_one, A+(nb)*lda, lda, d_x, m, alpha, b+nb, ldb);

                /* the rest blocks */
                for( i=nb; i < m; i += nb ) {
                    mm = min(m-i, nb);
                    magma_dgemm(Conjtrans, Notrans, mm, n, mm, one, d_dinvA+i*nb, nb, b+i, ldb, zero, d_x+i, m);

                    if (i+nb >= m)
                        break;

                    magma_dgemm(Conjtrans, Notrans, m-i-nb, n, nb, neg_one, A+(i+nb)*lda+i, lda, d_x+i, m, one, b+i+nb, ldb);
                }
            }
        }
    }
    else {
        // side=R
        /* invert the diagonals
         * Allocate device memory for the inverted diagonal blocks, size=n*BLOCK_SIZE
         */
        magma_dmalloc( &d_dinvA, nb*((n/nb) + (n % nb != 0))*nb );
        magma_dmalloc( &d_x,     n*m );
        cudaMemset(d_x,     0, n*m*sizeof(double));
        cudaMemset(d_dinvA, 0, nb*((n/nb)+(n % nb != 0))*nb*sizeof(double));
        diag_dtrtri (n, uplo, diag, A, d_dinvA, lda);

        if (transA == MagmaNoTrans) {
            /* the non-transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int nn = (n % nb == 0) ? nb : (n % nb);
                i = n-nn;
                magma_dgemm(Notrans, Notrans, m, nn, nn, alpha, b+ldb*i, ldb, d_dinvA+i*nb, nb, zero, d_x+i*m, m);

                if (i-nb < 0) {
                    dtrsm_copy();
                    magma_free( d_x );
                    magma_free( d_dinvA );
                    return;
                }

                magma_dgemm(Notrans, Notrans, m, i, nn, neg_one, d_x+i*m, m, A+i, lda, alpha, b, ldb);

                /* the rest blocks */
                for( i=n-nn-nb; i >= 0; i -= nb ) {
                    magma_dgemm(Notrans, Notrans, m, nb, nb, one, b+ldb*i, ldb, d_dinvA+i*nb, nb, zero, d_x+i*m, m);

                    if (i-nb < 0)
                        break;

                    magma_dgemm(Notrans, Notrans, m, i, nb, neg_one, d_x+i*m, m, A+i, lda, one, b, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int nn = min(nb, n);
                magma_dgemm(Notrans, Notrans, m, nn, nn, alpha, b, ldb, d_dinvA, nb, zero, d_x, m);

                if (nb >= n) {
                    dtrsm_copy();
                    magma_free( d_x );
                    magma_free( d_dinvA );
                    return;
                }

                magma_dgemm(Notrans, Notrans, m, n-nb, nb, neg_one, d_x, m, A+nb*lda, lda, alpha, b+nb*ldb, ldb);

                /* the rest blocks */
                for( i=nb; i < n; i += nb ) {
                    nn = min(nb, n-i);
                    magma_dgemm(Notrans, Notrans, m, nn, nn, one, b+ldb*i, ldb, d_dinvA+i*nb, nb, zero, d_x+i*m, m);

                    if (i+nb >= n)
                        break;

                    magma_dgemm(Notrans, Notrans, m, n-i-nb, nb, neg_one, d_x+i*m, m,   A+(i+nb)*lda+i, lda, one, b+(i+nb)*ldb, ldb);
                }
            }
        }
        else if (transA == MagmaTrans) {
            /* the transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int nn = min(nb, n);
                magma_dgemm(Notrans, Trans, m, nn, nn, alpha, b, ldb, d_dinvA, nb, zero, d_x, m);

                if (nb >= n) {
                    dtrsm_copy();
                    magma_free( d_x );
                    magma_free( d_dinvA );
                    return;
                }

                magma_dgemm(Notrans, Trans, m, n-nb, nb, neg_one, d_x, m, A+nb, lda, alpha, b+nb*ldb, ldb);

                /* the rest blocks */
                for( i=nb; i < n; i += nb ) {
                    nn = min(nb, n-i);
                    magma_dgemm(Notrans, Trans, m, nn, nn, one, b+ldb*i, ldb, d_dinvA+i*nb, nb, zero, d_x+i*m, m);

                    if (i+nb >= n)
                        break;

                    magma_dgemm(Notrans, Trans, m, n-i-nb, nb, neg_one, d_x+i*m, m,   A+i*lda+nb+i, lda, one, b+(i+nb)*ldb, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int nn = (n % nb == 0) ? nb : (n % nb);
                i = n-nn;
                magma_dgemm(Notrans, Trans, m, nn, nn, alpha, b+ldb*i, ldb, d_dinvA+i*nb, nb, zero, d_x+i*m, m);

                if (i-nb < 0) {
                    dtrsm_copy();
                    magma_free( d_x );
                    magma_free( d_dinvA );
                    return;
                }

                magma_dgemm(Notrans, Trans, m, i, nn, neg_one, d_x+i*m, m, A+i*lda, lda, alpha, b, ldb);

                /* the rest blocks */
                for( i=n-nn-nb; i >= 0; i -= nb ) {
                    magma_dgemm(Notrans, Trans, m, nb, nb, one, b+ldb*i, ldb, d_dinvA+i*nb, nb, zero, d_x+i*m, m);

                    if (i-nb < 0)
                        break;

                    magma_dgemm(Notrans, Trans, m, i, nb, neg_one, d_x+i*m, m, A+i*lda, lda, one, b, ldb);
                }
            }
        }
        else{
            /* the Conj transpose case */
            if (uplo == MagmaLower) {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int nn = min(nb, n);
                magma_dgemm(Notrans, Conjtrans, m, nn, nn, alpha, b, ldb, d_dinvA, nb, zero, d_x, m);

                if (nb >= n) {
                    dtrsm_copy();
                    magma_free( d_x );
                    magma_free( d_dinvA );
                    return;
                }

                magma_dgemm(Notrans, Conjtrans, m, n-nb, nb, neg_one, d_x, m, A+nb, lda, alpha, b+nb*ldb, ldb);

                /* the rest blocks */
                for( i=nb; i < n; i += nb ) {
                    nn = min(nb, n-i);
                    magma_dgemm(Notrans, Conjtrans, m, nn, nn, one, b+ldb*i, ldb, d_dinvA+i*nb, nb, zero, d_x+i*m, m);

                    if (i+nb >= n)
                        break;

                    magma_dgemm(Notrans, Conjtrans, m, n-i-nb, nb, neg_one, d_x+i*m, m,   
                                                A+i*lda+nb+i, lda, one, b+(i+nb)*ldb, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int nn = (n % nb == 0) ? nb : (n % nb);
                i = n-nn;
                magma_dgemm(Notrans, Conjtrans, m, nn, nn, alpha, b+ldb*i, ldb, d_dinvA+i*nb, nb, zero, d_x+i*m, m);

                if (i-nb < 0) {
                    dtrsm_copy();
                    magma_free( d_x );
                    magma_free( d_dinvA );
                    return;
                }

                magma_dgemm(Notrans, Conjtrans, m, i, nn, neg_one, d_x+i*m, m, A+i*lda, lda, alpha, b, ldb);

                /* the rest blocks */
                for( i=n-nn-nb; i >= 0; i -= nb ) {
                    magma_dgemm(Notrans, Conjtrans, m, nb, nb, one, b+ldb*i, ldb, d_dinvA+i*nb, nb, zero, d_x+i*m, m);

                    if (i-nb < 0)
                        break;

                    magma_dgemm(Notrans, Conjtrans, m, i, nb, neg_one, d_x+i*m, m, A+i*lda, lda, one, b, ldb);
                }
            }
        }

    }

    dtrsm_copy();
    magma_free( d_dinvA );
    magma_free( d_x );
}
