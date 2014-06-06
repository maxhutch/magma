/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:45 2013

       @author Peng Du
*/
#include "common_magma.h"

#define qmod(a, b) ((a)-(__mul24((b), (a)/(b))))

#define b_copy() \
    do { \
        dim3 dimBlock( (M>=MAX_THREAD_PER_BLOCK) ? MAX_THREAD_PER_BLOCK : (WARP_SIZE*((M/WARP_SIZE)+(M%WARP_SIZE!=0))), 1 ); \
        dim3 dimGrid( (M - 1)/dimBlock.x + 1, N ); \
        b_copy_kernel<<< dimGrid, dimBlock, 0, magma_stream >>>(M, N, b, ldb, d_x, M); \
    } while(0)
    // no magma_device_sync -- async function

#define MAX_THREAD_PER_BLOCK 512
#define WARP_SIZE 32

#define BLOCK_SIZE 16 // inner blocking size, <=32
#define NB 128        // outer blocking size, >BLOCK_SIZE

__global__ void
b_copy_kernel(int M, int N, float *b, int ldb, float *d_x, int ldx);

extern "C"
void diag_strtri(magma_int_t M, char uplo, char diag, const float *A, float *d_dinvA, magma_int_t lda);

/*
 * magmablas_strsm
 */
extern "C"
void magmablas_strsm_work(
    char side, char uplo, char tran, char diag, magma_int_t M, magma_int_t N,
    float alpha,
    const float* A, magma_int_t lda,
    float* b, magma_int_t ldb,
    int flag,
    float *d_dinvA, float *d_x )
{
/*  -- MAGMA (version 1.4.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    December 2013

    Purpose
    =======

    strsm solves one of the matrix equations on gpu

        op( A )*x = alpha*b,   or   x*op( A ) = alpha*b,

    where alpha is a scalar, x and b are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op( A ) is one of

        op( A ) = A   or   op( A ) = A^T.

    The matrix X is overwritten on B.

    When M or N is not a multiple of blocking size, which is 32 for now, cublasStrsm will
    be called instead. There soon will not be this limitation both for arbitrary problem
    size and blocking size.

    This is an asynchronous version of magmablas_strsm with "workspace" as an argument.

    Arguments
    ==========

    side    CHARACTER*1.
            On entry, side specifies whether op( A ) appears on the left
            or right of X as follows:

               side = 'L' or 'l'   op( A )*X = alpha*B.

               side = 'R' or 'r'   X*op( A ) = alpha*B.

            Unchanged on exit.

    uplo    CHARACTER*1.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:

               uplo = 'U' or 'u'   A is an upper triangular matrix.

               uplo = 'L' or 'l'   A is a lower triangular matrix.

            Unchanged on exit.

    tran    CHARACTER*1.
            On entry, tran specifies the form of op( A ) to be used in
            the matrix multiplication as follows:

               tran = 'N' or 'n'   op( A ) = A.

               tran = 'T' or 't'   op( A ) = A^T.

               tran = 'C' or 'c'   op( A ) = A^T.

            Unchanged on exit.

    diag    CHARACTER*1.
            On entry, diag specifies whether or not A is unit triangular
            as follows:

               diag = 'U' or 'u'   A is assumed to be unit triangular.

               diag = 'N' or 'n'   A is not assumed to be unit triangular.

            Unchanged on exit.

    m       INTEGER.
            On entry, m specifies the number of rows of B. m must be at
            least zero.
            Unchanged on exit.

    n       INTEGER.
            On entry, n specifies the number of columns of B. n must be
            at least zero.
            Unchanged on exit.

    alpha   REAL.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.
            Unchanged on exit.

    A       REAL array of DIMENSION ( lda, k ), where k is m
            when side = 'L' or 'l' and is n when side = 'R' or 'r'.
            Before entry with uplo = 'U' or 'u', the leading k by k
            upper triangular part of the array A must contain the upper
            triangular matrix and the strictly lower triangular part of
            A is not referenced.
            Before entry with uplo = 'L' or 'l', the leading k by k
            lower triangular part of the array A must contain the lower
            triangular matrix and the strictly upper triangular part of
            A is not referenced.
            Note that when diag = 'U' or 'u', the diagonal elements of
            A are not referenced either, but are assumed to be unity.
            Unchanged on exit.

    lda     INTEGER.
            On entry, lda specifies the first dimension of A as declared
            in the calling (sub) program. When side = 'L' or 'l' then
            lda must be at least max( 1, m ), when side = 'R' or 'r'
            then lda must be at least max( 1, n ).
            Unchanged on exit.

    b       REAL array of DIMENSION ( ldb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B, and on exit is
            overwritten by the solution matrix X.

    ldb     INTEGER.
            On entry, ldb specifies the first dimension of B as declared
            in the calling (sub) program. ldb must be at least
            max( 1, m ).
            Unchanged on exit.

    flag    BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks are already inverted. (?)

    d_dinvA workspace of size NB*((M+NB-1)/NB))*NB, on device.

    d_x     workspace of size N*M, on device.

    Level 3 Blas routine.
    ===================================================================== */

    int i;

    /* quick return on wrong size */
    if (M <= 0 || N <= 0)
        return;

    if (side == 'l' || side == 'L') {
        // side=L
        /* invert the diagonals */
        if (flag == 1) {
            diag_strtri (M, uplo, diag, A, d_dinvA, lda);
        }

        if (tran == 'N' || tran == 'n') {
            /* the non-transpose case */
            if (uplo == 'L' || uplo == 'l') {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int MM = min (NB, M);
                cublasSgemm ('N', 'N', MM, N, MM, alpha, d_dinvA, NB, b, ldb, 0, d_x, M);

                if (NB >= M) {
                    b_copy();
                    return;
                }

                cublasSgemm ('N', 'N', M-NB, N, NB, -1.0, A+NB, lda, d_x, M, alpha, b+NB, ldb);

                /* the rest blocks */
                for (i=NB; i < M; i += NB) {
                    MM = min (M-i, NB);
                    cublasSgemm ('N', 'N', MM, N, MM, 1.0, d_dinvA+i*NB, NB, b+i, ldb, 0, d_x+i, M);

                    if (i+NB >= M)
                        break;

                    cublasSgemm ('N', 'N', M-i-NB, N, NB, -1.0, A+i*lda+i+NB, lda, d_x+i, M, 1.0, b+i+NB, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int MM = (M%NB==0) ? NB : (M%NB);
                i = M-MM;
                cublasSgemm ('N', 'N', MM, N, MM, alpha, d_dinvA+i*NB, NB, b+i, ldb, 0.0, d_x+i, M);

                if (i-NB < 0) {
                    b_copy();
                    return;
                }

                cublasSgemm ('N', 'N', i, N, MM, -1.0, A+i*lda, lda, d_x+i, M, alpha, b, ldb);

                /* the rest blocks */
                for (i=M-MM-NB; i >= 0; i -= NB) {
                    cublasSgemm ('N', 'N', NB, N, NB, 1.0, d_dinvA+i*NB, NB, b+i, ldb, 0.0, d_x+i, M);

                    if (i-NB < 0)
                        break;

                    cublasSgemm ('N', 'N', i, N, NB, -1.0, A+i*lda, lda, d_x+i, M, 1.0, b, ldb);
                }
            }
        }
        else {
            /* the transpose case */
            if (uplo == 'L' || uplo == 'l') {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int MM = (M%NB==0) ? NB : (M%NB);
                i = M-MM;
                cublasSgemm ('T', 'N', MM, N, MM, alpha, d_dinvA+i*NB, NB, b+i, ldb, 0, d_x+i, M);

                if (i-NB < 0) {
                    b_copy();
                    return;
                }

                cublasSgemm ('T', 'N', i, N, MM, -1.0, A+i, lda, d_x+i, M, alpha, b, ldb);

                /* the rest blocks */
                for (i=M-MM-NB; i >= 0; i -= NB) {
                    cublasSgemm ('T', 'N', NB, N, NB, 1.0, d_dinvA+i*NB, NB, b+i, ldb, 0, d_x+i, M);

                    if (i-NB < 0)
                        break;

                    cublasSgemm ('T', 'N', i, N, NB, -1.0, A+i, lda, d_x+i, M, 1.0, b, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int MM = min (NB, M);
                cublasSgemm ('T', 'N', MM, N, MM, alpha, d_dinvA, NB, b, ldb, 0, d_x, M);

                if (NB >= M) {
                    b_copy();
                    return;
                }

                cublasSgemm ('T', 'N', M-NB, N, NB, -1.0, A+(NB)*lda, lda, d_x, M, alpha, b+NB, ldb);

                /* the rest blocks */
                for (i=NB; i < M; i += NB) {
                    MM = min (M-i, NB);
                    cublasSgemm ('T', 'N', MM, N, MM, 1.0, d_dinvA+i*NB, NB, b+i, ldb, 0, d_x+i, M);

                    if (i+NB >= M)
                        break;

                    cublasSgemm ('T', 'N', M-i-NB, N, NB, -1.0, A+(i+NB)*lda+i, lda, d_x+i, M, 1.0, b+i+NB, ldb);
                }
            }
        }
    }
    else {
        // side=R
        /* invert the diagonals */
        if (flag == 1) {
            diag_strtri (N, uplo, diag, A, d_dinvA, lda);
        }

        if (tran == 'N' || tran == 'n') {
            /* the non-transpose case */
            if (uplo == 'L' || uplo == 'l') {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int NN = (N%NB==0) ? NB : (N%NB);
                i = N-NN;
                cublasSgemm ('N', 'N', M, NN, NN, alpha, b+ldb*i, ldb, d_dinvA+i*NB, NB, 0.0, d_x+i*M, M);

                if (i-NB < 0) {
                    b_copy();
                    return;
                }

                cublasSgemm ('N', 'N', M, i, NN, -1.0, d_x+i*M, M, A+i, lda, alpha, b, ldb);

                /* the rest blocks */
                for (i=N-NN-NB; i >= 0; i -= NB) {
                    cublasSgemm ('N', 'N', M, NB, NB, 1.0, b+ldb*i, ldb, d_dinvA+i*NB, NB, 0.0, d_x+i*M, M);

                    if (i-NB < 0)
                        break;

                    cublasSgemm ('N', 'N', M, i, NB, -1.0, d_x+i*M, M, A+i, lda, 1.0, b, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int NN = min(NB, N);
                cublasSgemm ('N', 'N', M, NN, NN, alpha, b, ldb, d_dinvA, NB, 0, d_x, M);

                if (NB >= N) {
                    b_copy();
                    return;
                }

                cublasSgemm ('N', 'N', M, N-NB, NB, -1.0, d_x, M, A+NB*lda, lda, alpha, b+NB*ldb, ldb);

                /* the rest blocks */
                for (i=NB; i < N; i += NB) {
                    NN = min(NB, N-i);
                    cublasSgemm ('N', 'N', M, NN, NN, 1.0, b+ldb*i, ldb, d_dinvA+i*NB, NB, 0, d_x+i*M, M);

                    if (i+NB >= N)
                        break;

                    cublasSgemm ('N', 'N', M, N-i-NB, NB, -1.0, d_x+i*M, M,   A+(i+NB)*lda+i, lda, 1.0, b+(i+NB)*ldb, ldb);
                }
            }
        }
        else {
            /* the transpose case */
            if (uplo == 'L' || uplo == 'l') {
                /* the lower case */
                /* handle the first block seperately with alpha */
                int NN = min(NB, N);
                cublasSgemm ('N', 'T', M, NN, NN, alpha, b, ldb, d_dinvA, NB, 0, d_x, M);
                
                if (NB >= N) {
                    b_copy();
                    return;
                }

                cublasSgemm ('N', 'T', M, N-NB, NB, -1.0, d_x, M, A+NB, lda, alpha, b+NB*ldb, ldb);

                /* the rest blocks */
                for (i=NB; i < N; i += NB) {
                    NN = min(NB, N-i);
                    cublasSgemm ('N', 'T', M, NN, NN, 1.0, b+ldb*i, ldb, d_dinvA+i*NB, NB, 0, d_x+i*M, M);

                    if (i+NB >= N)
                        break;

                    cublasSgemm ('N', 'T', M, N-i-NB, NB, -1.0, d_x+i*M, M,   A+i*lda+NB+i, lda, 1.0, b+(i+NB)*ldb, ldb);
                }
            }
            else {
                /* the upper case */
                /* handle the first block seperately with alpha */
                int NN = (N%NB==0) ? NB : (N%NB);
                i = N-NN;
                cublasSgemm ('N', 'T', M, NN, NN, alpha, b+ldb*i, ldb, d_dinvA+i*NB, NB, 0.0, d_x+i*M, M);

                if (i-NB < 0) {
                    b_copy();
                    return;
                }

                cublasSgemm ('N', 'T', M, i, NN, -1.0, d_x+i*M, M, A+i*lda, lda, alpha, b, ldb);

                /* the rest blocks */
                for (i=N-NN-NB; i >= 0; i -= NB) {
                    cublasSgemm ('N', 'T', M, NB, NB, 1.0, b+ldb*i, ldb, d_dinvA+i*NB, NB, 0.0, d_x+i*M, M);

                    if (i-NB < 0)
                        break;

                    cublasSgemm ('N', 'T', M, i, NB, -1.0, d_x+i*M, M, A+i*lda, lda, 1.0, b, ldb);
                }
            }
        }
    }

    b_copy();
}
