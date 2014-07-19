/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @precisions normal z -> c d s

       @author Peng Du
       @author Tingxing Dong
*/
#include "common_magma.h"

#define BLOCK_SIZE 16 // inner blocking size, <=32
#define NB 128        // outer blocking size, >BLOCK_SIZE

__global__ void
ztrsm_copy_kernel(int m, int n, magmaDoubleComplex *dB, int lddb, magmaDoubleComplex *dX, int lddx)
{
    int by = blockIdx.y;
    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if (ind < m)
        dB[by*lddb + ind] = dX[by*lddx + ind];
}


#define MAX_THREAD_PER_BLOCK 512
#define WARP_SIZE 32


#define ztrsm_copy() \
    do { \
        dim3 threads( (m >= MAX_THREAD_PER_BLOCK) ? MAX_THREAD_PER_BLOCK : (WARP_SIZE*((m/WARP_SIZE)+(m % WARP_SIZE != 0))), 1 ); \
        dim3 grid( (m - 1)/threads.x + 1, n ); \
        ztrsm_copy_kernel<<< grid, threads, 0, magma_stream >>>(m, n, dB, lddb, dX, m); \
    } while(0)

// previously ztrsm_copy had sync -- there's no need; ztrsm should be async.
//        magma_device_sync(); \


/**
    Purpose
    -------
    ztrsm_work solves one of the matrix equations on gpu

        op(A)*X = alpha*B,   or   X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,   or   op(A) = A^T,  or  op(A) = A^H.

    The matrix X is overwritten on B.

    This is an asynchronous version of magmablas_ztrsm with flag,
    d_dinvA and dX workspaces as arguments.

    Arguments
    ----------
    @param[in]
    side    magma_side_t.
            On entry, side specifies whether op(A) appears on the left
            or right of X as follows:
      -     = MagmaLeft:       op(A)*X = alpha*B.
      -     = MagmaRight:      X*op(A) = alpha*B.

    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    transA  magma_trans_t.
            On entry, transA specifies the form of op(A) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op(A) = A.
      -     = MagmaTrans:      op(A) = A^T.
      -     = MagmaConjTrans:  op(A) = A^H.

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
    alpha   COMPLEX_16.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    dA      COMPLEX_16 array of DIMENSION ( ldda, k ), where k is m
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
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of A as declared
            in the calling (sub) program. When side = MagmaLeft then
            ldda must be at least max( 1, m ), when side = MagmaRight
            then ldda must be at least max( 1, n ).

    @param[in,out]
    dB      COMPLEX_16 array of DIMENSION ( lddb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B, and on exit is
            overwritten by the solution matrix X.

    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of B as declared
            in the calling (sub) program. lddb must be at least
            max( 1, m ).

    @param[in]
    flag    BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks are already inverted.

    @param
    d_dinvA (workspace) on device.
            If side == MagmaLeft,  d_dinvA must be of size >= ((m+NB-1)/NB)*NB*NB,
            If side == MagmaRight, d_dinvA must be of size >= ((n+NB-1)/NB)*NB*NB,
            where NB = 128.

    @param
    dX      (workspace) size m*n, on device.

    @param[in]
    stream  magma_queue_t
            Stream to execute in.

    @ingroup magma_zblas3
    ********************************************************************/
extern "C"
void magmablas_ztrsm_work(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex* dA, magma_int_t ldda,
    magmaDoubleComplex* dB, magma_int_t lddb,
    magma_int_t flag,
    magmaDoubleComplex* d_dinvA, magmaDoubleComplex *dX)
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dX(i_, j_) (dX + (i_) + (j_)*m)
    #define d_dinvA(i_) (d_dinvA + (i_)*NB)

    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;

    magma_int_t i, jb;
    magma_int_t nrowA = (side == MagmaLeft ? m : n);
    
    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != Magma_ConjTrans ) {
        info = -3;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -4;
    } else if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (ldda < max(1,nrowA)) {
        info = -9;
    } else if (lddb < max(1,m)) {
        info = -11;
    }
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (m == 0 || n == 0)
        return;

    if (side == MagmaLeft) {
        // invert diagonal blocks
        if (flag)
            magmablas_ztrtri_diag( uplo, diag, m, dA, ldda, d_dinvA );

        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // left, lower no-transpose
                // handle first block seperately with alpha
                jb = min(NB, m);
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(0), NB, dB, lddb, c_zero, dX, m );

                if (NB >= m) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-NB, n, NB, c_neg_one, dA(NB,0), ldda, dX, m, alpha, dB(NB,0), lddb );

                // remaining blocks
                for( i=NB; i < m; i += NB ) {
                    jb = min(m-i, NB);
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, jb, n, jb, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), m );

                    if (i+NB >= m)
                        break;
                    
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i-NB, n, NB, c_neg_one, dA(i+NB,i), ldda, dX(i,0), m, c_one, dB(i+NB,0), lddb );
                }
            }
            else {
                // left, upper no-transpose
                // handle first block seperately with alpha
                jb = (m % NB == 0) ? NB : (m % NB);
                i = m-jb;
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), m );

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaNoTrans, i, n, jb, c_neg_one, dA(0,i), ldda, dX(i,0), m, alpha, dB, lddb );

                // remaining blocks
                for( i=m-jb-NB; i >= 0; i -= NB ) {
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, NB, n, NB, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), m );

                    if (i-NB < 0)
                        break;

                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, i, n, NB, c_neg_one, dA(0,i), ldda, dX(i,0), m, c_one, dB, lddb );
                }
            }
        }
        else if( transA == MagmaTrans) {
            if (uplo == MagmaLower) {
                // left, lower transpose
                // handle first block seperately with alpha
                jb = (m % NB == 0) ? NB : (m % NB);
                i = m-jb;
                magma_zgemm( MagmaTrans, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), m );

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaTrans, MagmaNoTrans, i, n, jb, c_neg_one, dA(i,0), ldda, dX(i,0), m, alpha, dB, lddb );

                // remaining blocks
                for( i=m-jb-NB; i >= 0; i -= NB ) {
                    magma_zgemm( MagmaTrans, MagmaNoTrans, NB, n, NB, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), m );

                    if (i-NB < 0)
                        break;

                    magma_zgemm( MagmaTrans, MagmaNoTrans, i, n, NB, c_neg_one, dA(i,0), ldda, dX(i,0), m, c_one, dB, lddb );
                }
            }
            else {
                // left, upper transpose
                // handle first block seperately with alpha
                jb = min(NB, m);
                magma_zgemm( MagmaTrans, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(0), NB, dB, lddb, c_zero, dX, m );

                if (NB >= m) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaTrans, MagmaNoTrans, m-NB, n, NB, c_neg_one, dA(0,NB), ldda, dX, m, alpha, dB(NB,0), lddb );

                // remaining blocks
                for( i=NB; i < m; i += NB ) {
                    jb = min(m-i, NB);
                    magma_zgemm( MagmaTrans, MagmaNoTrans, jb, n, jb, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), m );

                    if (i+NB >= m)
                        break;

                    magma_zgemm( MagmaTrans, MagmaNoTrans, m-i-NB, n, NB, c_neg_one, dA(i,i+NB), ldda, dX(i,0), m, c_one, dB(i+NB,0), lddb );
                }
            }
        }
        else {  // transA == MagmaConjTras
            if (uplo == MagmaLower) {
                // left, lower conjugate-transpose
                // handle first block seperately with alpha
                jb = (m % NB == 0) ? NB : (m % NB);
                i = m-jb;
                magma_zgemm( MagmaConjTrans, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), m );

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaConjTrans, MagmaNoTrans, i, n, jb, c_neg_one, dA(i,0), ldda, dX(i,0), m, alpha, dB, lddb );

                // remaining blocks
                for( i=m-jb-NB; i >= 0; i -= NB ) {
                    magma_zgemm( MagmaConjTrans, MagmaNoTrans, NB, n, NB, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), m );

                    if (i-NB < 0)
                        break;

                    magma_zgemm( MagmaConjTrans, MagmaNoTrans, i, n, NB, c_neg_one, dA(i,0), ldda, dX(i,0), m, c_one, dB, lddb );
                }
            }
            else {
                // left, upper conjugate-transpose
                // handle first block seperately with alpha
                jb = min(NB, m);
                magma_zgemm( MagmaConjTrans, MagmaNoTrans, jb, n, jb, alpha, d_dinvA(0), NB, dB, lddb, c_zero, dX, m );

                if (NB >= m) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaConjTrans, MagmaNoTrans, m-NB, n, NB, c_neg_one, dA(0,NB), ldda, dX, m, alpha, dB(NB,0), lddb );

                // remaining blocks
                for( i=NB; i < m; i += NB ) {
                    jb = min(m-i, NB);
                    magma_zgemm( MagmaConjTrans, MagmaNoTrans, jb, n, jb, c_one, d_dinvA(i), NB, dB(i,0), lddb, c_zero, dX(i,0), m );

                    if (i+NB >= m)
                        break;

                    magma_zgemm( MagmaConjTrans, MagmaNoTrans, m-i-NB, n, NB, c_neg_one, dA(i,i+NB), ldda, dX(i,0), m, c_one, dB(i+NB,0), lddb );
                }
            }
        }
    }
    else {  // side == MagmaRight
        // invert diagonal blocks
        if (flag)
            magmablas_ztrtri_diag( uplo, diag, n, dA, ldda, d_dinvA );

        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // right, lower no-transpose
                // handle first block seperately with alpha
                int nn = (n % NB == 0) ? NB : (n % NB);
                i = n-nn;
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, nn, nn, alpha, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), m );

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, i, nn, c_neg_one, dX(0,i), m, dA(i,0), ldda, alpha, dB, lddb );

                // remaining blocks
                for( i=n-nn-NB; i >= 0; i -= NB ) {
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, NB, NB, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), m );

                    if (i-NB < 0)
                        break;

                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, i, NB, c_neg_one, dX(0,i), m, dA(i,0), ldda, c_one, dB, lddb );
                }
            }
            else {
                // right, upper no-transpose
                // handle first block seperately with alpha
                int nn = min(NB, n);
                magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, nn, nn, alpha, dB, lddb, d_dinvA(0), NB, c_zero, dX, m );

                if (NB >= n) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, n-NB, NB, c_neg_one, dX, m, dA(0,NB), ldda, alpha, dB(0,NB), lddb );

                // remaining blocks
                for( i=NB; i < n; i += NB ) {
                    nn = min(NB, n-i);
                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, nn, nn, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), m );

                    if (i+NB >= n)
                        break;

                    magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, n-i-NB, NB, c_neg_one, dX(0,i), m, dA(i,i+NB), ldda, c_one, dB(0,i+NB), lddb );
                }
            }
        }
        else if (transA == MagmaTrans) {
            if (uplo == MagmaLower) {
                // right, lower transpose
                // handle first block seperately with alpha
                int nn = min(NB, n);
                magma_zgemm( MagmaNoTrans, MagmaTrans, m, nn, nn, alpha, dB, lddb, d_dinvA(0), NB, c_zero, dX, m );

                if (NB >= n) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaTrans, m, n-NB, NB, c_neg_one, dX, m, dA(NB,0), ldda, alpha, dB(0,NB), lddb );

                // remaining blocks
                for( i=NB; i < n; i += NB ) {
                    nn = min(NB, n-i);
                    magma_zgemm( MagmaNoTrans, MagmaTrans, m, nn, nn, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), m );

                    if (i+NB >= n)
                        break;

                    magma_zgemm( MagmaNoTrans, MagmaTrans, m, n-i-NB, NB, c_neg_one, dX(0,i), m, dA(NB+i,i), ldda, c_one, dB(0,i+NB), lddb );
                }
            }
            else {
                // right, upper transpose
                // handle first block seperately with alpha
                int nn = (n % NB == 0) ? NB : (n % NB);
                i = n-nn;
                magma_zgemm( MagmaNoTrans, MagmaTrans, m, nn, nn, alpha, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), m );

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaTrans, m, i, nn, c_neg_one, dX(0,i), m, dA(0,i), ldda, alpha, dB, lddb );

                // remaining blocks
                for( i=n-nn-NB; i >= 0; i -= NB ) {
                    magma_zgemm( MagmaNoTrans, MagmaTrans, m, NB, NB, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), m );

                    if (i-NB < 0)
                        break;

                    magma_zgemm( MagmaNoTrans, MagmaTrans, m, i, NB, c_neg_one, dX(0,i), m, dA(0,i), ldda, c_one, dB, lddb );
                }
            }
        }
        else {  // TransA == MagmaConjTrans
            if (uplo == MagmaLower) {
                // right, lower conjugate-transpose
                // handle first block seperately with alpha
                int nn = min(NB, n);
                magma_zgemm( MagmaNoTrans, MagmaConjTrans, m, nn, nn, alpha, dB, lddb, d_dinvA(0), NB, c_zero, dX, m );

                if (NB >= n) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaConjTrans, m, n-NB, NB, c_neg_one, dX, m, dA(NB,0), ldda, alpha, dB(0,NB), lddb );

                // remaining blocks
                for( i=NB; i < n; i += NB ) {
                    nn = min(NB, n-i);
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans, m, nn, nn, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), m );

                    if (i+NB >= n)
                        break;

                    magma_zgemm(MagmaNoTrans, MagmaConjTrans, m, n-i-NB, NB, c_neg_one, dX(0,i), m,
                                                dA(NB+i,i), ldda, c_one, dB(0,i+NB), lddb);
                }
            }
            else {
                // right, upper conjugate-transpose
                // handle first block seperately with alpha
                int nn = (n % NB == 0) ? NB : (n % NB);
                i = n-nn;
                magma_zgemm( MagmaNoTrans, MagmaConjTrans, m, nn, nn, alpha, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), m );

                if (i-NB < 0) {
                    ztrsm_copy();
                    return;
                }

                magma_zgemm( MagmaNoTrans, MagmaConjTrans, m, i, nn, c_neg_one, dX(0,i), m, dA(0,i), ldda, alpha, dB, lddb );

                // remaining blocks
                for( i=n-nn-NB; i >= 0; i -= NB ) {
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans, m, NB, NB, c_one, dB(0,i), lddb, d_dinvA(i), NB, c_zero, dX(0,i), m );

                    if (i-NB < 0)
                        break;

                    magma_zgemm( MagmaNoTrans, MagmaConjTrans, m, i, NB, c_neg_one, dX(0,i), m, dA(0,i), ldda, c_one, dB, lddb );
                }
            }
        }
    }

    ztrsm_copy();
}


/**
    @see magmablas_ztrsm_work
    @ingroup magma_zblas3
    ********************************************************************/
extern "C"
void magmablas_ztrsm(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex* dA, magma_int_t ldda,
    magmaDoubleComplex* dB, magma_int_t lddb )
{
    magma_int_t nrowA = (side == MagmaLeft ? m : n);
    
    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != Magma_ConjTrans ) {
        info = -3;
    } else if ( diag != MagmaUnit && diag != MagmaNonUnit ) {
        info = -4;
    } else if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (ldda < max(1,nrowA)) {
        info = -9;
    } else if (lddb < max(1,m)) {
        info = -11;
    }
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    magmaDoubleComplex *d_dinvA, *dX;
    magma_int_t size_dinvA;
    magma_int_t size_x = m*n;
    if ( side == MagmaLeft ) {
        size_dinvA = ((m+NB-1)/NB)*NB*NB;
    }
    else {
        size_dinvA = ((n+NB-1)/NB)*NB*NB;
    }
    
    magma_zmalloc( &d_dinvA, size_dinvA );
    magma_zmalloc( &dX,     size_x    );
    if ( d_dinvA == NULL || dX == NULL ) {
        info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        goto cleanup;
    }
    
    magmablas_ztrsm_work( side, uplo, transA, diag, m, n, alpha,
                          dA, ldda, dB, lddb, 1, d_dinvA, dX );
    
cleanup:
    magma_free( d_dinvA );
    magma_free( dX );
}
