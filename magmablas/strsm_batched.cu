/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from ztrsm_batched.cu normal z -> s, Fri Jan 30 19:00:10 2015

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
*/
#include "common_magma.h"
#include "batched_kernel_param.h"
/**
    Purpose
    -------
    strsm_work solves one of the matrix equations on gpu

        op(A)*X = alpha*B,   or   X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,   or   op(A) = A^T,  or  op(A) = A^H.

    The matrix X is overwritten on B.

    This is an asynchronous version of magmablas_strsm with flag,
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
            On entry, m specifies the number of rows of B. m >= 0.

    @param[in]
    n       INTEGER.
            On entry, n specifies the number of columns of B. n >= 0.

    @param[in]
    alpha   REAL.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    dA      REAL array of dimension ( ldda, k ), where k is m
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
            On entry, ldda specifies the first dimension of A.
            When side = MagmaLeft,  ldda >= max( 1, m ),
            when side = MagmaRight, ldda >= max( 1, n ).

    @param[in]
    dB      REAL array of dimension ( lddb, n ).
            Before entry, the leading m by n part of the array B must
            contain the right-hand side matrix B.

    @param[in]
    lddb    INTEGER.
            On entry, lddb specifies the first dimension of B.
            lddb >= max( 1, m ).

    @param[in]
    flag    BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks (stored in d_dinvA) are already inverted.

    @param
    d_dinvA (workspace) on device.
            If side == MagmaLeft,  d_dinvA must be of size >= ((m+TRI_NB-1)/TRI_NB)*TRI_NB*TRI_NB,
            If side == MagmaRight, d_dinvA must be of size >= ((n+TRI_NB-1)/TRI_NB)*TRI_NB*TRI_NB,
            where TRI_NB = 128.

    @param[out]
    dX      REAL array of dimension ( lddx, n ).
            On exit it contain the solution matrix X.

    @ingroup magma_sblas3
    ********************************************************************/
extern "C"
void magmablas_strsm_outofplace_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    float alpha, 
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t lddb,
    float** dX_array,    magma_int_t lddx, 
    float** dinvA_array, magma_int_t dinvA_length,
    float** dA_displ, float** dB_displ, 
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue)
{
/*
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dB(i_, j_) (dB + (i_) + (j_)*lddb)
    #define dX(i_, j_) (dX + (i_) + (j_)*m)
    #define d_dinvA(i_) (d_dinvA + (i_)*TRI_NB)

*/
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const float c_one     = MAGMA_S_ONE;
    const float c_zero    = MAGMA_S_ZERO;

    magma_int_t i, jb;
    magma_int_t nrowA = (side == MagmaLeft ? m : n);

    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
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
    magma_int_t size_dinvA;
    if ( side == MagmaLeft ) {
        size_dinvA = ((m+TRI_NB-1)/TRI_NB)*TRI_NB*TRI_NB;
    }
    else {
        size_dinvA = ((n+TRI_NB-1)/TRI_NB)*TRI_NB*TRI_NB;
    }
    if(dinvA_length < size_dinvA) info = -19;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }

    // quick return if possible.
    if (m == 0 || n == 0)
        return;

    magma_sdisplace_pointers(dA_displ,       dA_array,    ldda,    0, 0, batchCount, queue); 
    magma_sdisplace_pointers(dB_displ,       dB_array,    lddb,    0, 0, batchCount, queue); 
    magma_sdisplace_pointers(dX_displ,       dX_array,    lddx,    0, 0, batchCount, queue); 
    magma_sdisplace_pointers(dinvA_displ, dinvA_array,  TRI_NB,    0, 0, batchCount, queue); 

    if (side == MagmaLeft) {
        // invert diagonal blocks
        if (flag)
            magmablas_strtri_diag_batched( uplo, diag, m, dA_displ, ldda, dinvA_displ, resetozero, batchCount, queue );

        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // left, lower no-transpose
                // handle first block seperately with alpha
                jb = min(TRI_NB, m);                
                magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, jb, n, jb, alpha, dinvA_displ, TRI_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );

                if (TRI_NB < m) {
                    magma_sdisplace_pointers(dA_displ,    dA_array, ldda, TRI_NB, 0, batchCount, queue); 
                    magma_sdisplace_pointers(dB_displ,    dB_array, lddb, TRI_NB, 0, batchCount, queue);                    
                    magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m-TRI_NB, n, TRI_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=TRI_NB; i < m; i += TRI_NB ) {
                        jb = min(m-i, TRI_NB);
                        magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dB_displ,       dB_array, lddb,    i, 0, batchCount, queue);
                        magma_sdisplace_pointers(dX_displ,       dX_array, lddx,    i, 0, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, jb, n, jb, c_one, dinvA_displ, TRI_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i+TRI_NB >= m)
                            break;
                        
                        magma_sdisplace_pointers(dA_displ,    dA_array,    ldda,  i+TRI_NB, i, batchCount, queue); 
                        magma_sdisplace_pointers(dB_displ,    dB_array,    lddb,  i+TRI_NB, 0, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m-i-TRI_NB, n, TRI_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
            else {
                // left, upper no-transpose
                // handle first block seperately with alpha
                jb = (m % TRI_NB == 0) ? TRI_NB : (m % TRI_NB);
                i = m-jb;
                magma_sdisplace_pointers(dinvA_displ,    dinvA_array, TRI_NB, 0, i, batchCount, queue);
                magma_sdisplace_pointers(dB_displ,          dB_array,    lddb, i, 0, batchCount, queue);
                magma_sdisplace_pointers(dX_displ,          dX_array,    lddx, i, 0, batchCount, queue);
                magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, jb, n, jb, alpha, dinvA_displ, TRI_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );

                if (i-TRI_NB >= 0) {
                    magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, 0, i, batchCount, queue); 
                    magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                    magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, i, n, jb, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=m-jb-TRI_NB; i >= 0; i -= TRI_NB ) {
                        magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dB_displ,       dB_array,    lddb, i, 0, batchCount, queue);
                        magma_sdisplace_pointers(dX_displ,       dX_array,    lddx, i, 0, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, TRI_NB, n, TRI_NB, c_one, dinvA_displ, TRI_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i-TRI_NB < 0)
                            break;

                        magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, 0, i, batchCount, queue); 
                        magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, i, n, TRI_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, batchCount, queue);
                    }
                }
            }
        }
        else {  // transA == MagmaTrans || transA == MagmaConjTrans
            if (uplo == MagmaLower) {
                // left, lower transpose
                // handle first block seperately with alpha
                jb = (m % TRI_NB == 0) ? TRI_NB : (m % TRI_NB);
                i = m-jb;
                magma_sdisplace_pointers(dinvA_displ,    dinvA_array, TRI_NB, 0, i, batchCount, queue);
                magma_sdisplace_pointers(dB_displ,          dB_array,    lddb, i, 0, batchCount, queue);
                magma_sdisplace_pointers(dX_displ,          dX_array,    lddx, i, 0, batchCount, queue);
                magmablas_sgemm_batched( transA, MagmaNoTrans, jb, n, jb, alpha, dinvA_displ, TRI_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );

                if (i-TRI_NB >= 0) {
                    magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, i, 0, batchCount, queue); 
                    magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                    magmablas_sgemm_batched( transA, MagmaNoTrans, i, n, jb, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=m-jb-TRI_NB; i >= 0; i -= TRI_NB ) {
                        magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dX_displ,       dX_array,    lddx, i, 0, batchCount, queue);
                        magma_sdisplace_pointers(dB_displ,       dB_array,    lddb, i, 0, batchCount, queue);
                        magmablas_sgemm_batched( transA, MagmaNoTrans, TRI_NB, n, TRI_NB, c_one, dinvA_displ, TRI_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i-TRI_NB < 0)
                            break;

                        magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, i, 0,  batchCount, queue); 
                        magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                        magmablas_sgemm_batched( transA, MagmaNoTrans, i, n, TRI_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
            else {
                // left, upper transpose
                // handle first block seperately with alpha
                jb = min(TRI_NB, m);
                magmablas_sgemm_batched( transA, MagmaNoTrans, jb, n, jb, alpha, dinvA_displ, TRI_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );

                if (TRI_NB < m) {
                    magma_sdisplace_pointers(dA_displ,    dA_array,    ldda,      0,   TRI_NB, batchCount, queue); 
                    magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, TRI_NB,        0, batchCount, queue);
                    magmablas_sgemm_batched( transA, MagmaNoTrans, m-TRI_NB, n, TRI_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=TRI_NB; i < m; i += TRI_NB ) {
                        jb = min(m-i, TRI_NB);
                        magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dB_displ,       dB_array,    lddb, i, 0, batchCount, queue);
                        magma_sdisplace_pointers(dX_displ,       dX_array,    lddx, i, 0, batchCount, queue);
                        magmablas_sgemm_batched( transA, MagmaNoTrans, jb, n, jb, c_one, dinvA_displ, TRI_NB, dB_displ, lddb, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i+TRI_NB >= m)
                            break;

                        magma_sdisplace_pointers(dA_displ,    dA_array,    ldda,  i,        i+TRI_NB, batchCount, queue); 
                        magma_sdisplace_pointers(dB_displ,    dB_array,    lddb,  i+TRI_NB,        0, batchCount, queue);
                        magmablas_sgemm_batched( transA, MagmaNoTrans, m-i-TRI_NB, n, TRI_NB, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
        }
    }
    else {  // side == MagmaRight
        // invert diagonal blocks
        if (flag)
            magmablas_strtri_diag_batched( uplo, diag, n, dA_displ, ldda, dinvA_displ, resetozero, batchCount, queue);

        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // right, lower no-transpose
                // handle first block seperately with alpha
                jb = (n % TRI_NB == 0) ? TRI_NB : (n % TRI_NB);
                i = n-jb;                
                magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                magma_sdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                magma_sdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m, jb, jb, alpha, dB_displ, lddb, dinvA_displ, TRI_NB, c_zero, dX_displ, lddx, batchCount, queue );

                if (i-TRI_NB >= 0) {
                    magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, i, 0, batchCount, queue);                        
                    magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                    magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m, i, jb, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=n-jb-TRI_NB; i >= 0; i -= TRI_NB ) {
                        magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m, TRI_NB, TRI_NB, c_one, dB_displ, lddb, dinvA_displ, TRI_NB, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i-TRI_NB < 0)
                            break;

                        magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, i, 0, batchCount, queue); 
                        magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m, i, TRI_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
            else {
                // right, upper no-transpose
                // handle first block seperately with alpha
                jb = min(TRI_NB, n);
                magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m, jb, jb, alpha, dB_displ, lddb, dinvA_displ, TRI_NB, c_zero, dX_displ, lddx, batchCount, queue );
                if (TRI_NB < n) {
                    magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, 0, TRI_NB, batchCount, queue); 
                    magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, TRI_NB, batchCount, queue);
                    magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m, n-TRI_NB, TRI_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=TRI_NB; i < n; i += TRI_NB ) {
                        jb = min(TRI_NB, n-i);
                        magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m, jb, jb, c_one, dB_displ, lddb, dinvA_displ, TRI_NB, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i+TRI_NB >= n)
                            break;

                        magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, i, i+TRI_NB, batchCount, queue); 
                        magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, i+TRI_NB, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, MagmaNoTrans, m, n-i-TRI_NB, TRI_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
        }
        else { // transA == MagmaTrans || transA == MagmaConjTrans
            if (uplo == MagmaLower) {
                // right, lower transpose
                // handle first block seperately with alpha
                jb = min(TRI_NB, n);
                magmablas_sgemm_batched( MagmaNoTrans, transA, m, jb, jb, alpha, dB_displ, lddb, dinvA_displ, TRI_NB, c_zero, dX_displ, lddx, batchCount, queue );
                if (TRI_NB < n) {
                    magma_sdisplace_pointers(dA_displ,    dA_array,    ldda,  TRI_NB,      0, batchCount, queue); 
                    magma_sdisplace_pointers(dB_displ,    dB_array,    lddb,       0, TRI_NB, batchCount, queue);
                    magmablas_sgemm_batched( MagmaNoTrans, transA, m, n-TRI_NB, TRI_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=TRI_NB; i < n; i += TRI_NB ) {
                        jb = min(TRI_NB, n-i);
                        magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, transA, m, jb, jb, c_one, dB_displ, lddb, dinvA_displ, TRI_NB, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i+TRI_NB >= n)
                            break;

                        magma_sdisplace_pointers(dA_displ,    dA_array,    ldda,  TRI_NB+i,        i, batchCount, queue); 
                        magma_sdisplace_pointers(dB_displ,    dB_array,    lddb,         0, i+TRI_NB, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, transA, m, n-i-TRI_NB, TRI_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
            else {
                // right, upper transpose
                // handle first block seperately with alpha
                jb = (n % TRI_NB == 0) ? TRI_NB : (n % TRI_NB);
                i = n-jb;
                magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                magma_sdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                magma_sdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                magmablas_sgemm_batched( MagmaNoTrans, transA, m, jb, jb, alpha, dB_displ, lddb, dinvA_displ, TRI_NB, c_zero, dX_displ, lddx, batchCount, queue );

                if (i-TRI_NB >= 0) {
                    magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, 0, i, batchCount, queue); 
                    magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                    magmablas_sgemm_batched( MagmaNoTrans, transA, m, i, jb, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, batchCount, queue );

                    // remaining blocks
                    for( i=n-jb-TRI_NB; i >= 0; i -= TRI_NB ) {
                        magma_sdisplace_pointers(dinvA_displ, dinvA_array, TRI_NB, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dB_displ,       dB_array,    lddb, 0, i, batchCount, queue);
                        magma_sdisplace_pointers(dX_displ,       dX_array,    lddx, 0, i, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, transA, m, TRI_NB, TRI_NB, c_one, dB_displ, lddb, dinvA_displ, TRI_NB, c_zero, dX_displ, lddx, batchCount, queue );
                        if (i-TRI_NB < 0)
                            break;

                        magma_sdisplace_pointers(dA_displ,    dA_array,    ldda, 0, i, batchCount, queue); 
                        magma_sdisplace_pointers(dB_displ,    dB_array,    lddb, 0, 0, batchCount, queue);
                        magmablas_sgemm_batched( MagmaNoTrans, transA, m, i, TRI_NB, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, batchCount, queue );
                    }
                }
            }
        }
    }
}

/**
    @see magmablas_strsm_outofplace_batched
    @ingroup magma_sblas3
    ********************************************************************/
extern "C"
void magmablas_strsm_work_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, magma_int_t m, magma_int_t n, 
    float alpha, 
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t lddb,
    float** dX_array,    magma_int_t lddx, 
    float** dinvA_array, magma_int_t dinvA_length,
    float** dA_displ, float** dB_displ, 
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t nrowA = (side == MagmaLeft ? m : n);

    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
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



    magmablas_strsm_outofplace_batched( 
                    side, uplo, transA, diag, flag,
                    m, n, alpha,
                    dA_array,    ldda,
                    dB_array,    lddb,
                    dX_array,    lddx, 
                    dinvA_array, dinvA_length,
                    dA_displ, dB_displ, 
                    dX_displ, dinvA_displ,
                    resetozero, batchCount, queue );
    // copy X to B
    magma_sdisplace_pointers(dX_displ,    dX_array, lddx, 0, 0, batchCount, queue);
    magma_sdisplace_pointers(dB_displ,    dB_array, lddb, 0, 0, batchCount, queue);
    magmablas_slacpy_batched( MagmaFull, m, n, dX_displ, lddx, dB_displ, lddb, batchCount, queue );
}

/**
    @see magmablas_strsm_work
    @ingroup magma_sblas3
    ********************************************************************/
extern "C"
void magmablas_strsm_batched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    float alpha,
    float** dA_array,    magma_int_t ldda,
    float** dB_array,    magma_int_t lddb,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t nrowA = (side == MagmaLeft ? m : n);

    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        info = -2;
    } else if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans ) {
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

    float **dA_displ     = NULL;
    float **dB_displ     = NULL;
    float **dX_displ     = NULL;
    float **dinvA_displ  = NULL;
    float **dX_array     = NULL;
    float **dinvA_array  = NULL;

    magma_malloc((void**)&dA_displ,  batchCount * sizeof(*dA_displ));
    magma_malloc((void**)&dB_displ,  batchCount * sizeof(*dB_displ));
    magma_malloc((void**)&dX_displ,  batchCount * sizeof(*dX_displ));
    magma_malloc((void**)&dinvA_displ,  batchCount * sizeof(*dinvA_displ));
    magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
    magma_malloc((void**)&dX_array, batchCount * sizeof(*dX_array));





    magma_int_t size_dinvA;
    magma_int_t lddx = m;
    magma_int_t size_x = lddx*n;

    if ( side == MagmaLeft ) {
        size_dinvA = ((m+TRI_NB-1)/TRI_NB)*TRI_NB*TRI_NB;
    }
    else {
        size_dinvA = ((n+TRI_NB-1)/TRI_NB)*TRI_NB*TRI_NB;
    }
    float *dinvA=NULL, *dX=NULL;
    magma_int_t resetozero = 0;
    magma_smalloc( &dinvA, size_dinvA*batchCount );
    magma_smalloc( &dX, size_x*batchCount );
    if ( dinvA == NULL || dX == NULL ) {
        info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return;
    }
    magmablas_slaset_q(MagmaFull, size_dinvA, batchCount, MAGMA_S_ZERO, MAGMA_S_ZERO, dinvA, size_dinvA, queue);
    magmablas_slaset_q(MagmaFull, lddx, n*batchCount, MAGMA_S_ZERO, MAGMA_S_ZERO, dX, lddx, queue);

    sset_pointer(dX_array, dX, lddx, 0, 0, size_x, batchCount, queue);
    sset_pointer(dinvA_array, dinvA, TRI_NB, 0, 0, size_dinvA, batchCount, queue);

    magmablas_strsm_work_batched( 
                    side, uplo, transA, diag, 1, 
                    m, n, alpha,
                    dA_array,    ldda,
                    dB_array,    lddb,
                    dX_array,    lddx, 
                    dinvA_array, size_dinvA,
                    dA_displ, dB_displ, 
                    dX_displ, dinvA_displ,
                    resetozero, batchCount, queue );


    magma_free( dinvA );
    magma_free( dX );
    magma_free(dA_displ);
    magma_free(dB_displ);
    magma_free(dX_displ);
    magma_free(dinvA_displ);
    magma_free(dinvA_array);
    magma_free(dX_array);


}


