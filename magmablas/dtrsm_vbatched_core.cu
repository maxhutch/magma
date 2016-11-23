/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from magmablas/ztrsm_vbatched_core.cu, normal z -> d, Sun Nov 20 20:20:33 2016

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
*/

#define PRECISION_d

#include "magma_internal.h"
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    dtrsm_outofplace solves one of the matrix equations on gpu

        op(A)*X = alpha*B,   or
        X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,    or
        op(A) = A^T,  or
        op(A) = A^H.

    The matrix X is overwritten on B.

    This is an asynchronous version of magmablas_dtrsm with flag,
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
    flag    BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks (stored in d_dinvA) are already inverted.

    @param[in]
    m       INTEGER array, dimension(batchCount).
            On entry, each element M specifies the number of rows of 
            the corresponding B. M >= 0.

    @param[in]
    n       INTEGER array, dimension(batchCount).
            On entry, each element N specifies the number of columns of 
            the corresponding B. N >= 0.

    @param[in]
    alpha   DOUBLE PRECISION.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array A of dimension ( LDDA, k ), where k is M
             when side = MagmaLeft and is N when side = MagmaRight.
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
    ldda    INTEGER array, dimension(batchCount).
            On entry, each element LDDA specifies the first dimension of each array A.
            When side = MagmaLeft,  LDDA >= max( 1, M ),
            when side = MagmaRight, LDDA >= max( 1, N ).

    @param[in]
    dB_array       Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array B of dimension ( LDDB, N ).
             Before entry, the leading M by N part of the array B must
             contain the right-hand side matrix B.

    @param[in]
    lddb    INTEGER array, dimension(batchCount).
            On entry, each element LDDB specifies the first dimension of each array B.
            LDDB >= max( 1, M ).

    @param[in,out]
    dX_array       Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array X of dimension ( LDDX, N ).
             On entry, should be set to 0
             On exit, the solution matrix X

    @param[in]
    lddx    INTEGER array, dimension(batchCount).
            On entry, each element LDDX specifies the first dimension of each array X.
            LDDX >= max( 1, M ).

    @param
    dinvA_array    Array of pointers, dimension (batchCount).
            Each is a DOUBLE PRECISION array dinvA, a workspace on device.
            If side == MagmaLeft,  dinvA must be of size >= ceil(M/DTRTRI_BATCHED_NB)*DTRTRI_BATCHED_NB*DTRTRI_BATCHED_NB
            If side == MagmaRight, dinvA must be of size >= ceil(N/DTRTRI_BATCHED_NB)*DTRTRI_BATCHED_NB*DTRTRI_BATCHED_NB

    @param[in]
    dinvA_length    INTEGER array, dimension(batchCount). 
                   The size of each workspace matrix dinvA
                   
    @param
    dA_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dB_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dX_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dinvA_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param[in]
    resetozero INTEGER
               Used internally by DTRTRI_DIAG routine
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    max_m  INTEGER
           The maximum value in m.
    
    @param[in]
    max_n  INTEGER
           The maximum value in n.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_trsm_batched
*******************************************************************************/
extern "C" 
void magmablas_dtrsm_outofplace_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, 
    magma_int_t *m, magma_int_t* n,
    double alpha, 
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    double** dX_array,    magma_int_t* lddx, 
    double** dinvA_array, magma_int_t* dinvA_length,
    double** dA_displ, double** dB_displ, 
    double** dX_displ, double** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue)
{
    const double c_neg_one = MAGMA_D_NEG_ONE;
    const double c_one     = MAGMA_D_ONE;
    const double c_zero    = MAGMA_D_ZERO;

    magma_int_t i;
    
    // quick return if possible.
    if (max_m == 0 || max_n == 0)
        return;

    // need 4 temp buffers for dimensions, make one allocation and use offsets
    magma_int_t* tmp; 
    magma_imalloc( &tmp, 4 * batchCount );
    
    magma_int_t* tri_nb_vec = tmp;
    magma_int_t* jbv        = tri_nb_vec + batchCount;         
    magma_int_t* mm         = jbv        + batchCount;
    magma_int_t* nn         = mm         + batchCount;
    
    magma_ivec_setc(batchCount, tri_nb_vec, DTRTRI_BATCHED_NB, queue);
    
    magma_ddisplace_pointers_var_cc(dA_displ,       dA_array,    ldda,    0, 0, batchCount, queue); 
    magma_ddisplace_pointers_var_cc(dB_displ,       dB_array,    lddb,    0, 0, batchCount, queue); 
    magma_ddisplace_pointers_var_cc(dX_displ,       dX_array,    lddx,    0, 0, batchCount, queue); 
    magma_ddisplace_pointers_var_cc(dinvA_displ, dinvA_array,  tri_nb_vec,    0, 0, batchCount, queue); 

    if (side == MagmaLeft) {
        // invert diagonal blocks
        if (flag)
            magmablas_dtrtri_diag_vbatched(uplo, diag, max_m, m, dA_displ, ldda, dinvA_displ, resetozero, batchCount, queue);


        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // left, lower no-transpose
                // handle first block seperately with alpha
                magma_ivec_minc(batchCount, m, DTRTRI_BATCHED_NB, jbv, queue);
                magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, jbv, n, jbv, alpha, dinvA_displ, tri_nb_vec, dB_displ, lddb, c_zero, dX_displ, lddx, 
                                          batchCount, min(max_m, DTRTRI_BATCHED_NB), max_n, min(max_m, DTRTRI_BATCHED_NB), queue );

                if (DTRTRI_BATCHED_NB < max_m) {
                    magma_ddisplace_pointers_var_cc(dA_displ, dA_array, ldda, DTRTRI_BATCHED_NB, 0, batchCount, queue); 
                    magma_ddisplace_pointers_var_cc(dB_displ, dB_array, lddb, DTRTRI_BATCHED_NB, 0, batchCount, queue);                    
                    magma_ivec_addc(batchCount, m, -DTRTRI_BATCHED_NB, mm, queue);
                    magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, mm, n, tri_nb_vec, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, 
                                          batchCount, max_m-DTRTRI_BATCHED_NB, max_n, DTRTRI_BATCHED_NB, queue);


                    // remaining blocks
                    for( i=DTRTRI_BATCHED_NB; i < max_m; i += DTRTRI_BATCHED_NB ) {
                        magma_ivec_addc(batchCount, m, -i, jbv, queue);
                        magma_ivec_minc(batchCount, jbv, DTRTRI_BATCHED_NB, jbv, queue);
                        
                        magma_ddisplace_pointers_var_cc(dinvA_displ, dinvA_array, tri_nb_vec, 0, i, batchCount, queue);
                        magma_ddisplace_pointers_var_cc(dB_displ   , dB_array   , lddb      , i, 0, batchCount, queue);
                        magma_ddisplace_pointers_var_cc(dX_displ   , dX_array   , lddx      , i, 0, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, jbv, n, jbv, c_one, dinvA_displ, tri_nb_vec, dB_displ, lddb, c_zero, dX_displ, lddx, 
                                                  batchCount, min(max_m-i, DTRTRI_BATCHED_NB), max_n, min(max_m-i, DTRTRI_BATCHED_NB), queue);
                        if (i+DTRTRI_BATCHED_NB >= max_m)
                            break;
                        
                        magma_ddisplace_pointers_var_cc(dA_displ,    dA_array,    ldda,  i+DTRTRI_BATCHED_NB, i, batchCount, queue); 
                        magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,  i+DTRTRI_BATCHED_NB, 0, batchCount, queue);
                        magma_ivec_addc(batchCount, m, -i-DTRTRI_BATCHED_NB, mm, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, mm, n, tri_nb_vec, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, 
                                                  batchCount, max_m-i-DTRTRI_BATCHED_NB, max_n, DTRTRI_BATCHED_NB, queue);
                    }
                }
            }
            else {
                // left, upper no-transpose
                // handle first block seperately with alpha
                magma_compute_trsm_jb(batchCount, m, DTRTRI_BATCHED_NB, jbv, queue);
                magma_int_t max_jb = magma_ivec_max(batchCount, jbv, mm, batchCount, queue);
                
                magma_ivec_add(batchCount, 1, m, -1, jbv, mm, queue);
                magma_int_t max_i = magma_ivec_max(batchCount, mm, nn, batchCount, queue);
                
                magma_ddisplace_pointers_var_cv(dinvA_displ,    dinvA_array, tri_nb_vec,       0,      mm, batchCount, queue);
                magma_ddisplace_pointers_var_vc(dB_displ,          dB_array,       lddb,      mm,       0, batchCount, queue);
                magma_ddisplace_pointers_var_vc(dX_displ,          dX_array,       lddx,      mm,       0, batchCount, queue);
                magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, jbv, n, jbv, alpha, dinvA_displ, tri_nb_vec, dB_displ, lddb, c_zero, dX_displ, lddx, 
                                          batchCount, max_jb, max_n, max_jb, queue);

                if (max_i-DTRTRI_BATCHED_NB >= 0) {
                    magma_ddisplace_pointers_var_cv(dA_displ,    dA_array,    ldda,       0,       mm, batchCount, queue); 
                    magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,       0,        0, batchCount, queue);
                    magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, mm, n, jbv, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, 
                                              batchCount, max_i, max_n, max_jb, queue);
                    // remaining blocks
                    magma_ivec_addc(batchCount, mm, -DTRTRI_BATCHED_NB, mm, queue);
                    max_i -= DTRTRI_BATCHED_NB;
                    for( i=max_i; i >= 0; i -= DTRTRI_BATCHED_NB ){
                        magma_ddisplace_pointers_var_cv(dinvA_displ, dinvA_array, tri_nb_vec,       0,      mm, batchCount, queue);
                        magma_ddisplace_pointers_var_vc(dB_displ,       dB_array,       lddb,      mm,       0, batchCount, queue);
                        magma_ddisplace_pointers_var_vc(dX_displ,       dX_array,       lddx,      mm,       0, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, tri_nb_vec, n, tri_nb_vec, c_one, dinvA_displ, tri_nb_vec, dB_displ, lddb, c_zero, dX_displ, lddx, 
                                                  batchCount, DTRTRI_BATCHED_NB, max_n, DTRTRI_BATCHED_NB, queue);
                        if (i-DTRTRI_BATCHED_NB < 0)
                            break;

                        magma_ddisplace_pointers_var_cv(dA_displ,    dA_array,    ldda,       0,      mm, batchCount, queue); 
                        magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,       0,       0, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, mm, n, tri_nb_vec, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, 
                                             batchCount, max_i, max_n, DTRTRI_BATCHED_NB, queue);
                    
                        magma_ivec_addc(batchCount, mm, -DTRTRI_BATCHED_NB, mm, queue);
                        max_i -= DTRTRI_BATCHED_NB;
                    }
                }
            }
        }
        else { 
            if (uplo == MagmaLower) {
                // left, lower transpose
                // handle first block seperately with alpha
                magma_compute_trsm_jb(batchCount, m, DTRTRI_BATCHED_NB, jbv, queue);
                magma_int_t max_jb = magma_ivec_max(batchCount, jbv, mm, batchCount, queue);
                
                magma_ivec_add(batchCount, 1, m, -1, jbv, mm, queue);
                magma_int_t max_i = magma_ivec_max(batchCount, mm, nn, batchCount, queue);
                
                magma_ddisplace_pointers_var_cv(dinvA_displ,    dinvA_array, tri_nb_vec,       0,      mm, batchCount, queue);
                magma_ddisplace_pointers_var_vc(dB_displ,          dB_array,       lddb,      mm,       0, batchCount, queue);
                magma_ddisplace_pointers_var_vc(dX_displ,          dX_array,       lddx,      mm,       0, batchCount, queue);
                magmablas_dgemm_vbatched_max_nocheck( transA, MagmaNoTrans, jbv, n, jbv, alpha, dinvA_displ, tri_nb_vec, dB_displ, lddb, c_zero, dX_displ, lddx, 
                                          batchCount, max_jb, max_n, max_jb, queue);

                if (max_i-DTRTRI_BATCHED_NB >= 0) {
                    magma_ddisplace_pointers_var_vc(dA_displ,    dA_array,    ldda,      mm,       0, batchCount, queue); 
                    magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,       0,       0, batchCount, queue);
                    magmablas_dgemm_vbatched_max_nocheck( transA, MagmaNoTrans, mm, n, jbv, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, 
                                              batchCount, max_i, max_n, max_jb, queue );

                    // remaining blocks
                    magma_ivec_addc(batchCount, mm, -DTRTRI_BATCHED_NB, mm, queue);
                    max_i -= DTRTRI_BATCHED_NB;
                    for( i=max_i; i >= 0; i -= DTRTRI_BATCHED_NB ) {
                        magma_ddisplace_pointers_var_cv(dinvA_displ, dinvA_array,  tri_nb_vec,       0,      mm, batchCount, queue);
                        magma_ddisplace_pointers_var_vc(dX_displ,       dX_array,        lddx,      mm,       0, batchCount, queue);
                        magma_ddisplace_pointers_var_vc(dB_displ,       dB_array,        lddb,      mm,       0, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( transA, MagmaNoTrans, tri_nb_vec, n, tri_nb_vec, c_one, dinvA_displ, tri_nb_vec, dB_displ, lddb, c_zero, dX_displ, lddx, 
                                                  batchCount, DTRTRI_BATCHED_NB, max_n, DTRTRI_BATCHED_NB, queue);
                        
                        if (i-DTRTRI_BATCHED_NB < 0)
                            break;

                        magma_ddisplace_pointers_var_vc(dA_displ,    dA_array,    ldda,      mm,       0,  batchCount, queue); 
                        magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,       0,       0, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( transA, MagmaNoTrans, mm, n, tri_nb_vec, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, 
                                                  batchCount, max_i, max_n, DTRTRI_BATCHED_NB, queue );
                    
                        magma_ivec_addc(batchCount, mm, -DTRTRI_BATCHED_NB, mm, queue);
                        max_i -= DTRTRI_BATCHED_NB;
                    }
                }
            }
            else {
                // left, upper transpose
                // handle first block seperately with alpha
                magma_ivec_minc(batchCount, m, DTRTRI_BATCHED_NB, jbv, queue);
                magmablas_dgemm_vbatched_max_nocheck( transA, MagmaNoTrans, jbv, n, jbv, alpha, dinvA_displ, tri_nb_vec, dB_displ, lddb, c_zero, dX_displ, lddx, 
                                          batchCount, min(max_m, DTRTRI_BATCHED_NB), max_n, min(max_m, DTRTRI_BATCHED_NB), queue );
                
                if (DTRTRI_BATCHED_NB < max_m) {
                    magma_ddisplace_pointers_var_cc(dA_displ,    dA_array,    ldda,      0,   DTRTRI_BATCHED_NB, batchCount, queue); 
                    magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb, DTRTRI_BATCHED_NB,        0, batchCount, queue);
                    
                    magma_ivec_addc(batchCount, m, -DTRTRI_BATCHED_NB, mm, queue);
                    magmablas_dgemm_vbatched_max_nocheck( transA, MagmaNoTrans, mm, n, tri_nb_vec, c_neg_one, dA_displ, ldda, dX_displ, lddx, alpha, dB_displ, lddb, 
                                              batchCount, max_m-DTRTRI_BATCHED_NB, max_n, DTRTRI_BATCHED_NB, queue );
                    
                    // remaining blocks
                    for( i=DTRTRI_BATCHED_NB; i < max_m; i += DTRTRI_BATCHED_NB ) {
                        magma_ivec_addc(batchCount, m, -i, jbv, queue);
                        magma_ivec_minc(batchCount, jbv, DTRTRI_BATCHED_NB, jbv, queue);
                        
                        magma_ddisplace_pointers_var_cc(dinvA_displ, dinvA_array,  tri_nb_vec, 0, i, batchCount, queue);
                        magma_ddisplace_pointers_var_cc(dB_displ,       dB_array,        lddb, i, 0, batchCount, queue);
                        magma_ddisplace_pointers_var_cc(dX_displ,       dX_array,        lddx, i, 0, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( transA, MagmaNoTrans, jbv, n, jbv, c_one, dinvA_displ, tri_nb_vec, dB_displ, lddb, c_zero, dX_displ, lddx, 
                                                  batchCount, min(max_m-i, DTRTRI_BATCHED_NB), max_n, min(max_m-i, DTRTRI_BATCHED_NB), queue );
                        if (i+DTRTRI_BATCHED_NB >= max_m)
                            break;

                        magma_ddisplace_pointers_var_cc(dA_displ,    dA_array,    ldda,  i,        i+DTRTRI_BATCHED_NB, batchCount, queue); 
                        magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,  i+DTRTRI_BATCHED_NB,        0, batchCount, queue);
                        
                        magma_ivec_addc(batchCount, m, -i-DTRTRI_BATCHED_NB, mm, queue);
                        magmablas_dgemm_vbatched_max_nocheck( transA, MagmaNoTrans, mm, n, tri_nb_vec, c_neg_one, dA_displ, ldda, dX_displ, lddx, c_one, dB_displ, lddb, 
                                                  batchCount, max_m-i-DTRTRI_BATCHED_NB, max_n, DTRTRI_BATCHED_NB, queue );
                    }
                }
            }
        }
    }
    else {  // side == MagmaRight
        // invert diagonal blocks
        if (flag)
            magmablas_dtrtri_diag_vbatched( uplo, diag, max_n, n, dA_displ, ldda, dinvA_displ, resetozero, batchCount, queue);

        if (transA == MagmaNoTrans) {
            if (uplo == MagmaLower) {
                // right, lower no-transpose
                // handle first block seperately with alpha
                
                magma_compute_trsm_jb(batchCount, n, DTRTRI_BATCHED_NB, jbv, queue);
                magma_int_t max_jb = magma_ivec_max(batchCount, jbv, mm, batchCount, queue);
                magma_ivec_add(batchCount, 1, n, -1, jbv, nn, queue);
                magma_int_t max_i = magma_ivec_max(batchCount, nn, mm, batchCount, queue);
                
                magma_ddisplace_pointers_var_cv(dinvA_displ, dinvA_array,  tri_nb_vec, 0, nn, batchCount, queue);
                magma_ddisplace_pointers_var_cv(dB_displ,       dB_array,        lddb, 0, nn, batchCount, queue);
                magma_ddisplace_pointers_var_cv(dX_displ,       dX_array,        lddx, 0, nn, batchCount, queue);
                magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, m, jbv, jbv, alpha, dB_displ, lddb, dinvA_displ, tri_nb_vec, c_zero, dX_displ, lddx, 
                                          batchCount, max_m, max_jb, max_jb, queue );

                if (max_i-DTRTRI_BATCHED_NB >= 0) {
                    magma_ddisplace_pointers_var_vc(dA_displ,    dA_array,    ldda, nn, 0, batchCount, queue);                        
                    magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,  0, 0, batchCount, queue);
                    magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, m, nn, jbv, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, 
                                              batchCount, max_m, max_i, max_jb, queue );

                    // remaining blocks
                    magma_ivec_addc(batchCount, nn, -DTRTRI_BATCHED_NB, nn, queue);
                    max_i -= DTRTRI_BATCHED_NB;
                    for( i=max_i; i >= 0; i -= DTRTRI_BATCHED_NB ) {
                        magma_ddisplace_pointers_var_cv(dinvA_displ, dinvA_array,  tri_nb_vec, 0, nn, batchCount, queue);
                        magma_ddisplace_pointers_var_cv(dB_displ,       dB_array,        lddb, 0, nn, batchCount, queue);
                        magma_ddisplace_pointers_var_cv(dX_displ,       dX_array,        lddx, 0, nn, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, m, tri_nb_vec, tri_nb_vec, c_one, dB_displ, lddb, dinvA_displ, tri_nb_vec, c_zero, dX_displ, lddx, 
                                                  batchCount, max_m, DTRTRI_BATCHED_NB, DTRTRI_BATCHED_NB, queue );
                        
                        if (i-DTRTRI_BATCHED_NB < 0)
                            break;

                        magma_ddisplace_pointers_var_vc(dA_displ,    dA_array,    ldda, nn, 0, batchCount, queue); 
                        magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,  0, 0, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, m, nn, tri_nb_vec, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, 
                                                  batchCount, max_m, max_i, DTRTRI_BATCHED_NB, queue );
                        
                        magma_ivec_addc(batchCount, nn, -DTRTRI_BATCHED_NB, nn, queue);
                        max_i -= DTRTRI_BATCHED_NB;
                    }
                }
            } 
            else {
                // right, upper no-transpose
                // handle first block seperately with alpha
                
                magma_ivec_minc(batchCount, n, DTRTRI_BATCHED_NB, jbv, queue);
                
                magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, m, jbv, jbv, alpha, dB_displ, lddb, dinvA_displ, tri_nb_vec, c_zero, dX_displ, lddx, 
                                          batchCount, max_m, min(max_n, DTRTRI_BATCHED_NB), min(max_n, DTRTRI_BATCHED_NB), queue );
                
                if (DTRTRI_BATCHED_NB < max_n) {
                    magma_ddisplace_pointers_var_cc(dA_displ,    dA_array,    ldda, 0, DTRTRI_BATCHED_NB, batchCount, queue); 
                    magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb, 0, DTRTRI_BATCHED_NB, batchCount, queue);
                    
                    magma_ivec_addc(batchCount, n, -DTRTRI_BATCHED_NB, nn, queue);
                    
                    magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, m, nn, tri_nb_vec, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, 
                                              batchCount, max_m, max_n-DTRTRI_BATCHED_NB, DTRTRI_BATCHED_NB, queue );

                    // remaining blocks
                    for( i=DTRTRI_BATCHED_NB; i < max_n; i += DTRTRI_BATCHED_NB ) {
                        magma_ivec_addc(batchCount, n, -i, jbv, queue);
                        magma_ivec_minc(batchCount, jbv, DTRTRI_BATCHED_NB, jbv, queue);
                        
                        magma_ddisplace_pointers_var_cc(dinvA_displ, dinvA_array,  tri_nb_vec, 0, i, batchCount, queue);
                        magma_ddisplace_pointers_var_cc(dB_displ,       dB_array,        lddb, 0, i, batchCount, queue);
                        magma_ddisplace_pointers_var_cc(dX_displ,       dX_array,        lddx, 0, i, batchCount, queue);
                        
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, m, jbv, jbv, c_one, dB_displ, lddb, dinvA_displ, tri_nb_vec, c_zero, dX_displ, lddx, 
                                                  batchCount, max_m, min(DTRTRI_BATCHED_NB, max_n-i), min(DTRTRI_BATCHED_NB, max_n-i), queue );
                        
                        if (i+DTRTRI_BATCHED_NB >= max_n)
                            break;

                        magma_ddisplace_pointers_var_cc(dA_displ,    dA_array,    ldda, i, i+DTRTRI_BATCHED_NB, batchCount, queue); 
                        magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb, 0, i+DTRTRI_BATCHED_NB, batchCount, queue);
                        
                        magma_ivec_addc(batchCount, n, -i-DTRTRI_BATCHED_NB, nn, queue);
                        
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, MagmaNoTrans, m, nn, tri_nb_vec, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, 
                                                  batchCount, max_m, max_n-i-DTRTRI_BATCHED_NB, DTRTRI_BATCHED_NB, queue );
                    }
                }
            }
        }
        else { 
            if (uplo == MagmaLower) {
                // right, lower transpose
                // handle first block seperately with alpha
                
                magma_ivec_minc(batchCount, n, DTRTRI_BATCHED_NB, jbv, queue);
                
                magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, transA, m, jbv, jbv, alpha, dB_displ, lddb, dinvA_displ, tri_nb_vec, c_zero, dX_displ, lddx, 
                                          batchCount, max_m, min(max_n, DTRTRI_BATCHED_NB), min(max_n, DTRTRI_BATCHED_NB), queue );
                if (DTRTRI_BATCHED_NB < max_n) {
                    magma_ddisplace_pointers_var_cc(dA_displ,    dA_array,    ldda,  DTRTRI_BATCHED_NB,      0, batchCount, queue); 
                    magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,       0, DTRTRI_BATCHED_NB, batchCount, queue);
                    magma_ivec_addc(batchCount, n, -DTRTRI_BATCHED_NB, nn, queue);
                    magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, transA, m, nn, tri_nb_vec, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, 
                                              batchCount, max_m, max_n-DTRTRI_BATCHED_NB, DTRTRI_BATCHED_NB, queue );

                    // remaining blocks
                    for( i=DTRTRI_BATCHED_NB; i < max_n; i += DTRTRI_BATCHED_NB ) {
                        magma_ivec_addc(batchCount, n, -i, jbv, queue);
                        magma_ivec_minc(batchCount, jbv, DTRTRI_BATCHED_NB, jbv, queue);
                        
                        magma_ddisplace_pointers_var_cc(dinvA_displ, dinvA_array,  tri_nb_vec, 0, i, batchCount, queue);
                        magma_ddisplace_pointers_var_cc(dB_displ,       dB_array,        lddb, 0, i, batchCount, queue);
                        magma_ddisplace_pointers_var_cc(dX_displ,       dX_array,        lddx, 0, i, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, transA, m, jbv, jbv, c_one, dB_displ, lddb, dinvA_displ, tri_nb_vec, c_zero, dX_displ, lddx, 
                                                  batchCount, max_m, min(DTRTRI_BATCHED_NB, max_n-i), min(DTRTRI_BATCHED_NB, max_n-i), queue );
                        
                        if (i+DTRTRI_BATCHED_NB >= max_n)
                            break;

                        magma_ddisplace_pointers_var_cc(dA_displ,    dA_array,    ldda,  DTRTRI_BATCHED_NB+i,        i, batchCount, queue); 
                        magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb,         0, i+DTRTRI_BATCHED_NB, batchCount, queue);
                        magma_ivec_addc(batchCount, n, -i-DTRTRI_BATCHED_NB, nn, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, transA, m, nn, tri_nb_vec, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, 
                                                  batchCount, max_m, max_n-i-DTRTRI_BATCHED_NB, DTRTRI_BATCHED_NB, queue );
                    }
                }
            }
            else {
                // right, upper transpose
                // handle first block seperately with alpha
                magma_compute_trsm_jb(batchCount, n, DTRTRI_BATCHED_NB, jbv, queue);
                magma_int_t max_jb = magma_ivec_max(batchCount, jbv, mm, batchCount, queue);
                
                magma_ivec_add(batchCount, 1, n, -1, jbv, nn, queue);
                magma_int_t max_i = magma_ivec_max(batchCount, nn, mm, batchCount, queue);
                
                magma_ddisplace_pointers_var_cv(dinvA_displ, dinvA_array, tri_nb_vec, 0, nn, batchCount, queue);
                magma_ddisplace_pointers_var_cv(dB_displ,       dB_array,       lddb, 0, nn, batchCount, queue);
                magma_ddisplace_pointers_var_cv(dX_displ,       dX_array,       lddx, 0, nn, batchCount, queue);
                magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, transA, m, jbv, jbv, alpha, dB_displ, lddb, dinvA_displ, tri_nb_vec, c_zero, dX_displ, lddx, 
                                          batchCount, max_m, max_jb, max_jb, queue );


                if (max_i-DTRTRI_BATCHED_NB >= 0) {
                    magma_ddisplace_pointers_var_cv(dA_displ,    dA_array,    ldda, 0, nn, batchCount, queue); 
                    magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb, 0,  0, batchCount, queue);
                    magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, transA, m, nn, jbv, c_neg_one, dX_displ, lddx, dA_displ, ldda, alpha, dB_displ, lddb, 
                                              batchCount, max_m, max_i, max_jb, queue );

                    // remaining blocks
                    magma_ivec_addc(batchCount, nn, -DTRTRI_BATCHED_NB, nn, queue);
                    max_i -= DTRTRI_BATCHED_NB;
                    for( i=max_i; i >= 0; i -= DTRTRI_BATCHED_NB ) {
                        magma_ddisplace_pointers_var_cv(dinvA_displ, dinvA_array, tri_nb_vec, 0, nn, batchCount, queue);
                        magma_ddisplace_pointers_var_cv(dB_displ,       dB_array,       lddb, 0, nn, batchCount, queue);
                        magma_ddisplace_pointers_var_cv(dX_displ,       dX_array,       lddx, 0, nn, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, transA, m, tri_nb_vec, tri_nb_vec, c_one, dB_displ, lddb, dinvA_displ, tri_nb_vec, c_zero, dX_displ, lddx, 
                                                  batchCount, max_m, DTRTRI_BATCHED_NB, DTRTRI_BATCHED_NB, queue );
                        
                        if (i-DTRTRI_BATCHED_NB < 0)
                            break;

                        magma_ddisplace_pointers_var_cv(dA_displ,    dA_array,    ldda, 0, nn, batchCount, queue); 
                        magma_ddisplace_pointers_var_cc(dB_displ,    dB_array,    lddb, 0,  0, batchCount, queue);
                        magmablas_dgemm_vbatched_max_nocheck( MagmaNoTrans, transA, m, nn, tri_nb_vec, c_neg_one, dX_displ, lddx, dA_displ, ldda, c_one, dB_displ, lddb, 
                                                  batchCount, max_m, max_i, DTRTRI_BATCHED_NB, queue );
                        
                        magma_ivec_addc(batchCount, nn, -DTRTRI_BATCHED_NB, nn, queue);
                        max_i -= DTRTRI_BATCHED_NB;
                    }
                }
            }
        }
    }
    // free workspace
    magma_free(tmp);
}


/***************************************************************************//**
    Purpose
    -------
    dtrsm_work solves one of the matrix equations on gpu

        op(A)*X = alpha*B,   or
        X*op(A) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit, upper or lower triangular matrix and op(A) is one of

        op(A) = A,    or
        op(A) = A^T,  or
        op(A) = A^H.

    The matrix X is overwritten on B.

    This is an asynchronous version of magmablas_dtrsm with flag,
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
    flag    BOOLEAN.
            If flag is true, invert diagonal blocks.
            If flag is false, assume diagonal blocks (stored in d_dinvA) are already inverted.

    @param[in]
    m       INTEGER array, dimension(batchCount).
            On entry, each element M specifies the number of rows of each B. M >= 0.

    @param[in]
    n       INTEGER array, dimension(batchCount).
            On entry, each element N specifies the number of columns of each B. N >= 0.

    @param[in]
    alpha   DOUBLE PRECISION.
            On entry, alpha specifies the scalar alpha. When alpha is
            zero then A is not referenced and B need not be set before
            entry.

    @param[in]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array A of dimension ( LDDA, k ), where k is M
             when side = MagmaLeft and is N when side = MagmaRight.
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
    ldda    INTEGER array, dimension(batchCount).
            On entry, each element LDDA specifies the first dimension of each array A.
            When side = MagmaLeft,  LDDA >= max( 1, M ),
            when side = MagmaRight, LDDA >= max( 1, N ).

    @param[in,out]
    dB_array       Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array B of dimension ( LDDB, N ).
             Before entry, the leading M by N part of the array B must
             contain the right-hand side matrix B.
             \n
             On exit, the solution matrix X

    @param[in]
    lddb    INTEGER array, dimension(batchCount).
            On entry, each element LDDB specifies the first dimension of each array B.
            lddb >= max( 1, M ).

    @param[in,out]
    dX_array       Array of pointers, dimension (batchCount).
             Each is a DOUBLE PRECISION array X of dimension ( LDDX, N ).
             On entry, should be set to 0
             On exit, the solution matrix X

    @param[in]
    lddx    INTEGER array, dimension(batchCount).
            On entry, each element LDDX specifies the first dimension of each array X.
            lddx >= max( 1, M ).

    @param
    dinvA_array    Array of pointers, dimension (batchCount).
            Each is a DOUBLE PRECISION array dinvA, a workspace on device.
            If side == MagmaLeft,  dinvA must be of size >= ceil(M/DTRTRI_BATCHED_NB)*DTRTRI_BATCHED_NB*DTRTRI_BATCHED_NB,
            If side == MagmaRight, dinvA must be of size >= ceil(N/DTRTRI_BATCHED_NB)*DTRTRI_BATCHED_NB*DTRTRI_BATCHED_NB.

    @param[in]
    dinvA_length    INTEGER array, dimension(batchCount).
                   The size of each workspace matrix dinvA
                   
    @param
    dA_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dB_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dX_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param
    dinvA_displ (workspace) Array of pointers, dimension (batchCount).
    
    @param[in]
    resetozero INTEGER
               Used internally by DTRTRI_DIAG routine
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    max_m  INTEGER
           The maximum value in m.
    
    @param[in]
    max_n  INTEGER
           The maximum value in n.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_trsm_batched
*******************************************************************************/
extern "C"
void magmablas_dtrsm_work_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag, 
    magma_int_t* m, magma_int_t* n, 
    double alpha, 
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    double** dX_array,    magma_int_t* lddx, 
    double** dinvA_array, magma_int_t* dinvA_length,
    double** dA_displ, double** dB_displ, 
    double** dX_displ, double** dinvA_displ,
    magma_int_t resetozero, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue)
{
    magmablas_dtrsm_outofplace_vbatched( 
                    side, uplo, transA, diag, flag,
                    m, n, alpha,
                    dA_array,    ldda,
                    dB_array,    lddb,
                    dX_array,    lddx, 
                    dinvA_array, dinvA_length,
                    dA_displ, dB_displ, 
                    dX_displ, dinvA_displ,
                    resetozero, batchCount, 
                    max_m, max_n, queue );
    // copy X to B
    magma_ddisplace_pointers_var_cc(dX_displ,    dX_array, lddx, 0, 0, batchCount, queue);
    magma_ddisplace_pointers_var_cc(dB_displ,    dB_array, lddb, 0, 0, batchCount, queue);
    magmablas_dlacpy_vbatched( MagmaFull, max_m, max_n, m, n, dX_displ, lddx, dB_displ, lddb, batchCount, queue );
}


/***************************************************************************//**
    @see magmablas_dtrsm_work_vbatched
    @ingroup magma_trsm_batched
*******************************************************************************/
extern "C"
void magmablas_dtrsm_vbatched_max_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag, 
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, 
    magma_queue_t queue)
{
    double **dA_displ     = NULL;
    double **dB_displ     = NULL;
    double **dX_displ     = NULL;
    double **dinvA_displ  = NULL;
    double **dX_array     = NULL;
    double **dinvA_array  = NULL;

    magma_malloc((void**)&dA_displ,  batchCount * sizeof(*dA_displ));
    magma_malloc((void**)&dB_displ,  batchCount * sizeof(*dB_displ));
    magma_malloc((void**)&dX_displ,  batchCount * sizeof(*dX_displ));
    magma_malloc((void**)&dinvA_displ,  batchCount * sizeof(*dinvA_displ));
    magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
    magma_malloc((void**)&dX_array, batchCount * sizeof(*dX_array));

    magma_int_t *lddx  = m;
    magma_int_t *size_dinvA_array  = NULL;
    magma_int_t *tmp = NULL;
    
    magma_imalloc( &size_dinvA_array, batchCount );
    
    // some tmp workspaces
    magma_imalloc( &tmp, 2*batchCount );
    magma_int_t *w1 = tmp;
    magma_int_t *w2 = w1 + batchCount; 
    
    // allocate and init dX (based on input sizes, not the max. size)
    double *dX=NULL;
    magma_ivec_mul(batchCount, n, lddx, w1, queue);
    magma_int_t total_size_x = magma_isum_reduce( batchCount, w1, w2, batchCount, queue);
    magma_prefix_sum_inplace_w(w1, batchCount, w2, batchCount, queue);
    
    magma_dmalloc( &dX, total_size_x );
    if ( dX == NULL ) {
        magma_int_t info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return;
    }
    magma_dsetvector_const( total_size_x, dX, MAGMA_D_MAKE(0.0, 0.0), queue);
    magma_dset_pointer_var_cc(dX_array, dX, lddx, 0, 0, w1, batchCount, queue);
    
    // allocate and init dinvA (based on input sizes, not the max. size)
    double *dinvA=NULL;
    magma_int_t total_size_dinvA;
    if ( side == MagmaLeft ) {
        magma_ivec_roundup( batchCount, m, DTRTRI_BATCHED_NB, size_dinvA_array, queue);
        magma_ivec_mulc( batchCount, size_dinvA_array, DTRTRI_BATCHED_NB, size_dinvA_array, queue); // done inplace
        total_size_dinvA = magma_isum_reduce(batchCount, size_dinvA_array, w2, batchCount, queue);
    }
    else {
        magma_ivec_roundup( batchCount, n, DTRTRI_BATCHED_NB, size_dinvA_array, queue);
        magma_ivec_mulc( batchCount, size_dinvA_array, DTRTRI_BATCHED_NB, size_dinvA_array, queue); //done inplace
        total_size_dinvA = magma_isum_reduce(batchCount, size_dinvA_array, w2, batchCount, queue);
    }
    magma_prefix_sum_outofplace_w(size_dinvA_array, w1, batchCount, w2, batchCount, queue);
    
    magma_dmalloc( &dinvA, total_size_dinvA );
    if ( dinvA == NULL) {
        magma_int_t info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return;
    }
    magma_dsetvector_const(total_size_dinvA, dinvA, MAGMA_D_MAKE(0.0, 0.0), queue);
    magma_ivec_setc( batchCount, w2, DTRTRI_BATCHED_NB, queue);
    magma_dset_pointer_var_cc(dinvA_array, dinvA, w2, 0, 0, w1, batchCount, queue);
    
    magma_int_t resetozero = 0;
    
    magmablas_dtrsm_work_vbatched( 
                    side, uplo, transA, diag, 1, 
                    m, n, alpha,
                    dA_array,    ldda,
                    dB_array,    lddb,
                    dX_array,    lddx, 
                    dinvA_array, size_dinvA_array,
                    dA_displ, dB_displ, 
                    dX_displ, dinvA_displ,
                    resetozero, batchCount, 
                    max_m, max_n, queue );


    magma_free( tmp );
    magma_free( dinvA );
    magma_free( dX );
    magma_free(dA_displ);
    magma_free(dB_displ);
    magma_free(dX_displ);
    magma_free(dinvA_displ);
    magma_free(dinvA_array);
    magma_free(dX_array);
    magma_free(size_dinvA_array);
}
