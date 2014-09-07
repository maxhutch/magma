/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
       @author Azzam Haidar 
*/
#include "common_magma.h"

extern "C"
void magmablas_zherk_gpu(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magma_int_t nb,
    double alpha, magmaDoubleComplex *dA, magma_int_t lda, magma_int_t aoff,
    double beta,           magmaDoubleComplex *dC, magma_int_t ldc,  magma_int_t offset)
{
    #define dA(i, j) (dA + (i) + (j)*lda + (aoff) )
    #define dC(i, j) (dC + (i) + (j)*ldc)
    magma_transA_t transA;
    magma_transB_t transB;  
    magmaDoubleComplex cbeta  = MAGMA_Z_MAKE( beta, 0. );
    magmaDoubleComplex calpha = MAGMA_Z_MAKE( alpha, 0. );
    
    if(trans==MagmaNoTrans){
        transA = MagmaNoTrans;
        transB = Magma_ConjTrans;
    }else{
        transA = Magma_ConjTrans;
        transB = MagmaNoTrans;
    }

    if(uplo==MagmaUpper){
            printf("Error not supported\n");
            return;
    }



    magma_int_t ib, ioff;
    magma_int_t blockoffset = offset % nb;
    // loop over all blocks and does A*A'
    // blockoffset is offset within first block; for subsequent blocks it is 0
    for( magma_int_t i = 0; i < n; i += ib ) {
        ib     = min( nb-blockoffset, n-i );  // block size
        ioff   = i + offset;                  // global index in parent matrix
        // C[i:n,i] += A[i:n,0] * A[i,0]'
        // printf( "zgemm  n=%4d, ib=%4d, k=%4d, i=%4d  ioff=%4d\n", n-i, ib, k, i, ioff );
        magma_zgemm( transA, transB, n-i, ib, k,
                     calpha, dA(i,0),       lda,
                             dA(i,0),       lda,
                     cbeta,  dC(ioff,ioff), ldc );
        blockoffset = 0;
    }
}
