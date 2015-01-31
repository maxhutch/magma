/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c
       @author Azzam Haidar 
*/
#include "common_magma.h"

extern "C"
void magmablas_zherk_gpu(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k, magma_int_t nb,
    double alpha,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t a_offset,
    double beta,
    magmaDoubleComplex_ptr dC, magma_int_t lddc, magma_int_t c_offset)
{
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda + (a_offset) )
    #define dC(i_, j_) (dC + (i_) + (j_)*lddc)
    
    magma_trans_t transA;
    magma_trans_t transB;  
    magmaDoubleComplex cbeta  = MAGMA_Z_MAKE( beta, 0. );
    magmaDoubleComplex calpha = MAGMA_Z_MAKE( alpha, 0. );
    
    if (trans == MagmaNoTrans) {
        transA = MagmaNoTrans;
        transB = Magma_ConjTrans;
    } else {
        transA = Magma_ConjTrans;
        transB = MagmaNoTrans;
    }

    if (uplo == MagmaUpper) {
            printf("Error not supported\n");
            return;
    }

    magma_int_t ib, ioff;
    magma_int_t blockoffset = c_offset % nb;
    // loop over all blocks and does A * A**H
    // blockoffset is c_offset within first block; for subsequent blocks it is 0
    for( magma_int_t i = 0; i < n; i += ib ) {
        ib     = min( nb-blockoffset, n-i );  // block size
        ioff   = i + c_offset;                  // global index in parent matrix
        // C[i:n,i] += A[i:n,0] * A[i,0]'
        // printf( "zgemm  n=%4d, ib=%4d, k=%4d, i=%4d  ioff=%4d\n", n-i, ib, k, i, ioff );
        magma_zgemm( transA, transB, n-i, ib, k,
                     calpha, dA(i,0),       ldda,
                             dA(i,0),       ldda,
                     cbeta,  dC(ioff,ioff), lddc );
        blockoffset = 0;
    }
}
