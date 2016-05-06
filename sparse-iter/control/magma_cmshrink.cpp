/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/control/magma_zmshrink.cpp normal z -> c, Mon May  2 23:30:52 2016
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

#include <cuda.h>  // for CUDA_VERSION





/**
    Purpose
    -------

    Shrinks a non-square matrix (m < n) to the smaller dimension.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                sparse matrix A
                
    @param[out]
    B           magma_c_matrix*
                sparse matrix A

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmshrink(
    magma_c_matrix A,
    magma_c_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_c_matrix hA={Magma_CSR}, hACSR={Magma_CSR}, hB={Magma_CSR}, hBCSR={Magma_CSR};
     
    if( A.num_rows<=A.num_cols){
        if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSR ){
            CHECK( magma_cmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            for(magma_int_t i=0; i<A.nnz; i++){
                if( B->col[i] >= A.num_rows ){
                    B->val[i] = MAGMA_C_ZERO;   
                }
            }
            CHECK( magma_cmcsrcompressor( B, queue ) );
            B->num_cols = B->num_rows;
        } else {
            CHECK( magma_cmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
            CHECK( magma_cmconvert( hA, &hACSR, A.storage_type, Magma_CSR, queue ));
            CHECK( magma_cmshrink( hACSR, &hBCSR, queue ));
            CHECK( magma_cmconvert( hBCSR, &hB, Magma_CSR, A.storage_type, queue ));
            CHECK( magma_cmtransfer( hB, B, Magma_CPU, A.memory_location, queue ));
        }
    } else {
        printf("%% error: A has too many rows: m > n.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    
cleanup:    
    magma_cmfree( &hA, queue );
    magma_cmfree( &hB, queue );
    magma_cmfree( &hACSR, queue );
    magma_cmfree( &hBCSR, queue );

    return info;
}
