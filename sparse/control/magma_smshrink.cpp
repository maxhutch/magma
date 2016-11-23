/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/control/magma_zmshrink.cpp, normal z -> s, Sun Nov 20 20:20:42 2016
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

/**
    Purpose
    -------

    Shrinks a non-square matrix (m < n) to the smaller dimension.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                sparse matrix A
                
    @param[out]
    B           magma_s_matrix*
                sparse matrix A

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_smshrink(
    magma_s_matrix A,
    magma_s_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_s_matrix hA={Magma_CSR}, hACSR={Magma_CSR}, hB={Magma_CSR}, hBCSR={Magma_CSR};
     
    if( A.num_rows<=A.num_cols){
        if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSR ){
            CHECK( magma_smconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            for(magma_int_t i=0; i<A.nnz; i++){
                if( B->col[i] >= A.num_rows ){
                    B->val[i] = MAGMA_S_ZERO;   
                }
            }
            CHECK( magma_smcsrcompressor( B, queue ) );
            B->num_cols = B->num_rows;
        } else {
            CHECK( magma_smtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
            CHECK( magma_smconvert( hA, &hACSR, A.storage_type, Magma_CSR, queue ));
            CHECK( magma_smshrink( hACSR, &hBCSR, queue ));
            CHECK( magma_smconvert( hBCSR, &hB, Magma_CSR, A.storage_type, queue ));
            CHECK( magma_smtransfer( hB, B, Magma_CPU, A.memory_location, queue ));
        }
    } else {
        printf("%% error: A has too many rows: m > n.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    
cleanup:    
    magma_smfree( &hA, queue );
    magma_smfree( &hB, queue );
    magma_smfree( &hACSR, queue );
    magma_smfree( &hBCSR, queue );

    return info;
}
