/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/control/magma_zmshrink.cpp normal z -> d, Mon May  2 23:30:52 2016
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
    A           magma_d_matrix
                sparse matrix A
                
    @param[out]
    B           magma_d_matrix*
                sparse matrix A

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dmshrink(
    magma_d_matrix A,
    magma_d_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_d_matrix hA={Magma_CSR}, hACSR={Magma_CSR}, hB={Magma_CSR}, hBCSR={Magma_CSR};
     
    if( A.num_rows<=A.num_cols){
        if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSR ){
            CHECK( magma_dmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            for(magma_int_t i=0; i<A.nnz; i++){
                if( B->col[i] >= A.num_rows ){
                    B->val[i] = MAGMA_D_ZERO;   
                }
            }
            CHECK( magma_dmcsrcompressor( B, queue ) );
            B->num_cols = B->num_rows;
        } else {
            CHECK( magma_dmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
            CHECK( magma_dmconvert( hA, &hACSR, A.storage_type, Magma_CSR, queue ));
            CHECK( magma_dmshrink( hACSR, &hBCSR, queue ));
            CHECK( magma_dmconvert( hBCSR, &hB, Magma_CSR, A.storage_type, queue ));
            CHECK( magma_dmtransfer( hB, B, Magma_CPU, A.memory_location, queue ));
        }
    } else {
        printf("%% error: A has too many rows: m > n.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
        goto cleanup;
    }
    
cleanup:    
    magma_dmfree( &hA, queue );
    magma_dmfree( &hB, queue );
    magma_dmfree( &hACSR, queue );
    magma_dmfree( &hBCSR, queue );

    return info;
}
