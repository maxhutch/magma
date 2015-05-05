/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from magma_zmcsrcompressor.cpp normal z -> s, Sun May  3 11:23:01 2015
       @author Hartwig Anzt

*/
#include "common_magmasparse.h"


/**
    Purpose
    -------

    Removes zeros in a CSR matrix.

    Arguments
    ---------

    @param[in,out]
    A           magma_s_matrix*
                input/output matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_smcsrcompressor(
    magma_s_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_s_matrix B={Magma_CSR};
    magma_s_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
        
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSR ) {


        CHECK( magma_smconvert( *A, &B, Magma_CSR, Magma_CSR, queue ));

        magma_free_cpu( A->row );
        magma_free_cpu( A->col );
        magma_free_cpu( A->val );
        CHECK( magma_s_csr_compressor(&B.val, &B.row, &B.col,
                       &A->val, &A->row, &A->col, &A->num_rows, queue ));
        A->nnz = A->row[A->num_rows];
    }
    else {

        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_smtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_smconvert( hA, &CSRA, hA.storage_type, Magma_CSR, queue ));

        CHECK( magma_smcsrcompressor( &CSRA, queue ));

        magma_smfree( &hA, queue );
        magma_smfree( A, queue );
        CHECK( magma_smconvert( CSRA, &hA, Magma_CSR, A_storage, queue ));
        CHECK( magma_smtransfer( hA, A, Magma_CPU, A_location, queue ));
        magma_smfree( &hA, queue );
        magma_smfree( &CSRA, queue );
    }
    
cleanup:
    magma_smfree( &hA, queue );
    magma_smfree( &CSRA, queue );
    magma_smfree( &B, queue );
    return info;
}


