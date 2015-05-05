/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from magma_zmcsrcompressor.cpp normal z -> c, Sun May  3 11:23:01 2015
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
    A           magma_c_matrix*
                input/output matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cmcsrcompressor(
    magma_c_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_c_matrix B={Magma_CSR};
    magma_c_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
        
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSR ) {


        CHECK( magma_cmconvert( *A, &B, Magma_CSR, Magma_CSR, queue ));

        magma_free_cpu( A->row );
        magma_free_cpu( A->col );
        magma_free_cpu( A->val );
        CHECK( magma_c_csr_compressor(&B.val, &B.row, &B.col,
                       &A->val, &A->row, &A->col, &A->num_rows, queue ));
        A->nnz = A->row[A->num_rows];
    }
    else {

        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_cmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_cmconvert( hA, &CSRA, hA.storage_type, Magma_CSR, queue ));

        CHECK( magma_cmcsrcompressor( &CSRA, queue ));

        magma_cmfree( &hA, queue );
        magma_cmfree( A, queue );
        CHECK( magma_cmconvert( CSRA, &hA, Magma_CSR, A_storage, queue ));
        CHECK( magma_cmtransfer( hA, A, Magma_CPU, A_location, queue ));
        magma_cmfree( &hA, queue );
        magma_cmfree( &CSRA, queue );
    }
    
cleanup:
    magma_cmfree( &hA, queue );
    magma_cmfree( &CSRA, queue );
    magma_cmfree( &B, queue );
    return info;
}


