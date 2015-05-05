/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from magma_zmcsrcompressor.cpp normal z -> d, Sun May  3 11:23:01 2015
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
    A           magma_d_matrix*
                input/output matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dmcsrcompressor(
    magma_d_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_d_matrix B={Magma_CSR};
    magma_d_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
        
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSR ) {


        CHECK( magma_dmconvert( *A, &B, Magma_CSR, Magma_CSR, queue ));

        magma_free_cpu( A->row );
        magma_free_cpu( A->col );
        magma_free_cpu( A->val );
        CHECK( magma_d_csr_compressor(&B.val, &B.row, &B.col,
                       &A->val, &A->row, &A->col, &A->num_rows, queue ));
        A->nnz = A->row[A->num_rows];
    }
    else {

        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_dmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_dmconvert( hA, &CSRA, hA.storage_type, Magma_CSR, queue ));

        CHECK( magma_dmcsrcompressor( &CSRA, queue ));

        magma_dmfree( &hA, queue );
        magma_dmfree( A, queue );
        CHECK( magma_dmconvert( CSRA, &hA, Magma_CSR, A_storage, queue ));
        CHECK( magma_dmtransfer( hA, A, Magma_CPU, A_location, queue ));
        magma_dmfree( &hA, queue );
        magma_dmfree( &CSRA, queue );
    }
    
cleanup:
    magma_dmfree( &hA, queue );
    magma_dmfree( &CSRA, queue );
    magma_dmfree( &B, queue );
    return info;
}


