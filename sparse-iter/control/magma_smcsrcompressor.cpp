/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_zmcsrcompressor.cpp normal z -> s, Sat Nov 15 19:54:23 2014
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>


/**
    Purpose
    -------

    Removes zeros in a CSR matrix.

    Arguments
    ---------

    @param[in,out]
    A           magma_s_sparse_matrix*
                input/output matrix 
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_smcsrcompressor(
    magma_s_sparse_matrix *A,
    magma_queue_t queue )
{
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSR ) {

        magma_s_sparse_matrix B;

        magma_s_mconvert( *A, &B, Magma_CSR, Magma_CSR, queue );

        magma_free_cpu( A->row );
        magma_free_cpu( A->col );
        magma_free_cpu( A->val );
        magma_s_csr_compressor(&B.val, &B.row, &B.col, 
                       &A->val, &A->row, &A->col, &A->num_rows, queue );  
        A->nnz = A->row[A->num_rows];

        magma_s_mfree( &B, queue );       

        return MAGMA_SUCCESS; 
    }
    else {

        magma_s_sparse_matrix hA, CSRA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_s_mtransfer( *A, &hA, A->memory_location, Magma_CPU, queue );
        magma_s_mconvert( hA, &CSRA, hA.storage_type, Magma_CSR, queue );

        magma_smcsrcompressor( &CSRA, queue );

        magma_s_mfree( &hA, queue );
        magma_s_mfree( A, queue );
        magma_s_mconvert( CSRA, &hA, Magma_CSR, A_storage, queue );
        magma_s_mtransfer( hA, A, Magma_CPU, A_location, queue );
        magma_s_mfree( &hA, queue );
        magma_s_mfree( &CSRA, queue );    

        return MAGMA_SUCCESS; 
    }
}


