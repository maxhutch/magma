/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from magma_zvinit.cpp normal z -> s, Sun May  3 11:23:01 2015
       @author Hartwig Anzt
*/
#include "common_magmasparse.h"


/**
    Purpose
    -------

    Allocates memory for magma_s_matrix and initializes it
    with the passed value.


    Arguments
    ---------

    @param[out]
    x           magma_s_matrix*
                vector to initialize

    @param[in]
    mem_loc     magma_location_t
                memory for vector

    @param[in]
    num_rows    magma_int_t
                desired length of vector

    @param[in]
    values      float
                entries in vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_svinit(
    magma_s_matrix *x,
    magma_location_t mem_loc,
    magma_int_t num_rows,
    magma_int_t num_cols,
    float values,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    x->memory_location = Magma_CPU;
    x->num_rows = num_rows;
    x->storage_type = Magma_DENSE;
    x->ld = num_rows;
    x->num_cols = num_cols;
    x->nnz = num_rows*num_cols;
    x->major = MagmaColMajor;
    if ( mem_loc == Magma_CPU ) {
        x->memory_location = Magma_CPU;
        CHECK( magma_smalloc_cpu( &x->val, x->nnz ));
        for( magma_int_t i=0; i<x->nnz; i++)
             x->val[i] = values;
    }
    else if ( mem_loc == Magma_DEV ) {
        x->memory_location = Magma_DEV;
        CHECK( magma_smalloc( &x->val, x->nnz ));
        magmablas_slaset(MagmaFull, x->num_rows, x->num_cols, values, values, x->val, x->num_rows);
    }
    
cleanup:
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}



   


