/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>
#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"



using namespace std;








/**
    Purpose
    -------

    Initialize a magma_z_vector.


    Arguments
    ---------

    @param[out]
    x           magma_z_vector*
                vector to initialize   

    @param[in]
    mem_loc     magma_location_t
                memory for vector 

    @param[in]
    num_rows    magma_int_t
                desired length of vector      

    @param[in]
    values      magmaDoubleComplex
                entries in vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_vinit(
    magma_z_vector *x, 
    magma_location_t mem_loc,
    magma_int_t num_rows, 
    magmaDoubleComplex values,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    x->memory_location = Magma_CPU;
    x->num_rows = num_rows;
    x->num_cols = 1;
    x->nnz = num_rows*1;
    x->major = MagmaColMajor;
    if ( mem_loc == Magma_CPU ) {
        x->memory_location = Magma_CPU;

        magma_zmalloc_cpu( &x->val, num_rows );
        if ( x->val == NULL ) {
            magmablasSetKernelStream( orig_queue );
            return MAGMA_ERR_HOST_ALLOC;
    }
        for( magma_int_t i=0; i<num_rows; i++)
             x->val[i] = values; 
    }
    else if ( mem_loc == Magma_DEV ) {
        x->memory_location = Magma_DEV;

        if (MAGMA_SUCCESS != magma_zmalloc( &x->dval, x->num_rows)){ 
            magmablasSetKernelStream( orig_queue );
            return MAGMA_ERR_DEVICE_ALLOC;
        }

        magmablas_zlaset(MagmaFull, num_rows, 1, values, values, x->val, num_rows);

    }
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}



   


