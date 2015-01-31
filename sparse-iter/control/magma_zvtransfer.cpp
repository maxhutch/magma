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

    Copies a vector from memory location src to memory location dst.


    Arguments
    ---------

    @param[in]
    x           magma_z_vector
                vector x

    @param[out]
    y           magma_z_vector*
                copy of x

    @param[in]
    src         magma_location_t
                original location x

    @param[in]
    dst         magma_location_t
                location of the copy of x

   
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_vtransfer(
    magma_z_vector x,
    magma_z_vector *y,
    magma_location_t src,
    magma_location_t dst,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    magma_int_t stat_cpu = 0, stat_dev = 0;
    y->val = NULL;
    y->dval = NULL;

    // first case: copy matrix from host to device
    if ( src == Magma_CPU && dst == Magma_DEV ) {
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        y->nnz = x.nnz;
        y->major = x.major;
        // memory allocation
        stat_dev += magma_zmalloc( &y->dval, x.num_rows ); 
        if( stat_dev != 0 ){ goto CLEANUP; }

        // data transfer
        magma_zsetvector( x.num_rows, x.val, 1, y->val, 1 );
    }
    // second case: copy matrix from host to host
    if ( src == Magma_CPU && dst == Magma_CPU ) {
        // fill in information for B
        y->memory_location = Magma_CPU;
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        y->nnz = x.nnz;
        y->major = x.major;
        // memory allocation
        stat_cpu += magma_zmalloc_cpu( &y->val, x.num_rows ); 
        if( stat_cpu != 0 ){ goto CLEANUP; }
        // data transfer
        for( magma_int_t i=0; i<x.num_rows; i++ )
            y->val[i] = x.val[i];
    }
    // third case: copy matrix from device to host
    if ( src == Magma_DEV && dst == Magma_CPU ) {
        // fill in information for B
        y->memory_location = Magma_CPU;
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        y->nnz = x.nnz;
        y->major = x.major;
        // memory allocation
        stat_cpu += magma_zmalloc_cpu( &y->val, x.num_rows ); 
        if( stat_cpu != 0 ){ goto CLEANUP; }
        // data transfer
        magma_zgetvector( x.num_rows, x.val, 1, y->val, 1 );
    }
    // fourth case: copy matrix from device to device
    if ( src == Magma_DEV && dst == Magma_DEV ) {
        // fill in information for B
        y->memory_location = Magma_DEV;
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        y->nnz = x.nnz;
        y->major = x.major;
        // memory allocation
        stat_dev += magma_zmalloc( &y->dval, x.num_rows ); 
        if( stat_dev != 0 ){ goto CLEANUP; }
       
        // data transfer
        magma_zcopyvector( x.num_rows, x.val, 1, y->val, 1 );
    }

CLEANUP:
    if( stat_cpu != 0 ){
        magma_z_vfree( y, queue );
        magmablasSetKernelStream( orig_queue );
        return MAGMA_ERR_HOST_ALLOC;
    }
    if( stat_dev != 0 ){
        magma_z_vfree( y, queue );
        magmablasSetKernelStream( orig_queue );
        return MAGMA_ERR_DEVICE_ALLOC;
    }  
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}
