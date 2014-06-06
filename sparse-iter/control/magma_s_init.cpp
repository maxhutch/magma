/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magma_z_init.cpp normal z -> s, Fri May 30 10:41:45 2014
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
#include "../include/magmasparse_s.h"
#include "../../include/magma.h"
#include "../include/mmio.h"



using namespace std;








/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Initialize a magma_s_vector.


    Arguments
    =========

    magma_s_vector x                     vector to initialize   
    magma_location_t memory_location     memory for vector 
    magma_int_t num_rows                 desired length of vector      
    float values            entries in vector

    ========================================================================  */

magma_int_t 
magma_s_vinit(    magma_s_vector *x, 
                  magma_location_t memory_location,
                  magma_int_t num_rows, 
                  float values ){

    x->memory_location = Magma_CPU;
    x->num_rows = num_rows;
    x->nnz = num_rows;
    if( memory_location == Magma_CPU ){
        x->memory_location = Magma_CPU;

        magma_smalloc_cpu( &x->val, num_rows );
        if ( x->val == NULL )
            return MAGMA_ERR_HOST_ALLOC;
        for( magma_int_t i=0; i<num_rows; i++)
             x->val[i] = values; 
        return MAGMA_SUCCESS;  
    }
    else if( memory_location == Magma_DEV ){
        x->memory_location = Magma_DEV;

        float *tmp;

        magma_smalloc_cpu( &tmp, num_rows );
        if ( tmp == NULL )
            return MAGMA_ERR_HOST_ALLOC;
        for( magma_int_t i=0; i<num_rows; i++)
             tmp[i] = values; 

        if (MAGMA_SUCCESS != magma_smalloc( &x->val, x->num_rows)) 
            return MAGMA_ERR_DEVICE_ALLOC;

        // data transfer
        magma_ssetvector( x->num_rows, tmp, 1, x->val, 1 );
        magma_free_cpu(tmp);

        return MAGMA_SUCCESS; 
    }
    return MAGMA_SUCCESS; 
}



   


