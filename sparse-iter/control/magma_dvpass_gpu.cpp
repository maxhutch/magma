/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_zvpass_gpu.cpp normal z -> d, Sat Nov 15 19:54:23 2014
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from 
//  the IO functions provided by MatrixMarket

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>

#include "magmasparse_d.h"
#include "magma.h"
#include "mmio.h"


using namespace std;




/**
    Purpose
    -------

    Passes a vector to MAGMA (located on DEV).

    Arguments
    ---------

    @param[in]
    m           magma_int_t 
                number of rows

    @param[in]
    n           magma_int_t 
                number of columns

    @param[in]
    val         magmaDouble_ptr 
                array containing vector entries

    @param[out]
    v           magma_d_vector*
                magma vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_dvset_gpu(
    magma_int_t m, magma_int_t n, 
    magmaDouble_ptr val,
    magma_d_vector *v,
    magma_queue_t queue )
{
    v->num_rows = m;
    v->num_cols = n;
    v->nnz = m*n;
    v->memory_location = Magma_DEV;
    v->dval = val;
    v->major = MagmaColMajor;

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Passes a MAGMA vector back (located on DEV).

    Arguments
    ---------

    @param[in]
    v           magma_d_vector
                magma vector

    @param[out]
    m           magma_int_t 
                number of rows

    @param[out]
    n           magma_int_t 
                number of columns

    @param[out]
    val         magmaDouble_ptr 
                array containing vector entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C"
magma_int_t
magma_vget_gpu(
    magma_d_vector v,
    magma_int_t *m, magma_int_t *n, 
    magmaDouble_ptr *val,
    magma_queue_t queue )
{
    if ( v.memory_location == Magma_DEV ) {

        *m = v.num_rows;
        *n = v.num_cols;
        *val = v.dval;
    } else {
        magma_d_vector v_DEV;
        magma_d_vtransfer( v, &v_DEV, v.memory_location, Magma_DEV, queue ); 
        magma_dvget_gpu( v_DEV, m, n, val, queue );
        magma_d_vfree( &v_DEV, queue );
    }
    return MAGMA_SUCCESS;
}


