/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @precisions normal z -> s d c
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

#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"


using namespace std;




/**
    Purpose
    -------

    Passes a vector to MAGMA.

    Arguments
    ---------

    @param[in]
    m           magma_int_t 
                number of rows

    @param[in]
    n           magma_int_t 
                number of columns

    @param[in]
    val         magmaDoubleComplex*
                array containing vector entries

    @param[out]
    v           magma_z_vector*
                magma vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zvset(
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex *val,
    magma_z_vector *v,
    magma_queue_t queue )
{
    v->num_rows = m;
    v->num_cols = n;
    v->nnz = m*n;
    v->memory_location = Magma_CPU;
    v->val = val;
    v->major = MagmaColMajor;

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Passes a MAGMA vector back.

    Arguments
    ---------

    @param[in]
    v           magma_z_vector
                magma vector

    @param[out]
    m           magma_int_t 
                number of rows

    @param[out]
    n           magma_int_t 
                number of columns

    @param[out]
    val         magmaDoubleComplex*
                array containing vector entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_vget(
    magma_z_vector v,
    magma_int_t *m, magma_int_t *n, 
    magmaDoubleComplex **val,
    magma_queue_t queue )
{
    if ( v.memory_location == Magma_CPU ) {

        *m = v.num_rows;
        *n = v.num_cols;
        *val = v.val;
    } else {
        magma_z_vector v_CPU;
        magma_z_vtransfer( v, &v_CPU, v.memory_location, Magma_CPU, queue ); 
        magma_zvget( v_CPU, m, n, val, queue );
        magma_z_vfree( &v_CPU, queue );
    }
    return MAGMA_SUCCESS;
}


