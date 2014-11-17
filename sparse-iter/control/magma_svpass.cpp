/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_zvpass.cpp normal z -> s, Sat Nov 15 19:54:23 2014
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

#include "magmasparse_s.h"
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
    val         float*
                array containing vector entries

    @param[out]
    v           magma_s_vector*
                magma vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_svset(
    magma_int_t m, magma_int_t n, 
    float *val,
    magma_s_vector *v,
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
    v           magma_s_vector
                magma vector

    @param[out]
    m           magma_int_t 
                number of rows

    @param[out]
    n           magma_int_t 
                number of columns

    @param[out]
    val         float*
                array containing vector entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_saux
    ********************************************************************/

extern "C"
magma_int_t
magma_vget(
    magma_s_vector v,
    magma_int_t *m, magma_int_t *n, 
    float **val,
    magma_queue_t queue )
{
    if ( v.memory_location == Magma_CPU ) {

        *m = v.num_rows;
        *n = v.num_cols;
        *val = v.val;
    } else {
        magma_s_vector v_CPU;
        magma_s_vtransfer( v, &v_CPU, v.memory_location, Magma_CPU, queue ); 
        magma_svget( v_CPU, m, n, val, queue );
        magma_s_vfree( &v_CPU, queue );
    }
    return MAGMA_SUCCESS;
}


