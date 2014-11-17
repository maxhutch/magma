/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from magma_zvtranspose.cpp normal z -> d, Sat Nov 15 19:54:23 2014
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
#include "magmasparse_d.h"
#include "magma.h"
#include "mmio.h"



using namespace std;








/**
    Purpose
    -------

    Transposes a vector from col to row major and vice versa.


    Arguments
    ---------

    @param[in]
    x           magma_d_vector
                input vector

    @param[out]
    y           magma_d_vector*
                output vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dvtranspose(
    magma_d_vector x,
    magma_d_vector *y,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( x.memory_location == Magma_DEV ) {
        magma_d_vinit( y, Magma_DEV, x.num_rows*x.num_cols, MAGMA_D_ZERO, queue );
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        magma_int_t    m = x.num_rows;
        magma_int_t    n = x.num_cols;
        if ( x.major == MagmaColMajor) {
            y->major = MagmaRowMajor;
            magmablas_dtranspose( m, n, x.val, m, y->val, n );
        }
        else {
            y->major = MagmaColMajor;
            magmablas_dtranspose( n, m, x.val, n, y->val, m );
        }
    } else {
        magma_d_vector x_d, y_d;
        magma_d_vtransfer( x, &x_d, Magma_CPU, Magma_DEV, queue );
        magma_dvtranspose( x_d, &y_d, queue );  
        magma_d_vtransfer( y_d, y, Magma_DEV, Magma_CPU, queue );
        magma_d_vfree( &x_d, queue );
        magma_d_vfree( &y_d, queue );

    }
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}



   


