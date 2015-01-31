/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_zvtranspose.cpp normal z -> c, Fri Jan 30 19:00:32 2015
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
#include "magmasparse_c.h"
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
    x           magma_c_vector
                input vector

    @param[out]
    y           magma_c_vector*
                output vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cvtranspose(
    magma_c_vector x,
    magma_c_vector *y,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( x.memory_location == Magma_DEV ) {
        magma_c_vinit( y, Magma_DEV, x.num_rows*x.num_cols, MAGMA_C_ZERO, queue );
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        magma_int_t    m = x.num_rows;
        magma_int_t    n = x.num_cols;
        if ( x.major == MagmaColMajor) {
            y->major = MagmaRowMajor;
            magmablas_ctranspose( m, n, x.val, m, y->val, n );
        }
        else {
            y->major = MagmaColMajor;
            magmablas_ctranspose( n, m, x.val, n, y->val, m );
        }
    } else {
        magma_c_vector x_d, y_d;
        magma_c_vtransfer( x, &x_d, Magma_CPU, Magma_DEV, queue );
        magma_cvtranspose( x_d, &y_d, queue );  
        magma_c_vtransfer( y_d, y, Magma_DEV, Magma_CPU, queue );
        magma_c_vfree( &x_d, queue );
        magma_c_vfree( &y_d, queue );

    }
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}



   


