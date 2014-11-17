/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

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

    Transposes a vector from col to row major and vice versa.


    Arguments
    ---------

    @param[in]
    x           magma_z_vector
                input vector

    @param[out]
    y           magma_z_vector*
                output vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zvtranspose(
    magma_z_vector x,
    magma_z_vector *y,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( x.memory_location == Magma_DEV ) {
        magma_z_vinit( y, Magma_DEV, x.num_rows*x.num_cols, MAGMA_Z_ZERO, queue );
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        magma_int_t    m = x.num_rows;
        magma_int_t    n = x.num_cols;
        if ( x.major == MagmaColMajor) {
            y->major = MagmaRowMajor;
            magmablas_ztranspose( m, n, x.val, m, y->val, n );
        }
        else {
            y->major = MagmaColMajor;
            magmablas_ztranspose( n, m, x.val, n, y->val, m );
        }
    } else {
        magma_z_vector x_d, y_d;
        magma_z_vtransfer( x, &x_d, Magma_CPU, Magma_DEV, queue );
        magma_zvtranspose( x_d, &y_d, queue );  
        magma_z_vtransfer( y_d, y, Magma_DEV, Magma_CPU, queue );
        magma_z_vfree( &x_d, queue );
        magma_z_vfree( &y_d, queue );

    }
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}



   


