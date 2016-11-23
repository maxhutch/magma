/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/control/magma_zvtranspose.cpp, normal z -> d, Sun Nov 20 20:20:43 2016
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"


/**
    Purpose
    -------

    Transposes a vector from col to row major and vice versa.


    Arguments
    ---------

    @param[in]
    x           magma_d_matrix
                input vector

    @param[out]
    y           magma_d_matrix*
                output vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dvtranspose(
    magma_d_matrix x,
    magma_d_matrix *y,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t    m = x.num_rows;
    magma_int_t    n = x.num_cols;
    
    magma_d_matrix dx={Magma_CSR}, dy={Magma_CSR};
            
    if ( x.memory_location == Magma_DEV ) {
        CHECK( magma_dvinit( y, Magma_DEV, x.num_rows,x.num_cols, MAGMA_D_ZERO, queue ));
        y->num_rows = x.num_rows;
        y->num_cols = x.num_cols;
        y->storage_type = x.storage_type;
        if ( x.major == MagmaColMajor) {
            y->major = MagmaRowMajor;
            magmablas_dtranspose( m, n, x.val, m, y->val, n, queue );
        }
        else {
            y->major = MagmaColMajor;
            magmablas_dtranspose( n, m, x.val, n, y->val, m, queue );
        }
    } else {
        CHECK( magma_dmtransfer( x, &dx, Magma_CPU, Magma_DEV, queue ));
        CHECK( magma_dvtranspose( dx, &dy, queue ));
        CHECK( magma_dmtransfer( dy, y, Magma_DEV, Magma_CPU, queue ));
    }
    
cleanup:
    if( info != 0 ){
        magma_dmfree( y, queue );
    }
    magma_dmfree( &dx, queue );
    magma_dmfree( &dy, queue );
    return info;
}
