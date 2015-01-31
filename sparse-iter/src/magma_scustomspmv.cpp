/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_zcustomspmv.cpp normal z -> s, Fri Jan 30 19:00:30 2015
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>


/**
    Purpose
    -------

    This is an interface to any custom sparse matrix vector product. 
    It should compute y = alpha*FUNCTION(x) + beta*y
    The vectors are located on the device, the scalars on the CPU.


    Arguments
    ---------

    @param[in]
    alpha       float
                scalar alpha

    @param[in]
    x           magma_s_vector
                input vector x  
                
    @param[in]
    beta        float
                scalar beta
    @param[out]
    y           magma_s_vector
                output vector y      
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sblas
    ********************************************************************/

extern "C" magma_int_t
magma_scustomspmv(
    float alpha, 
    magma_s_vector x, 
    float beta, 
    magma_s_vector y,
    magma_queue_t queue )
{
    // vector access via x.dval, y.dval
    // sizes are x.num_rows, x.num_cols
    

    return MAGMA_SUCCESS; 

}





