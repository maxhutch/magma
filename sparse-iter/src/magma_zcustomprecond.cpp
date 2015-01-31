/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/
// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

// project includes
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>


#define PRECISION_z



/**
    Purpose
    -------

    This is an interface to the left solve for any custom preconditioner. 
    It should compute x = FUNCTION(b)
    The vectors are located on the device.

    Arguments
    ---------

    @param[in]
    b           magma_z_vector
                RHS

    @param[in,out]
    x           magma_z_vector*
                vector to precondition

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zapplycustomprecond_l(
    magma_z_vector b, 
    magma_z_vector *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    // vector access via x.dval, y->dval
    
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    This is an interface to the right solve for any custom preconditioner. 
    It should compute x = FUNCTION(b)
    The vectors are located on the device.

    Arguments
    ---------

    @param[in]
    b           magma_z_vector
                RHS

    @param[in,out]
    x           magma_z_vector*
                vector to precondition

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zapplycustomprecond_r(
    magma_z_vector b, 
    magma_z_vector *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    // vector access via x.dval, y->dval
    // sizes are x.num_rows, x.num_cols
    
    
    return MAGMA_SUCCESS;
}





