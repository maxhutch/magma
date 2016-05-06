/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zdummy.cpp normal z -> c, Mon May  2 23:31:02 2016
*/
#include "magmasparse_internal.h"

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_c


/**
    Purpose
    -------

    Prepares the iterative threshold Incomplete Cholesky preconditioner.
    
    This function requires OpenMP, and is only available if OpenMP is activated. 

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A
                
    @param[in]
    b           magma_c_matrix
                input RHS b

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_citerictsetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{

    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
    
    

