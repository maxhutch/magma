/*
    -- MAGMA (version 2.1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date August 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zdummy.cpp, normal z -> d, Tue Aug 30 09:38:59 2016
*/
#include "magmasparse_internal.h"

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_d


/**
    Purpose
    -------

    Prepares the iterative threshold Incomplete Cholesky preconditioner.
    
    This function requires OpenMP, and is only available if OpenMP is activated. 

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A
                
    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_diterictsetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    printf("error: not yet released\n");

    return info;
}
