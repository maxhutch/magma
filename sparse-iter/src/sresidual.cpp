/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zresidual.cpp normal z -> s, Fri May 30 10:41:41 2014
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Computes the residual ||b-Ax|| for a solution approximation x.

    Arguments
    =========

    magma_s_sparse_matrix A                   input matrix A
    magma_s_vector b                          RHS b
    magma_s_vector x                          solution approximation
    float *res                   return residual

    ========================================================================  */


magma_int_t
magma_sresidual( magma_s_sparse_matrix A, magma_s_vector b, magma_s_vector x, 
                 float *res ){

    // some useful variables
    float zero = MAGMA_S_ZERO, one = MAGMA_S_ONE, 
                                            mone = MAGMA_S_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    
    
    magma_s_vector r;
    magma_s_vinit( &r, Magma_DEV, A.num_rows, zero );

    magma_s_spmv( one, A, x, zero, r );                   // r = A x
    magma_saxpy(dofs, mone, b.val, 1, r.val, 1);          // r = r - b
    *res =  magma_snrm2(dofs, r.val, 1);            // res = ||r||
    //               /magma_snrm2(dofs, b.val, 1);               /||b||
    //printf( "relative residual: %e\n", *res );

    magma_s_vfree(&r);

    return MAGMA_SUCCESS;
}


