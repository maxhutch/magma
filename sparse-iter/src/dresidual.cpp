/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zresidual.cpp normal z -> d, Fri May 30 10:41:41 2014
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

    magma_d_sparse_matrix A                   input matrix A
    magma_d_vector b                          RHS b
    magma_d_vector x                          solution approximation
    double *res                   return residual

    ========================================================================  */


magma_int_t
magma_dresidual( magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector x, 
                 double *res ){

    // some useful variables
    double zero = MAGMA_D_ZERO, one = MAGMA_D_ONE, 
                                            mone = MAGMA_D_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    
    
    magma_d_vector r;
    magma_d_vinit( &r, Magma_DEV, A.num_rows, zero );

    magma_d_spmv( one, A, x, zero, r );                   // r = A x
    magma_daxpy(dofs, mone, b.val, 1, r.val, 1);          // r = r - b
    *res =  magma_dnrm2(dofs, r.val, 1);            // res = ||r||
    //               /magma_dnrm2(dofs, b.val, 1);               /||b||
    //printf( "relative residual: %e\n", *res );

    magma_d_vfree(&r);

    return MAGMA_SUCCESS;
}


