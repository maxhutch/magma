/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zresidual.cpp normal z -> c, Fri May 30 10:41:42 2014
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

    magma_c_sparse_matrix A                   input matrix A
    magma_c_vector b                          RHS b
    magma_c_vector x                          solution approximation
    magmaFloatComplex *res                   return residual

    ========================================================================  */


magma_int_t
magma_cresidual( magma_c_sparse_matrix A, magma_c_vector b, magma_c_vector x, 
                 float *res ){

    // some useful variables
    magmaFloatComplex zero = MAGMA_C_ZERO, one = MAGMA_C_ONE, 
                                            mone = MAGMA_C_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    
    
    magma_c_vector r;
    magma_c_vinit( &r, Magma_DEV, A.num_rows, zero );

    magma_c_spmv( one, A, x, zero, r );                   // r = A x
    magma_caxpy(dofs, mone, b.val, 1, r.val, 1);          // r = r - b
    *res =  magma_scnrm2(dofs, r.val, 1);            // res = ||r||
    //               /magma_scnrm2(dofs, b.val, 1);               /||b||
    //printf( "relative residual: %e\n", *res );

    magma_c_vfree(&r);

    return MAGMA_SUCCESS;
}


