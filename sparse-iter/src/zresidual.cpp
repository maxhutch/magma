/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @precisions normal z -> s d c
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

    magma_z_sparse_matrix A                   input matrix A
    magma_z_vector b                          RHS b
    magma_z_vector x                          solution approximation
    magmaDoubleComplex *res                   return residual

    ========================================================================  */


magma_int_t
magma_zresidual( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector x, 
                 double *res ){

    // some useful variables
    magmaDoubleComplex zero = MAGMA_Z_ZERO, one = MAGMA_Z_ONE, 
                                            mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    
    
    magma_z_vector r;
    magma_z_vinit( &r, Magma_DEV, A.num_rows, zero );

    magma_z_spmv( one, A, x, zero, r );                   // r = A x
    magma_zaxpy(dofs, mone, b.val, 1, r.val, 1);          // r = r - b
    *res =  magma_dznrm2(dofs, r.val, 1);            // res = ||r||
    //               /magma_dznrm2(dofs, b.val, 1);               /||b||
    //printf( "relative residual: %e\n", *res );

    magma_z_vfree(&r);

    return MAGMA_SUCCESS;
}


