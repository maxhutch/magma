/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zresidual.cpp normal z -> d, Fri Jul 18 17:34:29 2014
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

/**
    Purpose
    -------

    Computes the residual ||b-Ax|| for a solution approximation x.

    Arguments
    ---------

    @param
    A           magma_d_sparse_matrix
                input matrix A

    @param
    b           magma_d_vector
                RHS b

    @param
    x           magma_d_vector
                solution approximation

    @param
    res         double*
                return residual


    @ingroup magmasparse_daux
    ********************************************************************/

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


