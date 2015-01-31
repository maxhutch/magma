/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zresidual.cpp normal z -> d, Fri Jan 30 19:00:30 2015
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

#define  r(i)  r.dval+i*dofs
#define  b(i)  b.dval+i*dofs

/**
    Purpose
    -------

    Computes the residual ||b-Ax|| for a solution approximation x.

    Arguments
    ---------

    @param[in]
    A           magma_d_sparse_matrix
                input matrix A

    @param[in]
    b           magma_d_vector
                RHS b

    @param[in]
    x           magma_d_vector
                solution approximation

    @param[out]
    res         double*
                return residual

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dresidual(
    magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector x, 
    double *res,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // some useful variables
    double zero = MAGMA_D_ZERO, one = MAGMA_D_ONE, 
                                            mone = MAGMA_D_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    
    if ( A.num_rows == b.num_rows ) {
        magma_d_vector r;
        magma_d_vinit( &r, Magma_DEV, A.num_rows, zero, queue );

        magma_d_spmv( one, A, x, zero, r, queue );                   // r = A x
        magma_daxpy(dofs, mone, b.dval, 1, r.dval, 1);          // r = r - b
        *res =  magma_dnrm2(dofs, r.dval, 1);            // res = ||r||
        //               /magma_dnrm2(dofs, b.dval, 1);               /||b||
        //printf( "relative residual: %e\n", *res );

        magma_d_vfree(&r, queue );
    } else if (b.num_rows%A.num_rows== 0 ) {
        magma_int_t num_vecs = b.num_rows/A.num_rows;

        magma_d_vector r;
        magma_d_vinit( &r, Magma_DEV, b.num_rows, zero, queue );

        magma_d_spmv( one, A, x, zero, r, queue );                   // r = A x

        for( magma_int_t i=0; i<num_vecs; i++) {
            magma_daxpy(dofs, mone, b(i), 1, r(i), 1);   // r = r - b
            res[i] =  magma_dnrm2(dofs, r(i), 1);        // res = ||r||
        }
        //               /magma_dnrm2(dofs, b.dval, 1);               /||b||
        //printf( "relative residual: %e\n", *res );

        magma_d_vfree(&r, queue );
    } else {
        printf("error: dimensions do not match.\n");
    }
    
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}

