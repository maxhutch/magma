/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from zresidual.cpp normal z -> c, Sat Nov 15 19:54:22 2014
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
    A           magma_c_sparse_matrix
                input matrix A

    @param[in]
    b           magma_c_vector
                RHS b

    @param[in]
    x           magma_c_vector
                solution approximation

    @param[out]
    res         magmaFloatComplex*
                return residual

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cresidual(
    magma_c_sparse_matrix A, magma_c_vector b, magma_c_vector x, 
    float *res,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // some useful variables
    magmaFloatComplex zero = MAGMA_C_ZERO, one = MAGMA_C_ONE, 
                                            mone = MAGMA_C_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    
    if ( A.num_rows == b.num_rows ) {
        magma_c_vector r;
        magma_c_vinit( &r, Magma_DEV, A.num_rows, zero, queue );

        magma_c_spmv( one, A, x, zero, r, queue );                   // r = A x
        magma_caxpy(dofs, mone, b.dval, 1, r.dval, 1);          // r = r - b
        *res =  magma_scnrm2(dofs, r.dval, 1);            // res = ||r||
        //               /magma_scnrm2(dofs, b.dval, 1);               /||b||
        //printf( "relative residual: %e\n", *res );

        magma_c_vfree(&r, queue );
    } else if (b.num_rows%A.num_rows== 0 ) {
        magma_int_t num_vecs = b.num_rows/A.num_rows;

        magma_c_vector r;
        magma_c_vinit( &r, Magma_DEV, b.num_rows, zero, queue );

        magma_c_spmv( one, A, x, zero, r, queue );                   // r = A x

        for( magma_int_t i=0; i<num_vecs; i++) {
            magma_caxpy(dofs, mone, b(i), 1, r(i), 1);   // r = r - b
            res[i] =  magma_scnrm2(dofs, r(i), 1);        // res = ||r||
        }
        //               /magma_scnrm2(dofs, b.dval, 1);               /||b||
        //printf( "relative residual: %e\n", *res );

        magma_c_vfree(&r, queue );
    } else {
        printf("error: dimensions do not match.\n");
    }
    
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}

