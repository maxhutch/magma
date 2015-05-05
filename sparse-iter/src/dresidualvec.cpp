/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @generated from zresidualvec.cpp normal z -> d, Sun May  3 11:22:59 2015
       @author Hartwig Anzt

*/
#include "common_magmasparse.h"

#define  r(i_)  (r->dval)+i_*dofs
#define  b(i_)  (b.dval)+i_*dofs

/**
    Purpose
    -------

    Computes the residual r = b-Ax for a solution approximation x.
    It returns both, the actual residual and the residual vector

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A

    @param[in]
    b           magma_d_matrix
                RHS b

    @param[in]
    x           magma_d_matrix
                solution approximation

    @param[in,out]
    r           magma_d_matrix*
                residual vector 
                
    @param[out]
    res         double*
                return residual 

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dresidualvec(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix x,
    magma_d_matrix *r, double *res,
    magma_queue_t queue )
{
    magma_int_t info =0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    // some useful variables
    double zero = MAGMA_D_ZERO, one = MAGMA_D_ONE,
                                            mone = MAGMA_D_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    
    if ( A.num_rows == b.num_rows ) {

        CHECK( magma_d_spmv( mone, A, x, zero, *r, queue ));      // r = A x
        magma_daxpy(dofs, one, b.dval, 1, r->dval, 1);          // r = r - b
        *res =  magma_dnrm2(dofs, r->dval, 1);            // res = ||r||
        //               /magma_dnrm2(dofs, b.dval, 1);               /||b||
        //printf( "relative residual: %e\n", *res );

    } else if ((b.num_rows*b.num_cols)%A.num_rows== 0 ) {
        magma_int_t num_vecs = b.num_rows*b.num_cols/A.num_rows;

        CHECK( magma_d_spmv( mone, A, x, zero, *r, queue ));           // r = A x

        for( magma_int_t i=0; i<num_vecs; i++) {
            magma_daxpy(dofs, one, b(i), 1, r(i), 1);   // r = r - b
            res[i] =  magma_dnrm2(dofs, r(i), 1);        // res = ||r||
        }
        //               /magma_dnrm2(dofs, b.dval, 1);               /||b||
        //printf( "relative residual: %e\n", *res );
    } else {
        printf("error: dimensions do not match.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
    }

cleanup:
    magmablasSetKernelStream( orig_queue );
    return info;
}

