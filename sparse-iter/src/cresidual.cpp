/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/src/zresidual.cpp normal z -> c, Mon May  2 23:31:04 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define  r(i_)  (r.dval + (i_)*dofs)
#define  b(i_)  (b.dval + (i_)*dofs)

/**
    Purpose
    -------

    Computes the residual ||b-Ax|| for a solution approximation x.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A

    @param[in]
    b           magma_c_matrix
                RHS b

    @param[in]
    x           magma_c_matrix
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
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix x,
    float *res,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // constants
    const magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    
    // some useful variables
    magma_int_t dofs = A.num_rows;
    magma_int_t num_vecs = b.num_rows*b.num_cols/A.num_rows;
    
    magma_c_matrix r = {Magma_CSR};
    
    if ( A.num_rows == b.num_rows ) {
        CHECK( magma_cvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

        CHECK( magma_c_spmv( c_one, A, x, c_zero, r, queue ));        // r = A x
        magma_caxpy( dofs, c_neg_one, b.dval, 1, r.dval, 1, queue );  // r = r - b
        *res = magma_scnrm2( dofs, r.dval, 1, queue );                // res = ||r||
    } else if ((b.num_rows*b.num_cols)%A.num_rows == 0 ) {
        CHECK( magma_cvinit( &r, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));

        CHECK( magma_c_spmv( c_one, A, x, c_zero, r, queue ));        // r = A x

        for( magma_int_t i=0; i < num_vecs; i++) {
            magma_caxpy( dofs, c_neg_one, b(i), 1, r(i), 1, queue );  // r = r - b
            res[i] = magma_scnrm2( dofs, r(i), 1, queue );            // res = ||r||
        }
    } else {
        printf("%%error: dimensions do not match.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    
cleanup:
    magma_cmfree( &r, queue );
    return info;
}




/**
    Purpose
    -------

    Computes the residual r=||b-Ax|| for the slice r(start:end) 
    for a solution approximation x.

    Arguments
    ---------

    @param[in]          
    start       magma_int_t
                start of slice (row-index)
                
    @param[in]          
    end         magma_int_t
                end of slice (row-index)
                
    @param[in]
    A           magma_c_matrix
                input matrix A

    @param[in]
    b           magma_c_matrix
                RHS b

    @param[in]
    x           magma_c_matrix
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
magma_cresidual_slice(
    magma_int_t start, magma_int_t end,
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix x,
    float *res,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // constants
    const magmaFloatComplex c_zero    = MAGMA_C_ZERO;
    const magmaFloatComplex c_one     = MAGMA_C_ONE;
    const magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    
    // some useful variables
    magma_int_t dofs = A.num_rows;
    magma_int_t num_vecs = b.num_rows*b.num_cols/A.num_rows;
    
    magma_c_matrix r = {Magma_CSR};
    
    if ( A.num_rows == b.num_rows ) {
        CHECK( magma_cvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

        CHECK( magma_c_spmv( c_one, A, x, c_zero, r, queue ));        // r = A x
        magma_caxpy( dofs, c_neg_one, b.dval, 1, r.dval, 1, queue );  // r = r - b
        *res = magma_scnrm2( end-start, r.dval+start, 1, queue );                // res = ||r(start:end)||
    } else if ((b.num_rows*b.num_cols)%A.num_rows == 0 ) {
        CHECK( magma_cvinit( &r, Magma_DEV, b.num_rows, b.num_cols, c_zero, queue ));

        CHECK( magma_c_spmv( c_one, A, x, c_zero, r, queue ));        // r = A x

        for( magma_int_t i=0; i < num_vecs; i++) {
            magma_caxpy( dofs, c_neg_one, b(i), 1, r(i), 1, queue );  // r = r - b
            res[i] = magma_scnrm2( end-start, r(i)+start, 1, queue );            // res = ||r(start:end)||
        }
    } else {
        printf("error: dimensions do not match.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    
cleanup:
    magma_cmfree( &r, queue );
    return info;
}
