/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from magma_z_precond_wrapper.cpp normal z -> d, Fri Jan 30 19:00:30 2015
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "magmasparse.h"




/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is chosen. It approximates x for A x = y.

    Arguments
    ---------

    @param[in]
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_d_vector
                input vector b     

    @param[in]
    x           magma_d_vector*
                output vector x        

    @param[in,out]
    precond     magma_d_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_d_precond(
    magma_d_sparse_matrix A, 
    magma_d_vector b, 
    magma_d_vector *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    // set up precond parameters as solver parameters   
    magma_d_solver_par psolver_par;
    psolver_par.epsilon = precond->epsilon;
    psolver_par.maxiter = precond->maxiter;
    psolver_par.restart = precond->restart;
    psolver_par.verbose = 0;

    switch( precond->solver ) {
        case  Magma_CG:
                magma_dcg_res( A, b, x, &psolver_par, queue );break;
        case  Magma_BICGSTAB:
                magma_dbicgstab( A, b, x, &psolver_par, queue );break;
        case  Magma_GMRES: 
                magma_dgmres( A, b, x, &psolver_par, queue );break;
        case  Magma_JACOBI: 
                magma_djacobi( A, b, x, &psolver_par, queue );break;
        case  Magma_BAITER: 
                magma_dbaiter( A, b, x, &psolver_par, queue );break;
    }
    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is preprocessed. 
    E.g. for Jacobi: the scaling-vetor, for ILU the factorization.

    Arguments
    ---------

    @param[in]
    A           magma_d_sparse_matrix
                sparse matrix A     

    @param[in]
    b           magma_d_vector
                input vector y      

    @param[in,out]
    precond     magma_d_preconditioner
                preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_d_precondsetup(
    magma_d_sparse_matrix A, magma_d_vector b, 
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    if ( precond->solver == Magma_JACOBI ) {
        magma_djacobisetup_diagscal( A, &(precond->d), queue );
        return MAGMA_SUCCESS;
    }
    else if ( precond->solver == Magma_PASTIX ) {
        magma_dpastixsetup( A, b, precond, queue );
        return MAGMA_SUCCESS;
    }
    else if ( precond->solver == Magma_ILU ) {
        magma_dcumilusetup( A, precond, queue );
        return MAGMA_SUCCESS;
    }
    else if ( precond->solver == Magma_ICC ) {
        magma_dcumiccsetup( A, precond, queue );
        return MAGMA_SUCCESS;
    }
    else if ( precond->solver == Magma_NONE ) {
        return MAGMA_SUCCESS;
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
}



/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is applied. 
    E.g. for Jacobi: the scaling-vetor, for ILU the triangular solves.

    Arguments
    ---------

    @param[in]
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_d_vector
                input vector b     

    @param[in,out]
    x           magma_d_vector*
                output vector x     

    @param[in]
    precond     magma_d_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_d_applyprecond(
    magma_d_sparse_matrix A, 
    magma_d_vector b, 
    magma_d_vector *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        magma_djacobi_diagscal( A.num_rows, precond->d, b, x, queue );
    }
    else if ( precond->solver == Magma_PASTIX ) {
        magma_dapplypastix( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_ILU ) {
        magma_d_vector tmp;
        magma_d_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_D_ZERO, queue );
        magma_d_vfree( &tmp, queue );
    }
    else if ( precond->solver == Magma_ICC ) {
        magma_d_vector tmp;
        magma_d_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_D_ZERO, queue );
        magma_d_vfree( &tmp, queue );
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_dcopy( b.num_rows, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        magmablasSetKernelStream( orig_queue );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective left preconditioner
    is applied. 
    E.g. for Jacobi: the scaling-vetor, for ILU the left triangular solve.

    Arguments
    ---------

    @param[in]
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_d_vector
                input vector b     

    @param[in,out]
    x           magma_d_vector*
                output vector x     

    @param[in]
    precond     magma_d_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_d_applyprecond_left(
    magma_d_sparse_matrix A, 
    magma_d_vector b, 
    magma_d_vector *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        magma_djacobi_diagscal( A.num_rows, precond->d, b, x, queue );
    }
    else if ( precond->solver == Magma_ILU || 
            ( precond->solver == Magma_AILU && precond->maxiter == -1) ) {
        magma_dapplycumilu_l( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_ICC ) {
        magma_dapplycumicc_l( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else if ( precond->solver == Magma_FUNCTION ) {
        magma_dapplycustomprecond_l( b, x, precond, queue );     
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        magmablasSetKernelStream( orig_queue );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective right-preconditioner
    is applied. 
    E.g. for Jacobi: the scaling-vetor, for ILU the right triangular solve.

    Arguments
    ---------

    @param[in]
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_d_vector
                input vector b     

    @param[in,out]
    x           magma_d_vector*
                output vector x     

    @param[in]
    precond     magma_d_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_d_applyprecond_right(
    magma_d_sparse_matrix A, 
    magma_d_vector b, 
    magma_d_vector *x, 
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );    // x = b
    }
    else if ( precond->solver == Magma_ILU || 
            ( precond->solver == Magma_AILU && precond->maxiter == -1)) {
        magma_dapplycumilu_r( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_ICC || 
            ( precond->solver == Magma_AICC && precond->maxiter == -1) ) {
        magma_dapplycumicc_r( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else if ( precond->solver == Magma_FUNCTION ) {
        magma_dapplycustomprecond_r( b, x, precond, queue );     
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        magmablasSetKernelStream( orig_queue );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}


