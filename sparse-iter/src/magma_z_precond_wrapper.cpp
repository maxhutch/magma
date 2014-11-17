/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @precisions normal z -> c d s
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
    A           magma_z_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_z_vector
                input vector b     

    @param[in]
    x           magma_z_vector*
                output vector x        

    @param[in,out]
    precond     magma_z_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_precond(
    magma_z_sparse_matrix A, 
    magma_z_vector b, 
    magma_z_vector *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    // set up precond parameters as solver parameters   
    magma_z_solver_par psolver_par;
    psolver_par.epsilon = precond->epsilon;
    psolver_par.maxiter = precond->maxiter;
    psolver_par.restart = precond->restart;
    psolver_par.verbose = 0;

    switch( precond->solver ) {
        case  Magma_CG:
                magma_zcg_res( A, b, x, &psolver_par, queue );break;
        case  Magma_BICGSTAB:
                magma_zbicgstab( A, b, x, &psolver_par, queue );break;
        case  Magma_GMRES: 
                magma_zgmres( A, b, x, &psolver_par, queue );break;
        case  Magma_JACOBI: 
                magma_zjacobi( A, b, x, &psolver_par, queue );break;
        case  Magma_BAITER: 
                magma_zbaiter( A, b, x, &psolver_par, queue );break;
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
    A           magma_z_sparse_matrix
                sparse matrix A     

    @param[in]
    b           magma_z_vector
                input vector y      

    @param[in,out]
    precond     magma_z_preconditioner
                preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_precondsetup(
    magma_z_sparse_matrix A, magma_z_vector b, 
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    if ( precond->solver == Magma_JACOBI ) {
        magma_zjacobisetup_diagscal( A, &(precond->d), queue );
        return MAGMA_SUCCESS;
    }
    else if ( precond->solver == Magma_PASTIX ) {
        magma_zpastixsetup( A, b, precond, queue );
        return MAGMA_SUCCESS;
    }
    else if ( precond->solver == Magma_ILU ) {
        magma_zcumilusetup( A, precond, queue );
        return MAGMA_SUCCESS;
    }
    else if ( precond->solver == Magma_ICC ) {
        magma_zcumiccsetup( A, precond, queue );
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
    A           magma_z_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_z_vector
                input vector b     

    @param[in,out]
    x           magma_z_vector*
                output vector x     

    @param[in]
    precond     magma_z_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_applyprecond(
    magma_z_sparse_matrix A, 
    magma_z_vector b, 
    magma_z_vector *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        magma_zjacobi_diagscal( A.num_rows, precond->d, b, x, queue );
    }
    else if ( precond->solver == Magma_PASTIX ) {
        magma_zapplypastix( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_ILU ) {
        magma_z_vector tmp;
        magma_z_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_Z_ZERO, queue );
        magma_z_vfree( &tmp, queue );
    }
    else if ( precond->solver == Magma_ICC ) {
        magma_z_vector tmp;
        magma_z_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_Z_ZERO, queue );
        magma_z_vfree( &tmp, queue );
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_zcopy( b.num_rows, b.dval, 1, x->dval, 1 );      //  x = b
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
    A           magma_z_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_z_vector
                input vector b     

    @param[in,out]
    x           magma_z_vector*
                output vector x     

    @param[in]
    precond     magma_z_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_applyprecond_left(
    magma_z_sparse_matrix A, 
    magma_z_vector b, 
    magma_z_vector *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        magma_zjacobi_diagscal( A.num_rows, precond->d, b, x, queue );
    }
    else if ( precond->solver == Magma_ILU || 
            ( precond->solver == Magma_AILU && precond->maxiter == -1) ) {
        magma_zapplycumilu_l( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_ICC ) {
        magma_zapplycumicc_l( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
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
    A           magma_z_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_z_vector
                input vector b     

    @param[in,out]
    x           magma_z_vector*
                output vector x     

    @param[in]
    precond     magma_z_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_applyprecond_right(
    magma_z_sparse_matrix A, 
    magma_z_vector b, 
    magma_z_vector *x, 
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );    // x = b
    }
    else if ( precond->solver == Magma_ILU || 
            ( precond->solver == Magma_AILU && precond->maxiter == -1)) {
        magma_zapplycumilu_r( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_ICC || 
            ( precond->solver == Magma_AICC && precond->maxiter == -1) ) {
        magma_zapplycumicc_r( b, x, precond, queue );
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        magmablasSetKernelStream( orig_queue );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}


