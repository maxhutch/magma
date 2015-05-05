/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @generated from magma_z_precond_wrapper.cpp normal z -> d, Mon May  4 11:57:23 2015
       @author Hartwig Anzt

*/
#include "common_magmasparse.h"




/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is chosen. It approximates x for A x = y.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                sparse matrix A

    @param[in]
    b           magma_d_matrix
                input vector b

    @param[in]
    x           magma_d_matrix*
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
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set up precond parameters as solver parameters
    magma_d_solver_par psolver_par;
    psolver_par.epsilon = precond->epsilon;
    psolver_par.maxiter = precond->maxiter;
    psolver_par.restart = precond->restart;
    psolver_par.verbose = 0;
    magma_d_preconditioner pprecond;
    pprecond.solver = Magma_NONE;

    switch( precond->solver ) {
        case  Magma_CG:
                CHECK( magma_dcg_res( A, b, x, &psolver_par, queue )); break;
        case  Magma_BICGSTAB:
                CHECK( magma_dbicgstab( A, b, x, &psolver_par, queue )); break;
        case  Magma_GMRES:
                CHECK( magma_dfgmres( A, b, x, &psolver_par, &pprecond, queue )); break;
        case  Magma_JACOBI:
                CHECK( magma_djacobi( A, b, x, &psolver_par, queue )); break;
        case  Magma_BAITER:
                CHECK( magma_dbaiter( A, b, x, &psolver_par, queue )); break;
        default:
                CHECK( magma_dcg_res( A, b, x, &psolver_par, queue )); break;

    }
cleanup:
    return info;
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
    A           magma_d_matrix
                sparse matrix A

    @param[in]
    b           magma_d_matrix
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
    magma_d_matrix A, magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    
    // make sure RHS is a dense matrix
    if ( b.storage_type != Magma_DENSE ) {
        printf( "error: sparse RHS not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

    if ( precond->solver == Magma_JACOBI ) {
        return magma_djacobisetup_diagscal( A, &(precond->d), queue );
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //return magma_dpastixsetup( A, b, precond, queue );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    else if ( precond->solver == Magma_ILU ) {
        return magma_dcumilusetup( A, precond, queue );
    }
    else if ( precond->solver == Magma_ICC ) {
        return magma_dcumiccsetup( A, precond, queue );
    }
    else if ( precond->solver == Magma_AICC ) {
        return magma_ditericsetup( A, b, precond, queue );
    }
    else if ( precond->solver == Magma_AILU ) {
        return magma_diterilusetup( A, b, precond, queue );
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
    A           magma_d_matrix
                sparse matrix A

    @param[in]
    b           magma_d_matrix
                input vector b

    @param[in,out]
    x           magma_d_matrix*
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
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_d_matrix tmp={Magma_CSR};
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        CHECK( magma_djacobi_diagscal( A.num_rows, precond->d, b, x, queue ));
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //CHECK( magma_dapplypastix( b, x, precond, queue ));
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    else if ( precond->solver == Magma_ILU ) {
        CHECK( magma_dvinit( &tmp, Magma_DEV, A.num_rows, b.num_cols, MAGMA_D_ZERO, queue ));
    }
    else if ( precond->solver == Magma_ICC ) {
        CHECK( magma_dvinit( &tmp, Magma_DEV, A.num_rows, b.num_cols, MAGMA_D_ZERO, queue ));
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        magmablasSetKernelStream( orig_queue );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    magma_dmfree( &tmp, queue );
    magmablasSetKernelStream( orig_queue );
    return info;
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
    A           magma_d_matrix
                sparse matrix A

    @param[in]
    b           magma_d_matrix
                input vector b

    @param[in,out]
    x           magma_d_matrix*
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
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        CHECK( magma_djacobi_diagscal( A.num_rows, precond->d, b, x, queue ));
    }
    else if ( ( precond->solver == Magma_ILU ||
                precond->solver == Magma_AILU ) && precond->maxiter >= 50 ) {
        CHECK( magma_dapplycumilu_l( b, x, precond, queue ));
    }
    else if ( ( precond->solver == Magma_ILU ||
                precond->solver == Magma_AILU ) && precond->maxiter < 50 ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols );
        magma_d_solver_par solver_par;
        solver_par.maxiter = precond->maxiter;
        //magma_djacobiiter_sys( precond->L, b, precond->d, precond->work1, x, &solver_par, queue );
        CHECK( magma_djacobispmvupdate(precond->maxiter, precond->L, precond->work1, b, precond->d, x, queue ));
    }
    else if ( ( precond->solver == Magma_ICC ||
                precond->solver == Magma_AICC ) && precond->maxiter >= 50 )  {
        CHECK( magma_dapplycumicc_l( b, x, precond, queue ));
    }
    else if ( ( precond->solver == Magma_ICC ||
                precond->solver == Magma_AICC ) && precond->maxiter < 50 )  {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols );
        magma_d_solver_par solver_par;
        solver_par.maxiter = precond->maxiter;
        //magma_djacobiiter_sys( precond->L, b, precond->d, precond->work1, x, &solver_par, queue );
        CHECK( magma_djacobispmvupdate(precond->maxiter, precond->L, precond->work1, b, precond->d, x, queue ));
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else if ( precond->solver == Magma_FUNCTION ) {
        CHECK( magma_dapplycustomprecond_l( b, x, precond, queue ));
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        magmablasSetKernelStream( orig_queue );
        info = MAGMA_ERR_NOT_SUPPORTED; 
    }
cleanup:
    magmablasSetKernelStream( orig_queue );
    return info;
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
    A           magma_d_matrix
                sparse matrix A

    @param[in]
    b           magma_d_matrix
                input vector b

    @param[in,out]
    x           magma_d_matrix*
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
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );    // x = b
    }
    else if ( ( precond->solver == Magma_ILU ||
                precond->solver == Magma_AILU ) && precond->maxiter >= 50 ) {
        CHECK( magma_dapplycumilu_r( b, x, precond, queue ));
    }
    else if ( ( precond->solver == Magma_ILU ||
                precond->solver == Magma_AILU ) && precond->maxiter < 50 ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols );
        magma_d_solver_par solver_par;
        solver_par.maxiter = precond->maxiter;
        //magma_djacobiiter_sys( precond->U, b, precond->d2, precond->work2, x, &solver_par, queue );
        CHECK( magma_djacobispmvupdate(precond->maxiter, precond->U, precond->work2, b, precond->d2, x, queue ));
    }

    else if ( ( precond->solver == Magma_ICC ||
                precond->solver == Magma_AICC ) && precond->maxiter >= 50 ) {
        CHECK( magma_dapplycumicc_r( b, x, precond, queue ));
    }
    else if ( ( precond->solver == Magma_ICC ||
               precond->solver == Magma_AICC ) && precond->maxiter < 50 ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols );
        magma_d_solver_par solver_par;
        solver_par.maxiter = precond->maxiter;
        //magma_djacobiiter_sys( precond->U, b, precond->d2, precond->work2, x, &solver_par, queue );
        CHECK( magma_djacobispmvupdate(precond->maxiter, precond->U, precond->work2, b, precond->d2, x, queue ));
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_dcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else if ( precond->solver == Magma_FUNCTION ) {
        CHECK( magma_dapplycustomprecond_r( b, x, precond, queue ));
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        magmablasSetKernelStream( orig_queue );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    magmablasSetKernelStream( orig_queue );
    return info;
}


