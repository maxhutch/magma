/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
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
    A           magma_z_matrix
                sparse matrix A

    @param[in]
    b           magma_z_matrix
                input vector b

    @param[in]
    x           magma_z_matrix*
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
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set up precond parameters as solver parameters
    magma_z_solver_par psolver_par;
    psolver_par.epsilon = precond->epsilon;
    psolver_par.maxiter = precond->maxiter;
    psolver_par.restart = precond->restart;
    psolver_par.verbose = 0;
    magma_z_preconditioner pprecond;
    pprecond.solver = Magma_NONE;

    switch( precond->solver ) {
        case  Magma_CG:
                CHECK( magma_zcg_res( A, b, x, &psolver_par, queue )); break;
        case  Magma_BICGSTAB:
                CHECK( magma_zbicgstab( A, b, x, &psolver_par, queue )); break;
        case  Magma_GMRES:
                CHECK( magma_zfgmres( A, b, x, &psolver_par, &pprecond, queue )); break;
        case  Magma_JACOBI:
                CHECK( magma_zjacobi( A, b, x, &psolver_par, queue )); break;
        case  Magma_BAITER:
                CHECK( magma_zbaiter( A, b, x, &psolver_par, queue )); break;
        default:
                CHECK( magma_zcg_res( A, b, x, &psolver_par, queue )); break;

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
    A           magma_z_matrix
                sparse matrix A

    @param[in]
    b           magma_z_matrix
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
    magma_z_matrix A, magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    
    // make sure RHS is a dense matrix
    if ( b.storage_type != Magma_DENSE ) {
        printf( "error: sparse RHS not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

    if ( precond->solver == Magma_JACOBI ) {
        return magma_zjacobisetup_diagscal( A, &(precond->d), queue );
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //return magma_zpastixsetup( A, b, precond, queue );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    else if ( precond->solver == Magma_ILU ) {
        return magma_zcumilusetup( A, precond, queue );
    }
    else if ( precond->solver == Magma_ICC ) {
        return magma_zcumiccsetup( A, precond, queue );
    }
    else if ( precond->solver == Magma_AICC ) {
        return magma_zitericsetup( A, b, precond, queue );
    }
    else if ( precond->solver == Magma_AILU ) {
        return magma_ziterilusetup( A, b, precond, queue );
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
    A           magma_z_matrix
                sparse matrix A

    @param[in]
    b           magma_z_matrix
                input vector b

    @param[in,out]
    x           magma_z_matrix*
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
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix tmp={Magma_CSR};
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        CHECK( magma_zjacobi_diagscal( A.num_rows, precond->d, b, x, queue ));
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //CHECK( magma_zapplypastix( b, x, precond, queue ));
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    else if ( precond->solver == Magma_ILU ) {
        CHECK( magma_zvinit( &tmp, Magma_DEV, A.num_rows, b.num_cols, MAGMA_Z_ZERO, queue ));
    }
    else if ( precond->solver == Magma_ICC ) {
        CHECK( magma_zvinit( &tmp, Magma_DEV, A.num_rows, b.num_cols, MAGMA_Z_ZERO, queue ));
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        magmablasSetKernelStream( orig_queue );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    magma_zmfree( &tmp, queue );
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
    A           magma_z_matrix
                sparse matrix A

    @param[in]
    b           magma_z_matrix
                input vector b

    @param[in,out]
    x           magma_z_matrix*
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
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        CHECK( magma_zjacobi_diagscal( A.num_rows, precond->d, b, x, queue ));
    }
    else if ( ( precond->solver == Magma_ILU ||
                precond->solver == Magma_AILU ) && precond->maxiter >= 50 ) {
        CHECK( magma_zapplycumilu_l( b, x, precond, queue ));
    }
    else if ( ( precond->solver == Magma_ILU ||
                precond->solver == Magma_AILU ) && precond->maxiter < 50 ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols );
        magma_z_solver_par solver_par;
        solver_par.maxiter = precond->maxiter;
        //magma_zjacobiiter_sys( precond->L, b, precond->d, precond->work1, x, &solver_par, queue );
        CHECK( magma_zjacobispmvupdate(precond->maxiter, precond->L, precond->work1, b, precond->d, x, queue ));
    }
    else if ( ( precond->solver == Magma_ICC ||
                precond->solver == Magma_AICC ) && precond->maxiter >= 50 )  {
        CHECK( magma_zapplycumicc_l( b, x, precond, queue ));
    }
    else if ( ( precond->solver == Magma_ICC ||
                precond->solver == Magma_AICC ) && precond->maxiter < 50 )  {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols );
        magma_z_solver_par solver_par;
        solver_par.maxiter = precond->maxiter;
        //magma_zjacobiiter_sys( precond->L, b, precond->d, precond->work1, x, &solver_par, queue );
        CHECK( magma_zjacobispmvupdate(precond->maxiter, precond->L, precond->work1, b, precond->d, x, queue ));
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else if ( precond->solver == Magma_FUNCTION ) {
        CHECK( magma_zapplycustomprecond_l( b, x, precond, queue ));
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
    A           magma_z_matrix
                sparse matrix A

    @param[in]
    b           magma_z_matrix
                input vector b

    @param[in,out]
    x           magma_z_matrix*
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
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    if ( precond->solver == Magma_JACOBI ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );    // x = b
    }
    else if ( ( precond->solver == Magma_ILU ||
                precond->solver == Magma_AILU ) && precond->maxiter >= 50 ) {
        CHECK( magma_zapplycumilu_r( b, x, precond, queue ));
    }
    else if ( ( precond->solver == Magma_ILU ||
                precond->solver == Magma_AILU ) && precond->maxiter < 50 ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols );
        magma_z_solver_par solver_par;
        solver_par.maxiter = precond->maxiter;
        //magma_zjacobiiter_sys( precond->U, b, precond->d2, precond->work2, x, &solver_par, queue );
        CHECK( magma_zjacobispmvupdate(precond->maxiter, precond->U, precond->work2, b, precond->d2, x, queue ));
    }

    else if ( ( precond->solver == Magma_ICC ||
                precond->solver == Magma_AICC ) && precond->maxiter >= 50 ) {
        CHECK( magma_zapplycumicc_r( b, x, precond, queue ));
    }
    else if ( ( precond->solver == Magma_ICC ||
               precond->solver == Magma_AICC ) && precond->maxiter < 50 ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, b.num_cols, x->dval, b.num_cols );
        magma_z_solver_par solver_par;
        solver_par.maxiter = precond->maxiter;
        //magma_zjacobiiter_sys( precond->U, b, precond->d2, precond->work2, x, &solver_par, queue );
        CHECK( magma_zjacobispmvupdate(precond->maxiter, precond->U, precond->work2, b, precond->d2, x, queue ));
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1 );      //  x = b
    }
    else if ( precond->solver == Magma_FUNCTION ) {
        CHECK( magma_zapplycustomprecond_r( b, x, precond, queue ));
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


