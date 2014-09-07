/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

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

    @param
    A           magma_z_sparse_matrix
                sparse matrix A    

    @param
    b           magma_z_vector
                input vector b     

    @param
    x           magma_z_vector*
                output vector x        

    @param
    precond     magma_z_preconditioner
                preconditioner

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_z_precond( magma_z_sparse_matrix A, magma_z_vector b, 
                 magma_z_vector *x, magma_z_preconditioner *precond )
{
// set up precond parameters as solver parameters   
    magma_z_solver_par psolver_par;
    psolver_par.epsilon = precond->epsilon;
    psolver_par.maxiter = precond->maxiter;
    psolver_par.restart = precond->restart;
    psolver_par.verbose = 0;
   
    if( precond->solver == Magma_CG ){
// printf( "start CG preconditioner with epsilon: %f and maxiter: %d: ", 
//                            psolver_par.epsilon, psolver_par.maxiter );
        magma_zcg( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond->solver == Magma_GMRES ){
// printf( "start GMRES preconditioner with epsilon: %f and maxiter: %d: ", 
//                               psolver_par.epsilon, psolver_par.maxiter );
        magma_zgmres( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond->solver == Magma_BICGSTAB ){
// printf( "start BICGSTAB preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_zbicgstab( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond->solver == Magma_JACOBI ){
// printf( "start JACOBI preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_zjacobi( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond->solver == Magma_BAITER ){
// printf( "start BAITER preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_zbaiter( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond->solver == Magma_NONE ){
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

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

    @param
    A           magma_z_sparse_matrix
                sparse matrix A     

    @param
    b           magma_z_vector
                input vector y      

    @param
    precond     magma_z_preconditioner
                preconditioner

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_z_precondsetup( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_zjacobisetup_diagscal( A, &(precond->d) );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_PASTIX ){
        magma_zpastixsetup( A, b, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_zcuilusetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_zcuiccsetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_NONE ){
        return MAGMA_SUCCESS;
    }
    else{
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

    @param
    A           magma_z_sparse_matrix
                sparse matrix A    

    @param
    b           magma_z_vector
                input vector b     

    @param
    x           magma_z_vector*
                output vector x     

    @param
    precond     magma_z_preconditioner
                preconditioner

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_z_applyprecond( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_vector *x, magma_z_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_zjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_PASTIX ){
        magma_zapplypastix( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_z_vector tmp;
        magma_z_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_Z_MAKE(1.0, 0.0) );
     //   magma_zapplycuilu_l( b, &tmp, precond ); 
     //   magma_zapplycuilu_r( tmp, x, precond );
        magma_z_vfree( &tmp );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_z_vector tmp;
        magma_z_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_Z_MAKE(1.0, 0.0) );
       // magma_ztrisv_l_nu( precond->L, b, &tmp );
       // magma_ztrisv_r_nu( precond->L, tmp, x );
        magma_z_vfree( &tmp );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_NONE ){
        magma_zcopy( b.num_rows, b.val, 1, x->val, 1 );      //  x = b
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

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

    @param
    A           magma_z_sparse_matrix
                sparse matrix A    

    @param
    b           magma_z_vector
                input vector b     

    @param
    x           magma_z_vector*
                output vector x     

    @param
    precond     magma_z_preconditioner
                preconditioner

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_z_applyprecond_left( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_vector *x, magma_z_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_zjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_zapplycuilu_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_zapplycuicc_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_NONE ){
        magma_zcopy( b.num_rows, b.val, 1, x->val, 1 );      //  x = b
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

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

    @param
    A           magma_z_sparse_matrix
                sparse matrix A    

    @param
    b           magma_z_vector
                input vector b     

    @param
    x           magma_z_vector*
                output vector x  

    @param
    precond     magma_z_preconditioner
                preconditioner

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_z_applyprecond_right( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_vector *x, magma_z_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        //magma_zjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        magma_zcopy( b.num_rows, b.val, 1, x->val, 1 );    // x = b
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_zapplycuilu_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC || 
            ( precond->solver == Magma_AICC && precond->maxiter == -1) ){
        magma_zapplycuicc_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_NONE ){
        magma_zcopy( b.num_rows, b.val, 1, x->val, 1 );      //  x = b
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}


