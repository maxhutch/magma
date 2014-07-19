/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from magma_z_precond_wrapper.cpp normal z -> d, Fri Jul 18 17:34:29 2014
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"




/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is chosen. It approximates x for A x = y.

    Arguments
    ---------

    @param
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param
    x           magma_d_vector
                input vector x  

    @param
    y           magma_d_vector
                input vector y      

    @param
    precond     magma_d_preconditioner
                preconditioner

    @ingroup magmasparse_daux
    ********************************************************************/

magma_int_t
magma_d_precond( magma_d_sparse_matrix A, magma_d_vector b, 
                 magma_d_vector *x, magma_d_preconditioner precond )
{
// set up precond parameters as solver parameters   
    magma_d_solver_par psolver_par;
    psolver_par.epsilon = precond.epsilon;
    psolver_par.maxiter = precond.maxiter;
    psolver_par.restart = precond.restart;
   
    if( precond.solver == Magma_CG ){
// printf( "start CG preconditioner with epsilon: %f and maxiter: %d: ", 
//                            psolver_par.epsilon, psolver_par.maxiter );
        magma_dcg( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_GMRES ){
// printf( "start GMRES preconditioner with epsilon: %f and maxiter: %d: ", 
//                               psolver_par.epsilon, psolver_par.maxiter );
        magma_dgmres( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_BICGSTAB ){
// printf( "start BICGSTAB preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_dbicgstab( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_JACOBI ){
// printf( "start JACOBI preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_djacobi( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_BCSRLU ){
// printf( "start BCSRLU preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_dbcsrlu( A, b, x, &psolver_par );
// printf( "done.\n");
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
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param
    x           magma_d_vector
                input vector x  

    @param
    y           magma_d_vector
                input vector y      

    @param
    precond     magma_d_preconditioner
                preconditioner

    @ingroup magmasparse_daux
    ********************************************************************/

magma_int_t
magma_d_precondsetup( magma_d_sparse_matrix A, magma_d_vector b, 
                      magma_d_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_djacobisetup_diagscal( A, &(precond->d) );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_PASTIX ){
        magma_dpastixsetup( A, b, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_dcuilusetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
//        magma_dcuilusetup( A, precond );
        magma_dcuiccsetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AILU ){
        magma_dailusetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AICC ){
        //magma_dailusetup( A, precond );
        magma_daiccsetup( A, precond );
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
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param
    x           magma_d_vector
                input vector x  

    @param
    y           magma_d_vector
                input vector y      

    @param
    precond     magma_d_preconditioner
                preconditioner

    @ingroup magmasparse_daux
    ********************************************************************/

magma_int_t
magma_d_applyprecond( magma_d_sparse_matrix A, magma_d_vector b, 
                      magma_d_vector *x, magma_d_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_djacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_PASTIX ){
        magma_dapplypastix( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_d_vector tmp;
        magma_d_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_D_MAKE(1.0, 0.0) );
     //   magma_dapplycuilu_l( b, &tmp, precond ); 
     //   magma_dapplycuilu_r( tmp, x, precond );
        magma_d_vfree( &tmp );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_d_vector tmp;
        magma_d_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_D_MAKE(1.0, 0.0) );
       // magma_dtrisv_l_nu( precond->L, b, &tmp );
       // magma_dtrisv_r_nu( precond->L, tmp, x );
        magma_d_vfree( &tmp );
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
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param
    x           magma_d_vector
                input vector x  

    @param
    y           magma_d_vector
                input vector y      

    @param
    precond     magma_d_preconditioner
                preconditioner

    @ingroup magmasparse_daux
    ********************************************************************/

magma_int_t
magma_d_applyprecond_left( magma_d_sparse_matrix A, magma_d_vector b, 
                      magma_d_vector *x, magma_d_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_djacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_dapplycuilu_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AILU ){
        magma_dapplycuilu_l( b, x, precond );
//        magma_dapplyailu_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        //magma_dapplycuilu_l( b, x, precond );
        magma_dapplycuicc_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AICC ){
        //magma_dapplycuilu_l( b, x, precond );
        magma_dapplycuicc_l( b, x, precond );
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
    A           magma_d_sparse_matrix
                sparse matrix A    

    @param
    x           magma_d_vector
                input vector x  

    @param
    y           magma_d_vector
                input vector y      

    @param
    precond     magma_d_preconditioner
                preconditioner

    @ingroup magmasparse_daux
    ********************************************************************/

magma_int_t
magma_d_applyprecond_right( magma_d_sparse_matrix A, magma_d_vector b, 
                      magma_d_vector *x, magma_d_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        //magma_djacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        magma_dcopy( b.num_rows, b.val, 1, x->val, 1 );    // x = b
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_dapplycuilu_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AILU ){
        magma_dapplycuilu_r( b, x, precond );
//        magma_dapplyailu_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_dapplycuicc_r( b, x, precond );
//        magma_dapplycuilu_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AICC ){
        magma_dapplycuicc_r( b, x, precond );
//        magma_dapplycuilu_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}


