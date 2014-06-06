/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magma_z_precond_wrapper.cpp normal z -> c, Fri May 30 10:41:42 2014
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"




/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is chosen. It approximates x for A x = y.

    Arguments
    =========

    magma_c_sparse_matrix A         sparse matrix A    
    magma_c_vector x                input vector x  
    magma_c_vector y                input vector y      
    magma_c_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_c_precond( magma_c_sparse_matrix A, magma_c_vector b, 
                 magma_c_vector *x, magma_c_preconditioner precond )
{
// set up precond parameters as solver parameters   
    magma_c_solver_par psolver_par;
    psolver_par.epsilon = precond.epsilon;
    psolver_par.maxiter = precond.maxiter;
    psolver_par.restart = precond.restart;
   
    if( precond.solver == Magma_CG ){
// printf( "start CG preconditioner with epsilon: %f and maxiter: %d: ", 
//                            psolver_par.epsilon, psolver_par.maxiter );
        magma_ccg( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_GMRES ){
// printf( "start GMRES preconditioner with epsilon: %f and maxiter: %d: ", 
//                               psolver_par.epsilon, psolver_par.maxiter );
        magma_cgmres( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_BICGSTAB ){
// printf( "start BICGSTAB preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_cbicgstab( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_JACOBI ){
// printf( "start JACOBI preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_cjacobi( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_BCSRLU ){
// printf( "start BCSRLU preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_cbcsrlu( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }

    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is preprocessed. 
    E.g. for Jacobi: the scaling-vetor, for ILU the factorization...

    Arguments
    =========

    magma_c_sparse_matrix A         sparse matrix A    
    magma_c_vector x                input vector x  
    magma_c_vector y                input vector y      
    magma_c_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_c_precondsetup( magma_c_sparse_matrix A, magma_c_vector b, 
                      magma_c_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_cjacobisetup_diagscal( A, &(precond->d) );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_PASTIX ){
        magma_cpastixsetup( A, b, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_ccuilusetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_ccuiccsetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AILU ){
        magma_cailusetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AICC ){
        magma_caiccsetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}



/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is applied. 
    E.g. for Jacobi: the scaling-vetor, for ILU the triangular solves...

    Arguments
    =========

    magma_c_sparse_matrix A         sparse matrix A    
    magma_c_vector x                input vector x  
    magma_c_vector y                input vector y      
    magma_c_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_c_applyprecond( magma_c_sparse_matrix A, magma_c_vector b, 
                      magma_c_vector *x, magma_c_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_cjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_PASTIX ){
        magma_capplypastix( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_c_vector tmp;
        magma_c_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_C_MAKE(1.0, 0.0) );
     //   magma_capplycuilu_l( b, &tmp, precond ); 
     //   magma_capplycuilu_r( tmp, x, precond );
        magma_c_vfree( &tmp );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_c_vector tmp;
        magma_c_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_C_MAKE(1.0, 0.0) );
       // magma_ctrisv_l_nu( precond->L, b, &tmp );
       // magma_ctrisv_r_nu( precond->L, tmp, x );
        magma_c_vfree( &tmp );
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective left preconditioner
    is applied. 
    E.g. for Jacobi: the scaling-vetor, for ILU the left triangular solve...

    Arguments
    =========

    magma_c_sparse_matrix A         sparse matrix A    
    magma_c_vector x                input vector x  
    magma_c_vector y                input vector y      
    magma_c_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_c_applyprecond_left( magma_c_sparse_matrix A, magma_c_vector b, 
                      magma_c_vector *x, magma_c_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_cjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_capplycuilu_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AILU ){
        magma_capplycuilu_l( b, x, precond );
//        magma_capplyailu_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_capplycuicc_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AICC ){
        magma_capplycuicc_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective right-preconditioner
    is applied. 
    E.g. for Jacobi: the scaling-vetor, for ILU the right triangular solve...

    Arguments
    =========

    magma_c_sparse_matrix A         sparse matrix A    
    magma_c_vector x                input vector x  
    magma_c_vector y                input vector y      
    magma_c_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_c_applyprecond_right( magma_c_sparse_matrix A, magma_c_vector b, 
                      magma_c_vector *x, magma_c_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        //magma_cjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        magma_ccopy( b.num_rows, b.val, 1, x->val, 1 );    // x = b
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_capplycuilu_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AILU ){
        magma_capplycuilu_r( b, x, precond );
//        magma_capplyailu_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_capplycuicc_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AICC ){
        magma_capplycuicc_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}


