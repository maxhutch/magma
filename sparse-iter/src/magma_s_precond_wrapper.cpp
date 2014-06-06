/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from magma_z_precond_wrapper.cpp normal z -> s, Fri May 30 10:41:42 2014
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

    magma_s_sparse_matrix A         sparse matrix A    
    magma_s_vector x                input vector x  
    magma_s_vector y                input vector y      
    magma_s_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_s_precond( magma_s_sparse_matrix A, magma_s_vector b, 
                 magma_s_vector *x, magma_s_preconditioner precond )
{
// set up precond parameters as solver parameters   
    magma_s_solver_par psolver_par;
    psolver_par.epsilon = precond.epsilon;
    psolver_par.maxiter = precond.maxiter;
    psolver_par.restart = precond.restart;
   
    if( precond.solver == Magma_CG ){
// printf( "start CG preconditioner with epsilon: %f and maxiter: %d: ", 
//                            psolver_par.epsilon, psolver_par.maxiter );
        magma_scg( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_GMRES ){
// printf( "start GMRES preconditioner with epsilon: %f and maxiter: %d: ", 
//                               psolver_par.epsilon, psolver_par.maxiter );
        magma_sgmres( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_BICGSTAB ){
// printf( "start BICGSTAB preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_sbicgstab( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_JACOBI ){
// printf( "start JACOBI preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_sjacobi( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_BCSRLU ){
// printf( "start BCSRLU preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_sbcsrlu( A, b, x, &psolver_par );
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

    magma_s_sparse_matrix A         sparse matrix A    
    magma_s_vector x                input vector x  
    magma_s_vector y                input vector y      
    magma_s_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_s_precondsetup( magma_s_sparse_matrix A, magma_s_vector b, 
                      magma_s_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_sjacobisetup_diagscal( A, &(precond->d) );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_PASTIX ){
        magma_spastixsetup( A, b, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_scuilusetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_scuiccsetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AILU ){
        magma_ialusetup( A, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AICC ){
        magma_saiccsetup( A, precond );
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

    magma_s_sparse_matrix A         sparse matrix A    
    magma_s_vector x                input vector x  
    magma_s_vector y                input vector y      
    magma_s_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_s_applyprecond( magma_s_sparse_matrix A, magma_s_vector b, 
                      magma_s_vector *x, magma_s_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_sjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_PASTIX ){
        magma_sapplypastix( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_s_vector tmp;
        magma_s_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_S_MAKE(1.0, 0.0) );
     //   magma_sapplycuilu_l( b, &tmp, precond ); 
     //   magma_sapplycuilu_r( tmp, x, precond );
        magma_s_vfree( &tmp );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_s_vector tmp;
        magma_s_vinit( &tmp, Magma_DEV, A.num_rows, MAGMA_S_MAKE(1.0, 0.0) );
       // magma_strisv_l_nu( precond->L, b, &tmp );
       // magma_strisv_r_nu( precond->L, tmp, x );
        magma_s_vfree( &tmp );
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

    magma_s_sparse_matrix A         sparse matrix A    
    magma_s_vector x                input vector x  
    magma_s_vector y                input vector y      
    magma_s_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_s_applyprecond_left( magma_s_sparse_matrix A, magma_s_vector b, 
                      magma_s_vector *x, magma_s_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_sjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_sapplycuilu_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AILU ){
        magma_sapplycuilu_l( b, x, precond );
//        magma_sapplyailu_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_sapplycuicc_l( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AICC ){
        magma_sapplycuicc_l( b, x, precond );
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

    magma_s_sparse_matrix A         sparse matrix A    
    magma_s_vector x                input vector x  
    magma_s_vector y                input vector y      
    magma_s_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_s_applyprecond_right( magma_s_sparse_matrix A, magma_s_vector b, 
                      magma_s_vector *x, magma_s_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        //magma_sjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        magma_scopy( b.num_rows, b.val, 1, x->val, 1 );    // x = b
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ILU ){
        magma_sapplycuilu_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AILU ){
        magma_sapplycuilu_r( b, x, precond );
//        magma_sapplyailu_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_ICC ){
        magma_sapplycuicc_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else if( precond->solver == Magma_AICC ){
        magma_sapplycuicc_r( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}


