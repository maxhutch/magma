/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @author Hartwig Anzt 

       @generated from ziterref.cpp normal z -> c, Fri May 30 10:41:41 2014
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Iterative Refinement method.
    The inner solver is passed via the preconditioner argument.

    Arguments
    =========

    magma_c_sparse_matrix A                   input matrix A
    magma_c_vector b                          RHS b
    magma_c_vector *x                         solution approximation
    magma_c_solver_par *solver_par       solver parameters
    magma_c_preconditioner *precond_par       inner solver

    ========================================================================  */

magma_int_t
magma_citerref( magma_c_sparse_matrix A, magma_c_vector b, magma_c_vector *x,  
   magma_c_solver_par *solver_par, magma_c_preconditioner *precond_par ){

    // prepare solver feedback
    solver_par->solver = Magma_ITERREF;
    solver_par->numiter = 0;
    solver_par->info = 0;

    float residual;
    magma_cresidual( A, b, *x, &residual );
    solver_par->init_res = residual;

    // some useful variables
    magmaFloatComplex c_zero = MAGMA_C_ZERO, c_one = MAGMA_C_ONE, 
                                                c_mone = MAGMA_C_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_c_vector r,z;
    magma_c_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_c_vinit( &z, Magma_DEV, dofs, c_zero );

    // solver variables
    float nom, nom0, r0;

    // solver setup
    magma_cscal( dofs, c_zero, x->val, 1) ;                    // x = 0

    magma_c_spmv( c_mone, A, *x, c_zero, r );                  // r = - A x
    magma_caxpy(dofs,  c_one, b.val, 1, r.val, 1);             // r = r + b
    nom0 = magma_scnrm2(dofs, r.val, 1);                       // nom0 = || r ||
    nom = nom0 * nom0;
    
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    if( solver_par->verbose > 0 ){
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }
    
    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){

        magma_cscal( dofs, MAGMA_C_MAKE(1./nom, 0.), r.val, 1) ;  // scale it
        magma_c_precond( A, r, &z, *precond_par );  // inner solver:  A * z = r
        magma_cscal( dofs, MAGMA_C_MAKE(nom, 0.), z.val, 1) ;  // scale it
        magma_caxpy(dofs,  c_one, z.val, 1, x->val, 1);        // x = x + z
        magma_c_spmv( c_mone, A, *x, c_zero, r );              // r = - A x
        magma_caxpy(dofs,  c_one, b.val, 1, r.val, 1);         // r = r + b
        nom = magma_scnrm2(dofs, r.val, 1);                    // nom = || r || 

        if( solver_par->verbose > 0 ){
            magma_device_sync(); tempo2=magma_wtime();
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) nom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if (  nom  < r0 ) {
            break;
        }
    } 
    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_cresidual( A, b, *x, &residual );
    solver_par->final_res = residual;

    if( solver_par->numiter < solver_par->maxiter){
        solver_par->info = 0;
    }else if( solver_par->init_res > solver_par->final_res ){
        if( solver_par->verbose > 0 ){
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) nom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -2;
    }
    else{
        if( solver_par->verbose > 0 ){
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) nom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -1;
    }   
    magma_c_vfree(&r);
    magma_c_vfree(&z);


    return MAGMA_SUCCESS;
}   /* magma_citerref */


