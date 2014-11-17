/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @author Hartwig Anzt 

       @generated from ziterref.cpp normal z -> s, Sat Nov 15 19:54:22 2014
*/

#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the Iterative Refinement method.
    The inner solver is passed via the preconditioner argument.

    Arguments
    ---------

    @param[in]
    A           magma_s_sparse_matrix
                input matrix A

    @param[in]
    b           magma_s_vector
                RHS b

    @param[in,out]
    x           magma_s_vector*
                solution approximation

    @param[in,out]
    solver_par  magma_s_solver_par*
                solver parameters

    @param[in,out]
    precond_par magma_s_preconditioner*
                inner solver
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgesv
    ********************************************************************/

extern "C" magma_int_t
magma_siterref(
    magma_s_sparse_matrix A, magma_s_vector b, magma_s_vector *x,  
    magma_s_solver_par *solver_par, magma_s_preconditioner *precond_par,
    magma_queue_t queue )
{
    // set queue for old dense routines
    magma_queue_t orig_queue;
    magmablasGetKernelStream( &orig_queue );

    // prepare solver feedback
    solver_par->solver = Magma_ITERREF;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;

    float residual;
    magma_sresidual( A, b, *x, &residual, queue );
    solver_par->init_res = residual;

    // some useful variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE, 
                                                c_mone = MAGMA_S_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_s_vector r,z;
    magma_s_vinit( &r, Magma_DEV, dofs, c_zero, queue );
    magma_s_vinit( &z, Magma_DEV, dofs, c_zero, queue );

    // solver variables
    float nom, nom0, r0;

    // solver setup
    magma_sscal( dofs, c_zero, x->dval, 1) ;                    // x = 0

    magma_scopy( dofs, b.dval, 1, r.dval, 1 );                    // r = b
    nom0 = magma_snrm2(dofs, r.dval, 1);                       // nom0 = || r ||
    nom = nom0 * nom0;
    solver_par->init_res = nom0;

    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 ) {
        magmablasSetKernelStream( orig_queue );
        return MAGMA_SUCCESS;
    }
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }
    
    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ) {

        magma_sscal( dofs, MAGMA_S_MAKE(1./nom, 0.), r.dval, 1) ;  // scale it
        magma_s_precond( A, r, &z, precond_par, queue );  // inner solver:  A * z = r
        magma_sscal( dofs, MAGMA_S_MAKE(nom, 0.), z.dval, 1) ;  // scale it
        magma_saxpy(dofs,  c_one, z.dval, 1, x->dval, 1);        // x = x + z
        magma_s_spmv( c_mone, A, *x, c_zero, r, queue );              // r = - A x
        magma_saxpy(dofs,  c_one, b.dval, 1, r.dval, 1);         // r = r + b
        nom = magma_snrm2(dofs, r.dval, 1);                    // nom = || r || 

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
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
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_sresidual( A, b, *x, &residual, queue );
    solver_par->final_res = residual;
    solver_par->iter_res = nom;

    if ( solver_par->numiter < solver_par->maxiter) {
        solver_par->info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) nom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = MAGMA_SLOW_CONVERGENCE;
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) nom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = MAGMA_DIVERGENCE;
    }   
    magma_s_vfree(&r, queue );
    magma_s_vfree(&z, queue );


    magmablasSetKernelStream( orig_queue );
    return MAGMA_SUCCESS;
}   /* magma_siterref */


