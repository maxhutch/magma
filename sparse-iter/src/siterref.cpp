/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/ziterref.cpp normal z -> s, Mon May  2 23:30:58 2016
*/

#include "magmasparse_internal.h"

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
    A           magma_s_matrix
                input matrix A

    @param[in]
    b           magma_s_matrix
                RHS b

    @param[in,out]
    x           magma_s_matrix*
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
    magma_s_matrix A, magma_s_matrix b, magma_s_matrix *x,
    magma_s_solver_par *solver_par, magma_s_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // some useful variables
    float c_zero = MAGMA_S_ZERO;
    float c_one  = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;

    // prepare solver feedback
    solver_par->solver = Magma_ITERREF;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    magma_int_t dofs = A.num_rows*b.num_cols;

    // solver variables
    float nom, nom0;
    
    // workspace
    magma_s_matrix r={Magma_CSR}, z={Magma_CSR};
    CHECK( magma_svinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    float residual;
    CHECK( magma_sresidual( A, b, *x, &residual, queue ));
    solver_par->init_res = residual;
   

    // solver setup
    magma_sscal( dofs, c_zero, x->dval, 1, queue );                    // x = 0
    //CHECK(  magma_sresidualvec( A, b, *x, &r, nom, queue));
    magma_scopy( dofs, b.dval, 1, r.dval, 1, queue );                    // r = b
    nom0 = magma_snrm2( dofs, r.dval, 1, queue );                       // nom0 = || r ||
    nom = nom0 * nom0;
    solver_par->init_res = nom0;

    if( nom0 < solver_par->atol ||
        nom0/solver_par->init_res < solver_par->rtol ){
        solver_par->final_res = solver_par->init_res;
        solver_par->iter_res = solver_par->init_res;
        info = MAGMA_SUCCESS;
        goto cleanup;
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
        magma_sscal( dofs, MAGMA_S_MAKE(1./nom, 0.), r.dval, 1, queue );  // scale it
        CHECK( magma_s_precond( A, r, &z, precond_par, queue )); // inner solver:  A * z = r
        magma_sscal( dofs, MAGMA_S_MAKE(nom, 0.), z.dval, 1, queue );  // scale it
        magma_saxpy( dofs,  c_one, z.dval, 1, x->dval, 1, queue );        // x = x + z
        CHECK( magma_s_spmv( c_neg_one, A, *x, c_zero, r, queue ));      // r = - A x
        solver_par->spmv_count++;
        magma_saxpy( dofs,  c_one, b.dval, 1, r.dval, 1, queue );         // r = r + b
        nom = magma_snrm2( dofs, r.dval, 1, queue );                    // nom = || r ||

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) nom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if( nom < solver_par->atol ||
            nom/solver_par->init_res < solver_par->rtol ){
            break;
        }
    }
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    CHECK(  magma_sresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->final_res = residual;
    solver_par->iter_res = nom;

    if ( solver_par->numiter < solver_par->maxiter ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) nom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->atol ||
            solver_par->iter_res/solver_par->init_res < solver_par->rtol ){
            info = MAGMA_SUCCESS;
        }
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
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_smfree(&r, queue );
    magma_smfree(&z, queue );

    solver_par->info = info;
    return info;
}   /* magma_siterref */
