/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zcg_res.cpp normal z -> d, Mon May  2 23:30:55 2016
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the preconditioned Conjugate
    Gradient method.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A

    @param[in]
    b           magma_d_matrix
                RHS b

    @param[in,out]
    x           magma_d_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_d_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dposv
    ********************************************************************/

extern "C" magma_int_t
magma_dcg_res(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_CG;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    // solver variables
    double alpha, beta;
    double nom, nom0, r0,  res, nomb;
    double den, gammanew, gammaold = MAGMA_D_MAKE(1.0,0.0);
    // local variables
    double c_zero = MAGMA_D_ZERO, c_one = MAGMA_D_ONE;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // GPU workspace
    magma_d_matrix r={Magma_CSR}, p={Magma_CSR}, q={Magma_CSR};
    CHECK( magma_dvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    

    // solver setup
    CHECK(  magma_dresidualvec( A, b, *x, &r, &nom0, queue));

    magma_dcopy( dofs, r.dval, 1, p.dval, 1, queue );                    // p = h
    nom = MAGMA_D_ABS( magma_ddot( dofs, r.dval, 1, r.dval, 1, queue) );
    CHECK( magma_d_spmv( c_one, A, p, c_zero, q, queue ));             // q = A p
    solver_par->spmv_count++;
    den =  magma_ddot( dofs, p.dval, 1, q.dval, 1, queue ); // den = p dot q
    solver_par->init_res = nom0;
            
    nomb = magma_dnrm2( dofs, b.dval, 1, queue );
    if ( nomb == 0.0 ){
        nomb=1.0;
    }       
    if ( (r0 = nomb * solver_par->rtol) < ATOLERANCE ){
        r0 = ATOLERANCE;
    }
    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom < r0 ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }
    // check positive definite
    if ( MAGMA_D_ABS(den) <= 0.0 ) {
        info = MAGMA_NONSPD;
        goto cleanup;
    }

    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;

        gammanew = magma_ddot( dofs, r.dval, 1, r.dval, 1, queue );
                                                            // gn = < r,r>

        if ( solver_par->numiter == 1 ) {
            magma_dcopy( dofs, r.dval, 1, p.dval, 1, queue );                    // p = r
        } else {
            beta = (gammanew/gammaold);       // beta = gn/go
            magma_dscal( dofs, beta, p.dval, 1, queue );            // p = beta*p
            magma_daxpy( dofs, c_one, r.dval, 1, p.dval, 1, queue ); // p = p + r
        }

        CHECK( magma_d_spmv( c_one, A, p, c_zero, q, queue ));   // q = A p
        solver_par->spmv_count++;
        den = magma_ddot( dofs, p.dval, 1, q.dval, 1, queue );
                // den = p dot q

        alpha = gammanew / den;
        magma_daxpy( dofs,  alpha, p.dval, 1, x->dval, 1, queue );     // x = x + alpha p
        magma_daxpy( dofs, -alpha, q.dval, 1, r.dval, 1, queue );      // r = r - alpha q
        gammaold = gammanew;

        res = magma_dnrm2( dofs, r.dval, 1, queue );
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose == 0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if ( res/nomb <= solver_par->rtol || res <= solver_par->atol ){
            break;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_dresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == 0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->rtol*solver_par->init_res ||
            solver_par->iter_res < solver_par->atol ) {
            info = MAGMA_SUCCESS;
        }
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == 0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_dmfree(&r, queue );
    magma_dmfree(&p, queue );
    magma_dmfree(&q, queue );

    solver_par->info = info;
    return info;
}   /* magma_dcg */
