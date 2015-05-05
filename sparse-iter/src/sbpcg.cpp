/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @author Hartwig Anzt

       @generated from zbpcg.cpp normal z -> s, Sun May  3 11:22:59 2015
*/

#include "common_magmasparse.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )

#define  r(i)  r.dval+i*dofs
#define  b(i)  b.dval+i*dofs
#define  h(i)  h.dval+i*dofs
#define  p(i)  p.dval+i*dofs
#define  q(i)  q.dval+i*dofs



/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the block preconditioned Conjugate
    Gradient method.

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A

    @param[in]
    b           magma_s_matrix
                RHS b - can be a block

    @param[in,out]
    x           magma_s_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_s_solver_par*
                solver parameters

    @param[in]
    precond_par magma_s_preconditioner*
                preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sposv
    ********************************************************************/

extern "C" magma_int_t
magma_sbpcg(
    magma_s_matrix A, magma_s_matrix b, magma_s_matrix *x,
    magma_s_solver_par *solver_par,
    magma_s_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set queue for old dense routines
    magma_queue_t orig_queue=NULL;
    magmablasGetKernelStream( &orig_queue );

    
    magma_int_t i, num_vecs = b.num_rows/A.num_rows;

    // prepare solver feedback
    solver_par->solver = Magma_PCG;
    solver_par->numiter = 0;
    solver_par->info = MAGMA_SUCCESS;

    // local variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE;
    
    magma_int_t dofs = A.num_rows;

    // GPU workspace
    magma_s_matrix r={Magma_CSR}, rt={Magma_CSR}, p={Magma_CSR}, q={Magma_CSR}, h={Magma_CSR};

    
    // solver variables
    float *alpha={0}, *beta={0};
    alpha = NULL;
    beta = NULL;


    float *nom={0}, *nom0={0}, *r0={0}, *gammaold={0}, *gammanew={0}, *den={0}, *res={0}, *residual={0};
    nom        = NULL;
    nom0       = NULL;
    r0         = NULL;
    gammaold   = NULL;
    gammanew   = NULL;
    den        = NULL;
    res        = NULL;
    residual   = NULL;
    
    CHECK( magma_smalloc_cpu(&alpha, num_vecs));
    CHECK( magma_smalloc_cpu(&beta, num_vecs));
    CHECK( magma_smalloc_cpu(&residual, num_vecs));
    CHECK( magma_smalloc_cpu(&nom, num_vecs));
    CHECK( magma_smalloc_cpu(&nom0, num_vecs));
    CHECK( magma_smalloc_cpu(&r0, num_vecs));
    CHECK( magma_smalloc_cpu(&gammaold, num_vecs));
    CHECK( magma_smalloc_cpu(&gammanew, num_vecs));
    CHECK( magma_smalloc_cpu(&den, num_vecs));
    CHECK( magma_smalloc_cpu(&res, num_vecs));
    CHECK( magma_smalloc_cpu(&residual, num_vecs));
    
    CHECK( magma_svinit( &r, Magma_DEV, dofs*num_vecs, 1, c_zero, queue ));
    CHECK( magma_svinit( &rt, Magma_DEV, dofs*num_vecs, 1, c_zero, queue ));
    CHECK( magma_svinit( &p, Magma_DEV, dofs*num_vecs, 1, c_zero, queue ));
    CHECK( magma_svinit( &q, Magma_DEV, dofs*num_vecs, 1, c_zero, queue ));
    CHECK( magma_svinit( &h, Magma_DEV, dofs*num_vecs, 1, c_zero, queue ));

    // solver setup
    CHECK(  magma_sresidualvec( A, b, *x, &r, nom0, queue));

    // preconditioner
    CHECK( magma_s_applyprecond_left( A, r, &rt, precond_par, queue ));
    CHECK( magma_s_applyprecond_right( A, rt, &h, precond_par, queue ));

    magma_scopy( dofs*num_vecs, h.dval, 1, p.dval, 1 );                 // p = h

    for( i=0; i<num_vecs; i++) {
        nom[i] = MAGMA_S_REAL( magma_sdot(dofs, r(i), 1, h(i), 1) );
        nom0[i] = magma_snrm2( dofs, r(i), 1 );
    }
                                          
    CHECK( magma_s_spmv( c_one, A, p, c_zero, q, queue ));             // q = A p

    for( i=0; i<num_vecs; i++)
        den[i] = MAGMA_S_REAL( magma_sdot(dofs, p(i), 1, q(i), 1) );  // den = p dot q

    solver_par->init_res = nom0[0];
    
    if ( (r0[0] = nom[0] * solver_par->epsilon) < ATOLERANCE )
        r0[0] = ATOLERANCE;
    // check positive definite
    if (den[0] <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den[0]);
        magmablasSetKernelStream( orig_queue );
        info = MAGMA_NONSPD; 
        goto cleanup;
    }
    if ( nom[0] < r0[0] ) {
        solver_par->final_res = solver_par->init_res;
        solver_par->iter_res = solver_par->init_res;
        goto cleanup;
    }

    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t)nom0[0];
        solver_par->timing[0] = 0.0;
    }
    
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        // preconditioner
        CHECK( magma_s_applyprecond_left( A, r, &rt, precond_par, queue ));
        CHECK( magma_s_applyprecond_right( A, rt, &h, precond_par, queue ));


        for( i=0; i<num_vecs; i++)
            gammanew[i] = MAGMA_S_REAL( magma_sdot(dofs, r(i), 1, h(i), 1) );  // gn = < r,h>


        if ( solver_par->numiter==1 ) {
            magma_scopy( dofs*num_vecs, h.dval, 1, p.dval, 1 );                    // p = h
        } else {
            for( i=0; i<num_vecs; i++) {
                beta[i] = MAGMA_S_MAKE(gammanew[i]/gammaold[i], 0.);       // beta = gn/go
                magma_sscal(dofs, beta[i], p(i), 1);            // p = beta*p
                magma_saxpy(dofs, c_one, h(i), 1, p(i), 1); // p = p + h
            }
        }

        CHECK( magma_s_spmv( c_one, A, p, c_zero, q, queue ));   // q = A p

     //   magma_s_bspmv_tuned( dofs, num_vecs, c_one, A, p.dval, c_zero, q.dval, queue );


        for( i=0; i<num_vecs; i++) {
            den[i] = MAGMA_S_REAL(magma_sdot(dofs, p(i), 1, q(i), 1));
                // den = p dot q

            alpha[i] = MAGMA_S_MAKE(gammanew[i]/den[i], 0.);
            magma_saxpy(dofs,  alpha[i], p(i), 1, x->dval+dofs*i, 1); // x = x + alpha p
            magma_saxpy(dofs, -alpha[i], q(i), 1, r(i), 1);      // r = r - alpha q
            gammaold[i] = gammanew[i];

            res[i] = magma_snrm2( dofs, r(i), 1 );
        }

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res[0];
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }


        if (  res[0]/nom0[0]  < solver_par->epsilon ) {
            break;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    CHECK( magma_sresidual( A, b, *x, residual, queue ));
    solver_par->iter_res = res[0];
    solver_par->final_res = residual[0];

    if ( solver_par->numiter < solver_par->maxiter ) {
        solver_par->info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res[0];
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->epsilon*solver_par->init_res ){
            info = MAGMA_SUCCESS;
        }
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res[0];
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    for( i=0; i<num_vecs; i++) {
        printf("%.4e  ",res[i]);
    }
    printf("\n");
    for( i=0; i<num_vecs; i++) {
        printf("%.4e  ",residual[i]);
    }
    printf("\n");

cleanup:
    magma_smfree(&r, queue );
    magma_smfree(&rt, queue );
    magma_smfree(&p, queue );
    magma_smfree(&q, queue );
    magma_smfree(&h, queue );

    magma_free_cpu(alpha);
    magma_free_cpu(beta);
    magma_free_cpu(nom);
    magma_free_cpu(nom0);
    magma_free_cpu(r0);
    magma_free_cpu(gammaold);
    magma_free_cpu(gammanew);
    magma_free_cpu(den);
    magma_free_cpu(res);

    magmablasSetKernelStream( orig_queue );
    solver_par->info = info;
    return info;
}   /* magma_sbpcg */


