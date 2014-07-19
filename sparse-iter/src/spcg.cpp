/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @author Hartwig Anzt 

       @generated from zpcg.cpp normal z -> s, Fri Jul 18 17:34:29 2014
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the Conjugate Gradient method.

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                input matrix A

    @param
    b           magma_s_vector
                RHS b

    @param
    x           magma_s_vector*
                solution approximation

    @param
    solver_par  magma_s_solver_par*
                solver parameters

    @param
    precond_par magma_s_preconditioner*
                preconditioner

    @ingroup magmasparse_sposv
    ********************************************************************/

magma_int_t
magma_spcg( magma_s_sparse_matrix A, magma_s_vector b, magma_s_vector *x,  
            magma_s_solver_par *solver_par, 
            magma_s_preconditioner *precond_par ){

    // prepare solver feedback
    solver_par->solver = Magma_PCG;
    solver_par->numiter = 0;
    solver_par->info = 0;

    // local variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE;
    
    magma_int_t dofs = A.num_rows;

    // GPU workspace
    magma_s_vector r, rt, p, q, h;
    magma_s_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &rt, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &p, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &q, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &h, Magma_DEV, dofs, c_zero );
    
    // solver variables
    float alpha, beta;
    float nom, nom0, r0, gammaold, gammanew, den, res;

    // solver setup
    magma_sscal( dofs, c_zero, x->val, 1) ;                     // x = 0
    magma_scopy( dofs, b.val, 1, r.val, 1 );                    // r = b

    // preconditioner
    magma_s_applyprecond_left( A, r, &rt, precond_par );
    magma_s_applyprecond_right( A, rt, &h, precond_par );

    magma_scopy( dofs, h.val, 1, p.val, 1 );                    // p = h
    nom = MAGMA_S_REAL( magma_sdot(dofs, r.val, 1, h.val, 1) );          
    nom0 = magma_snrm2( dofs, r.val, 1 );                                                 
    magma_s_spmv( c_one, A, p, c_zero, q );                     // q = A p
    den = MAGMA_S_REAL( magma_sdot(dofs, p.val, 1, q.val, 1) );// den = p dot q
    solver_par->init_res = nom0;
    
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    // check positive definite
    if (den <= 0.0) {
        printf("Operator A is not postive definite. (Ar,r) = %f\n", den);
        return -100;
    }

    //Chronometry
    real_Double_t tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    if( solver_par->verbose > 0 ){
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    
    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){
        // preconditioner
        magma_s_applyprecond_left( A, r, &rt, precond_par );
        magma_s_applyprecond_right( A, rt, &h, precond_par );

        gammanew = MAGMA_S_REAL( magma_sdot(dofs, r.val, 1, h.val, 1) );   
                                                            // gn = < r,h>

        if( solver_par->numiter==1 ){
            magma_scopy( dofs, h.val, 1, p.val, 1 );                    // p = h            
        }else{
            beta = MAGMA_S_MAKE(gammanew/gammaold, 0.);       // beta = gn/go
            magma_sscal(dofs, beta, p.val, 1);            // p = beta*p
            magma_saxpy(dofs, c_one, h.val, 1, p.val, 1); // p = p + h 
        }

        magma_s_spmv( c_one, A, p, c_zero, q );           // q = A p
        den = MAGMA_S_REAL(magma_sdot(dofs, p.val, 1, q.val, 1));    
                // den = p dot q 

        alpha = MAGMA_S_MAKE(gammanew/den, 0.);
        magma_saxpy(dofs,  alpha, p.val, 1, x->val, 1);     // x = x + alpha p
        magma_saxpy(dofs, -alpha, q.val, 1, r.val, 1);      // r = r - alpha q
        gammaold = gammanew;

        res = magma_snrm2( dofs, r.val, 1 );
        if( solver_par->verbose > 0 ){
            magma_device_sync(); tempo2=magma_wtime();
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }


        if (  res/nom0  < solver_par->epsilon ) {
            break;
        }
    } 
    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    float residual;
    magma_sresidual( A, b, *x, &residual );
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if( solver_par->numiter < solver_par->maxiter){
        solver_par->info = 0;
    }else if( solver_par->init_res > solver_par->final_res ){
        if( solver_par->verbose > 0 ){
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) res;
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
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -1;
    }
    magma_s_vfree(&r);
    magma_s_vfree(&rt);
    magma_s_vfree(&p);
    magma_s_vfree(&q);
    magma_s_vfree(&h);

    return MAGMA_SUCCESS;
}   /* magma_scg */


