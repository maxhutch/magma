/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zpbicgstab.cpp normal z -> s, Fri May 30 10:41:41 2014
       @author Hartwig Anzt

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
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the preconditioned 
    Biconjugate Gradient Stabelized method.

    Arguments
    =========

    magma_s_sparse_matrix A                   input matrix A
    magma_s_vector b                          RHS b
    magma_s_vector *x                         solution approximation
    magma_s_solver_par *solver_par            solver parameters
    magma_s_preconditioner *precond_par       preconditioner parameters

    ========================================================================  */


magma_int_t
magma_spbicgstab( magma_s_sparse_matrix A, magma_s_vector b, magma_s_vector *x,  
                    magma_s_solver_par *solver_par, 
                    magma_s_preconditioner *precond_par ){



    // prepare solver feedback
    solver_par->solver = Magma_PBICGSTAB;
    solver_par->numiter = 0;
    solver_par->info = 0;

    // some useful variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE, 
                                            c_mone = MAGMA_S_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_s_vector r,rr,p,v,s,t,ms,mt,y,z;
    magma_s_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &rr, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &p, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &v, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &s, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &t, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &ms, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &mt, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &y, Magma_DEV, dofs, c_zero );
    magma_s_vinit( &z, Magma_DEV, dofs, c_zero );

    
    // solver variables
    float alpha, beta, omega, rho_old, rho_new;
    float nom, betanom, nom0, r0, den;

    // solver setup
    magma_sscal( dofs, c_zero, x->val, 1) ;                    // x = 0
    magma_scopy( dofs, b.val, 1, r.val, 1 );                   // r = b
    magma_scopy( dofs, b.val, 1, rr.val, 1 );                  // rr = b
    nom0 = betanom = magma_snrm2( dofs, r.val, 1 );           // nom = || r ||
    nom = nom0*nom0;
    rho_new = omega = alpha = MAGMA_S_MAKE( 1.0, 0. );
    solver_par->init_res = nom0;

    magma_s_spmv( c_one, A, r, c_zero, v );                      // z = A r
    den = MAGMA_S_REAL( magma_sdot(dofs, v.val, 1, r.val, 1) ); // den = z' * r

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
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }

    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){
        rho_old = rho_new;                                   // rho_old=rho
        rho_new = magma_sdot( dofs, rr.val, 1, r.val, 1 );  // rho=<rr,r>
        beta = rho_new/rho_old * alpha/omega;   // beta=rho/rho_old *alpha/omega
        magma_sscal( dofs, beta, p.val, 1 );                 // p = beta*p
        magma_saxpy( dofs, c_mone * omega * beta, v.val, 1 , p.val, 1 );        
                                                        // p = p-omega*beta*v
        magma_saxpy( dofs, c_one, r.val, 1, p.val, 1 );      // p = p+r

        // preconditioner
        magma_s_applyprecond_left( A, p, &mt, precond_par );      
        magma_s_applyprecond_right( A, mt, &y, precond_par );        

        magma_s_spmv( c_one, A, y, c_zero, v );              // v = Ap

        alpha = rho_new / magma_sdot( dofs, rr.val, 1, v.val, 1 );
        magma_scopy( dofs, r.val, 1 , s.val, 1 );            // s=r
        magma_saxpy( dofs, c_mone * alpha, v.val, 1 , s.val, 1 ); // s=s-alpha*v

        // preconditioner
        magma_s_applyprecond_left( A, s, &ms, precond_par ); 
        magma_s_applyprecond_right( A, ms, &z, precond_par );      

        magma_s_spmv( c_one, A, z, c_zero, t );               // t=As

        // preconditioner
        magma_s_applyprecond_left( A, s, &ms, precond_par );      
        magma_s_applyprecond_left( A, t, &mt, precond_par );        

        // omega = <ms,mt>/<mt,mt>  
        omega = magma_sdot( dofs, mt.val, 1, ms.val, 1 )
                   / magma_sdot( dofs, mt.val, 1, mt.val, 1 );

        magma_saxpy( dofs, alpha, y.val, 1 , x->val, 1 );     // x=x+alpha*p
        magma_saxpy( dofs, omega, z.val, 1 , x->val, 1 );     // x=x+omega*s

        magma_scopy( dofs, s.val, 1 , r.val, 1 );             // r=s
        magma_saxpy( dofs, c_mone * omega, t.val, 1 , r.val, 1 ); // r=r-omega*t
        betanom = magma_snrm2( dofs, r.val, 1 );

        nom = betanom*betanom;


        if( solver_par->verbose > 0 ){
            magma_device_sync(); tempo2=magma_wtime();
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if ( betanom  < r0 ) {
            break;
        }
    }
    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    float residual;
    magma_sresidual( A, b, *x, &residual );
    solver_par->final_res = (real_Double_t) betanom;//residual;

    if( solver_par->numiter < solver_par->maxiter){
        solver_par->info = 0;
    }else if( solver_par->init_res > solver_par->final_res ){
        if( solver_par->verbose > 0 ){
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) betanom;
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
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        solver_par->info = -1;
    }
    magma_s_vfree(&r);
    magma_s_vfree(&rr);
    magma_s_vfree(&p);
    magma_s_vfree(&v);
    magma_s_vfree(&s);
    magma_s_vfree(&t);
    magma_s_vfree(&ms);
    magma_s_vfree(&mt);
    magma_s_vfree(&y);
    magma_s_vfree(&z);


    return MAGMA_SUCCESS;
}   /* magma_sbicgstab */


