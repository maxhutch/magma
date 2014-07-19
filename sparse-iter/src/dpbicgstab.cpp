/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from zpbicgstab.cpp normal z -> d, Fri Jul 18 17:34:29 2014
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>


#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real symmetric N-by-N positive definite matrix A.
    This is a GPU implementation of the preconditioned 
    Biconjugate Gradient Stabelized method.

    Arguments
    ---------

    @param
    A           magma_d_sparse_matrix
                input matrix A

    @param
    b           magma_d_vector
                RHS b

    @param
    x           magma_d_vector*
                solution approximation

    @param
    solver_par  magma_d_solver_par*
                solver parameters

    @param
    precond_par magma_d_preconditioner*
                preconditioner parameters

    @ingroup magmasparse_gesv
    ********************************************************************/

magma_int_t
magma_dpbicgstab( magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector *x,  
                    magma_d_solver_par *solver_par, 
                    magma_d_preconditioner *precond_par ){



    // prepare solver feedback
    solver_par->solver = Magma_PBICGSTAB;
    solver_par->numiter = 0;
    solver_par->info = 0;

    // some useful variables
    double c_zero = MAGMA_D_ZERO, c_one = MAGMA_D_ONE, 
                                            c_mone = MAGMA_D_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_d_vector r,rr,p,v,s,t,ms,mt,y,z;
    magma_d_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_d_vinit( &rr, Magma_DEV, dofs, c_zero );
    magma_d_vinit( &p, Magma_DEV, dofs, c_zero );
    magma_d_vinit( &v, Magma_DEV, dofs, c_zero );
    magma_d_vinit( &s, Magma_DEV, dofs, c_zero );
    magma_d_vinit( &t, Magma_DEV, dofs, c_zero );
    magma_d_vinit( &ms, Magma_DEV, dofs, c_zero );
    magma_d_vinit( &mt, Magma_DEV, dofs, c_zero );
    magma_d_vinit( &y, Magma_DEV, dofs, c_zero );
    magma_d_vinit( &z, Magma_DEV, dofs, c_zero );

    
    // solver variables
    double alpha, beta, omega, rho_old, rho_new;
    double nom, betanom, nom0, r0, den, res;

    // solver setup
    magma_dscal( dofs, c_zero, x->val, 1) ;                    // x = 0
    magma_dcopy( dofs, b.val, 1, r.val, 1 );                   // r = b
    magma_dcopy( dofs, b.val, 1, rr.val, 1 );                  // rr = b
    nom0 = betanom = magma_dnrm2( dofs, r.val, 1 );           // nom = || r ||
    nom = nom0*nom0;
    rho_new = omega = alpha = MAGMA_D_MAKE( 1.0, 0. );
    solver_par->init_res = nom0;

    magma_d_spmv( c_one, A, r, c_zero, v );                      // z = A r
    den = MAGMA_D_REAL( magma_ddot(dofs, v.val, 1, r.val, 1) ); // den = z' * r

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
        rho_new = magma_ddot( dofs, rr.val, 1, r.val, 1 );  // rho=<rr,r>
        beta = rho_new/rho_old * alpha/omega;   // beta=rho/rho_old *alpha/omega
        magma_dscal( dofs, beta, p.val, 1 );                 // p = beta*p
        magma_daxpy( dofs, c_mone * omega * beta, v.val, 1 , p.val, 1 );        
                                                        // p = p-omega*beta*v
        magma_daxpy( dofs, c_one, r.val, 1, p.val, 1 );      // p = p+r

        // preconditioner
        magma_d_applyprecond_left( A, p, &mt, precond_par );      
        magma_d_applyprecond_right( A, mt, &y, precond_par );        

        magma_d_spmv( c_one, A, y, c_zero, v );              // v = Ap

        alpha = rho_new / magma_ddot( dofs, rr.val, 1, v.val, 1 );
        magma_dcopy( dofs, r.val, 1 , s.val, 1 );            // s=r
        magma_daxpy( dofs, c_mone * alpha, v.val, 1 , s.val, 1 ); // s=s-alpha*v

        // preconditioner
        magma_d_applyprecond_left( A, s, &ms, precond_par ); 
        magma_d_applyprecond_right( A, ms, &z, precond_par );      

        magma_d_spmv( c_one, A, z, c_zero, t );               // t=As

        // preconditioner
        magma_d_applyprecond_left( A, s, &ms, precond_par );      
        magma_d_applyprecond_left( A, t, &mt, precond_par );        

        // omega = <ms,mt>/<mt,mt>  
        omega = magma_ddot( dofs, mt.val, 1, ms.val, 1 )
                   / magma_ddot( dofs, mt.val, 1, mt.val, 1 );

        magma_daxpy( dofs, alpha, y.val, 1 , x->val, 1 );     // x=x+alpha*p
        magma_daxpy( dofs, omega, z.val, 1 , x->val, 1 );     // x=x+omega*s

        magma_dcopy( dofs, s.val, 1 , r.val, 1 );             // r=s
        magma_daxpy( dofs, c_mone * omega, t.val, 1 , r.val, 1 ); // r=r-omega*t
        res = betanom = magma_dnrm2( dofs, r.val, 1 );

        nom = betanom*betanom;


        if( solver_par->verbose > 0 ){
            magma_device_sync(); tempo2=magma_wtime();
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if ( res/nom0  < solver_par->epsilon ) {
            break;
        }
    }
    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    magma_dresidual( A, b, *x, &residual );
    solver_par->final_res = residual;
    solver_par->iter_res = res;

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
    magma_d_vfree(&r);
    magma_d_vfree(&rr);
    magma_d_vfree(&p);
    magma_d_vfree(&v);
    magma_d_vfree(&s);
    magma_d_vfree(&t);
    magma_d_vfree(&ms);
    magma_d_vfree(&mt);
    magma_d_vfree(&y);
    magma_d_vfree(&z);


    return MAGMA_SUCCESS;
}   /* magma_dbicgstab */


