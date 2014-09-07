/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>


#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Biconjugate Gradient Stabelized method.

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix A

    @param
    b           magma_z_vector
                RHS b

    @param
    x           magma_z_vector*
                solution approximation

    @param
    solver_par  magma_z_solver_par*
                solver parameters

    @ingroup magmasparse_zgesv
    ********************************************************************/

magma_int_t
magma_zbicgstab( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                    magma_z_solver_par *solver_par ){

    // prepare solver feedback
    solver_par->solver = Magma_BICGSTAB;
    solver_par->numiter = 0;
    solver_par->info = 0;

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                            c_mone = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_z_vector r,rr,p,v,s,t;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &rr, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &p, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &v, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &s, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &t, Magma_DEV, dofs, c_zero );

    
    // solver variables
    magmaDoubleComplex alpha, beta, omega, rho_old, rho_new;
    double nom, betanom, nom0, r0, den, res;

    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                    // x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );                   // r = b
    magma_zcopy( dofs, b.val, 1, rr.val, 1 );                  // rr = b
    nom0 = betanom = magma_dznrm2( dofs, r.val, 1 );           // nom = || r ||
    nom = nom0*nom0;
    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    solver_par->init_res = nom0;

    magma_z_spmv( c_one, A, r, c_zero, v );                      // z = A r
    den = MAGMA_Z_REAL( magma_zdotc(dofs, v.val, 1, r.val, 1) ); // den = z' * r

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

        rho_new = magma_zdotc( dofs, rr.val, 1, r.val, 1 );  // rho=<rr,r>
        beta = rho_new/rho_old * alpha/omega;   // beta=rho/rho_old *alpha/omega
        magma_zscal( dofs, beta, p.val, 1 );                 // p = beta*p
        magma_zaxpy( dofs, c_mone * omega * beta, v.val, 1 , p.val, 1 );        
                                                        // p = p-omega*beta*v
        magma_zaxpy( dofs, c_one, r.val, 1, p.val, 1 );      // p = p+r
        magma_z_spmv( c_one, A, p, c_zero, v );              // v = Ap

        alpha = rho_new / magma_zdotc( dofs, rr.val, 1, v.val, 1 );
        magma_zcopy( dofs, r.val, 1 , s.val, 1 );            // s=r
        magma_zaxpy( dofs, c_mone * alpha, v.val, 1 , s.val, 1 ); // s=s-alpha*v

        magma_z_spmv( c_one, A, s, c_zero, t );               // t=As
        omega = magma_zdotc( dofs, t.val, 1, s.val, 1 )   // omega = <s,t>/<t,t>
                   / magma_zdotc( dofs, t.val, 1, t.val, 1 );

        magma_zaxpy( dofs, alpha, p.val, 1 , x->val, 1 );     // x=x+alpha*p
        magma_zaxpy( dofs, omega, s.val, 1 , x->val, 1 );     // x=x+omega*s

        magma_zcopy( dofs, s.val, 1 , r.val, 1 );             // r=s
        magma_zaxpy( dofs, c_mone * omega, t.val, 1 , r.val, 1 ); // r=r-omega*t
        res = betanom = magma_dznrm2( dofs, r.val, 1 );

        nom = betanom*betanom;
        rho_old = rho_new;                                    // rho_old=rho

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
    magma_zresidual( A, b, *x, &residual );
    solver_par->iter_res = res;
    solver_par->final_res = residual;

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
    magma_z_vfree(&r);
    magma_z_vfree(&rr);
    magma_z_vfree(&p);
    magma_z_vfree(&v);
    magma_z_vfree(&s);
    magma_z_vfree(&t);

    return MAGMA_SUCCESS;
}   /* magma_zbicgstab */


