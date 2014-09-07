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

#define  q(i)     (q.val + (i)*dofs)

/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Biconjugate Gradient Stabelized method.
    The difference to magma_zbicgstab is that we use specifically designed kernels
    merging multiple operations into one kernel.

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
magma_zbicgstab_merge2( magma_z_sparse_matrix A, magma_z_vector b, 
        magma_z_vector *x, magma_z_solver_par *solver_par ){

    // prepare solver feedback
    solver_par->solver = Magma_BICGSTABMERGE2;
    solver_par->numiter = 0;
    solver_par->info = 0;

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    
    magma_int_t dofs = A.num_rows;

    // GPU stream
    magma_queue_t stream[2];
    magma_event_t event[1];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_event_create( &event[0] );

    // workspace
    magma_z_vector q, r,rr,p,v,s,t;
    magmaDoubleComplex *d1, *d2, *skp;
    magma_zmalloc( &d1, dofs*(2) );
    magma_zmalloc( &d2, dofs*(2) );

    // array for the parameters
    magma_zmalloc( &skp, 8 );     
    // skp = [alpha|beta|omega|rho_old|rho|nom|tmp1|tmp2]  
    magma_z_vinit( &q, Magma_DEV, dofs*6, c_zero );

    // q = rr|r|p|v|s|t
    rr.memory_location = Magma_DEV; rr.val = NULL; rr.num_rows = rr.nnz = dofs;
    r.memory_location = Magma_DEV; r.val = NULL; r.num_rows = r.nnz = dofs;
    p.memory_location = Magma_DEV; p.val = NULL; p.num_rows = p.nnz = dofs;
    v.memory_location = Magma_DEV; v.val = NULL; v.num_rows = v.nnz = dofs;
    s.memory_location = Magma_DEV; s.val = NULL; s.num_rows = s.nnz = dofs;
    t.memory_location = Magma_DEV; t.val = NULL; t.num_rows = t.nnz = dofs;

    rr.val = q(0);
    r.val = q(1);
    p.val = q(2);
    v.val = q(3);
    s.val = q(4);
    t.val = q(5);
    
    // solver variables
    magmaDoubleComplex alpha, beta, omega, rho_old, rho_new, *skp_h;
    double nom, nom0, betanom, r0, den;

    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                            // x = 0
    magma_zcopy( dofs, b.val, 1, q(0), 1 );                            // rr = b
    magma_zcopy( dofs, b.val, 1, q(1), 1 );                            // r = b

    rho_new = magma_zdotc( dofs, r.val, 1, r.val, 1 );           // rho=<rr,r>
    nom = MAGMA_Z_REAL(magma_zdotc( dofs, r.val, 1, r.val, 1 ));    
    nom0 = betanom = sqrt(nom);                                 // nom = || r ||   
    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    beta = rho_new;
    solver_par->init_res = nom0;
    // array on host for the parameters 
    magma_zmalloc_cpu( &skp_h, 8 );
    skp_h[0]=alpha; 
    skp_h[1]=beta; 
    skp_h[2]=omega; 
    skp_h[3]=rho_old; 
    skp_h[4]=rho_new; 
    skp_h[5]=MAGMA_Z_MAKE(nom, 0.0);
    magma_zsetvector( 8, skp_h, 1, skp, 1 );

    magma_z_spmv( c_one, A, r, c_zero, v );                     // z = A r
    den = MAGMA_Z_REAL( magma_zdotc(dofs, v.val, 1, r.val, 1) );// den = z dot r

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

        magmablasSetKernelStream(stream[0]);

        // computes p=r+beta*(p-omega*v)
        magma_zbicgmerge1( dofs, skp, v.val, r.val, p.val );
        magma_zbicgmerge_spmv1(  A, d1, d2, q(2), q(0), q(3), skp );          
        magma_zbicgmerge2( dofs, skp, r.val, v.val, s.val );   // s=r-alpha*v
        magma_zbicgmerge_spmv2( A, d1, d2, q(4), q(5), skp); 
        magma_zbicgmerge_xrbeta( dofs, d1, d2, q(0), q(1), q(2), 
                                                    q(4), q(5), x->val, skp);  

        // check stopping criterion (asynchronous copy)
        magma_zgetvector_async( 1 , skp+5, 1, 
                                                        skp_h+5, 1, stream[1] );

        betanom = sqrt(MAGMA_Z_REAL(skp_h[5]));

        if( solver_par->verbose > 0 ){
            magma_device_sync(); tempo2=magma_wtime();
            if( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose] 
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if (  betanom  < r0 ) {
            break;
        }
    }
    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    magma_zresidual( A, b, *x, &residual );
    solver_par->iter_res = betanom;
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
    magma_z_vfree(&q);  // frees all vectors

    magma_free(d1);
    magma_free(d2);
    magma_free( skp );
    magma_free_cpu( skp_h );

    return MAGMA_SUCCESS;
}   /* zbicgstab_merge2 */


