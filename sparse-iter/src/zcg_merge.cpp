/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @author Hartwig Anzt 

       @precisions normal z -> s d c
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
    This is a GPU implementation of the Conjugate Gradient method in variant,
    where multiple operations are merged into one compute kernel.    

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

    @ingroup magmasparse_zhesv
    ********************************************************************/

magma_int_t
magma_zcg_merge( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_z_solver_par *solver_par ){

    // prepare solver feedback
    solver_par->solver = Magma_CGMERGE;
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

    // GPU workspace
    magma_z_vector r, d, z;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &d, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &z, Magma_DEV, dofs, c_zero );
    
    magmaDoubleComplex *d1, *d2, *skp;
    magma_zmalloc( &d1, dofs*(1) );
    magma_zmalloc( &d2, dofs*(1) );
    // array for the parameters
    magma_zmalloc( &skp, 6 );       // skp = [alpha|beta|gamma|rho|tmp1|tmp2]


    // solver variables
    magmaDoubleComplex alpha, beta, gamma, rho, tmp1, *skp_h;
    double nom, nom0, r0, betanom, den;

    // solver setup
    magma_zscal( dofs, c_zero, x->val, 1) ;                     // x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );                    // r = b
    magma_zcopy( dofs, b.val, 1, d.val, 1 );                    // d = b
    nom0 = betanom = magma_dznrm2( dofs, r.val, 1 );               
    nom = nom0 * nom0;                                           // nom = r' * r
    magma_z_spmv( c_one, A, d, c_zero, z );                      // z = A d
    den = MAGMA_Z_REAL( magma_zdotc(dofs, d.val, 1, z.val, 1) ); // den = d'* z
    solver_par->init_res = nom0;
    
    // array on host for the parameters
    magma_zmalloc_cpu( &skp_h, 6 );

    alpha = rho = gamma = tmp1 = c_one; 
    beta =  magma_zdotc(dofs, r.val, 1, r.val, 1);
    skp_h[0]=alpha; 
    skp_h[1]=beta; 
    skp_h[2]=gamma; 
    skp_h[3]=rho; 
    skp_h[4]=tmp1; 
    skp_h[5]=MAGMA_Z_MAKE(nom, 0.0);

    magma_zsetvector( 6, skp_h, 1, skp, 1 );
    
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
        solver_par->res_vec[0] = (real_Double_t) nom0;
        solver_par->timing[0] = 0.0;
    }
    
    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){

        magmablasSetKernelStream(stream[0]);
        
        // computes SpMV and dot product
        magma_zcgmerge_spmv1(  A, d1, d2, d.val, z.val, skp ); 
            
        // updates x, r, computes scalars and updates d
        magma_zcgmerge_xrbeta( dofs, d1, d2, x->val, r.val, d.val, z.val, skp ); 

        // check stopping criterion (asynchronous copy)
        magma_zgetvector_async( 1 , skp+1, 1, 
                                                    skp_h+1, 1, stream[1] );
        betanom = sqrt(MAGMA_Z_REAL(skp_h[1]));

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
    magma_z_vfree(&r);
    magma_z_vfree(&z);
    magma_z_vfree(&d);

    magma_free( d1 );
    magma_free( d2 );
    magma_free( skp );
    magma_free_cpu( skp_h );

    return MAGMA_SUCCESS;
}   /* magma_zcg_merge */


