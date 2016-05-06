/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/src/zpbicgstab_merge.cpp normal z -> d, Mon May  2 23:31:01 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a general matrix.
    This is a GPU implementation of the preconditioned
    Biconjugate Gradient Stabelized method.

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
    precond_par magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgesv
    ********************************************************************/

extern "C" magma_int_t
magma_dpbicgstab_merge(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_BICGSTAB;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;

    // some useful variables
    double c_zero = MAGMA_D_ZERO;
    double c_one  = MAGMA_D_ONE;
    
    magma_int_t dofs = A.num_rows * b.num_cols;

    // workspace
    magma_d_matrix r={Magma_CSR}, rr={Magma_CSR}, p={Magma_CSR}, v={Magma_CSR}, 
    z={Magma_CSR}, y={Magma_CSR}, ms={Magma_CSR}, mt={Magma_CSR}, 
    s={Magma_CSR}, t={Magma_CSR}, d1={Magma_CSR}, d2={Magma_CSR};
    CHECK( magma_dvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &rr,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &ms,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &mt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &y, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &d1, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &d2, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // solver variables
    double alpha, beta, omega, rho_old, rho_new;
    double nom, betanom, nom0, r0, res, nomb;
    res=0;
    //double den;

    // solver setup
    CHECK(  magma_dresidualvec( A, b, *x, &r, &nom0, queue));
    magma_dcopy( dofs, r.dval, 1, rr.dval, 1, queue );                  // rr = r
    betanom = nom0;
    nom = nom0*nom0;
    rho_new = magma_ddot( dofs, r.dval, 1, r.dval, 1, queue );             // rho=<rr,r>
    rho_old = omega = alpha = MAGMA_D_MAKE( 1.0, 0. );
    solver_par->init_res = nom0;

    CHECK( magma_d_spmv( c_one, A, r, c_zero, v, queue ));              // z = A r

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
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom < r0 ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }
    
    //Chronometry
    real_Double_t tempo1, tempo2, tempop1, tempop2;
    tempo1 = magma_sync_wtime( queue );

    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        rho_old = rho_new;                                    // rho_old=rho

        rho_new = magma_ddot( dofs, rr.dval, 1, r.dval, 1, queue );  // rho=<rr,r>
        beta = rho_new/rho_old * alpha/omega;   // beta=rho/rho_old *alpha/omega
        if( magma_d_isnan_inf( beta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        // p = r + beta * ( p - omega * v )
        magma_dbicgstab_1(  
        r.num_rows, 
        r.num_cols, 
        beta,
        omega,
        r.dval, 
        v.dval,
        p.dval,
        queue );

        // preconditioner
        tempop1 = magma_sync_wtime( queue );
        CHECK( magma_d_applyprecond_left( MagmaNoTrans, A, p, &mt, precond_par, queue ));
        CHECK( magma_d_applyprecond_right( MagmaNoTrans, A, mt, &y, precond_par, queue ));
        tempop2 = magma_sync_wtime( queue );
        precond_par->runtime += tempop2-tempop1;

        CHECK( magma_d_spmv( c_one, A, y, c_zero, v, queue ));      // v = Ap
        solver_par->spmv_count++;
        //alpha = rho_new / tmpval;
        alpha = rho_new /magma_ddot( dofs, rr.dval, 1, v.dval, 1, queue );
        if( magma_d_isnan_inf( alpha ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        // s = r - alpha v
        magma_dbicgstab_2(  
        r.num_rows, 
        r.num_cols, 
        alpha,
        r.dval,
        v.dval,
        s.dval, 
        queue );
        
        // preconditioner
        tempop1 = magma_sync_wtime( queue );
        CHECK( magma_d_applyprecond_left( MagmaNoTrans, A, s, &ms, precond_par, queue ));
        CHECK( magma_d_applyprecond_right( MagmaNoTrans, A, ms, &z, precond_par, queue ));
        tempop2 = magma_sync_wtime( queue );
        precond_par->runtime += tempop2-tempop1;

        CHECK( magma_d_spmv( c_one, A, z, c_zero, t, queue ));       // t=As
        solver_par->spmv_count++;
        omega = magma_ddot( dofs, t.dval, 1, s.dval, 1, queue )   // omega = <s,t>/<t,t>
                   / magma_ddot( dofs, t.dval, 1, t.dval, 1, queue );
                        
        // x = x + alpha * y + omega * z
        // r = s - omega * t
        magma_dbicgstab_4(  
        r.num_rows, 
        r.num_cols, 
        alpha,
        omega,
        y.dval,
        z.dval,
        s.dval,
        t.dval,
        x->dval,
        r.dval,
        queue );

        res = betanom = magma_dnrm2( dofs, r.dval, 1, queue );

        nom = betanom*betanom;

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
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
    solver_par->final_res = residual;
    solver_par->iter_res = res;

    if ( solver_par->numiter < solver_par->maxiter && info == MAGMA_SUCCESS ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) betanom;
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
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_dmfree(&r, queue );
    magma_dmfree(&rr, queue );
    magma_dmfree(&p, queue );
    magma_dmfree(&v, queue );
    magma_dmfree(&s, queue );
    magma_dmfree(&y, queue );
    magma_dmfree(&z, queue );
    magma_dmfree(&t, queue );
    magma_dmfree(&ms, queue );
    magma_dmfree(&mt, queue );
    magma_dmfree(&d1, queue );
    magma_dmfree(&d2, queue );

    solver_par->info = info;
    return info;
}   /* magma_dbicgstab_merge */
