/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/src/zpbicgstab.cpp normal z -> c, Mon May  2 23:31:00 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"


#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex N-by-N general matrix.
    This is a GPU implementation of the preconditioned
    Biconjugate Gradient Stabelized method.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A

    @param[in]
    b           magma_c_matrix
                RHS b

    @param[in,out]
    x           magma_c_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_c_solver_par*
                solver parameters

    @param[in]
    precond_par magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgesv
    ********************************************************************/

extern "C" magma_int_t
magma_cpbicgstab(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_PBICGSTAB;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;

    // some useful variables
    magmaFloatComplex c_zero = MAGMA_C_ZERO;
    magmaFloatComplex c_one  = MAGMA_C_ONE;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    
    magma_int_t dofs = A.num_rows*b.num_cols;

    // workspace
    magma_c_matrix r={Magma_CSR}, rr={Magma_CSR}, p={Magma_CSR}, v={Magma_CSR}, s={Magma_CSR}, t={Magma_CSR}, ms={Magma_CSR}, mt={Magma_CSR}, y={Magma_CSR}, z={Magma_CSR};
    CHECK( magma_cvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &rr,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &ms,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &mt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &y, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // solver variables
    magmaFloatComplex alpha, beta, omega, rho_old, rho_new;
    float nom, betanom, nom0, r0, res, nomb;
    res=0;
    //float den;

    // solver setup
    CHECK(  magma_cresidualvec( A, b, *x, &r, &nom0, queue));
    magma_ccopy( dofs, r.dval, 1, rr.dval, 1, queue );                  // rr = r
    betanom = nom0;
    nom = nom0*nom0;
    rho_new = omega = alpha = MAGMA_C_MAKE( 1.0, 0. );
    solver_par->init_res = nom0;

    nomb = magma_scnrm2( dofs, b.dval, 1, queue );
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

        rho_new = magma_cdotc( dofs, rr.dval, 1, r.dval, 1, queue );  // rho=<rr,r>
        beta = rho_new/rho_old * alpha/omega;   // beta=rho/rho_old *alpha/omega
        if( magma_c_isnan_inf( beta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        magma_cscal( dofs, beta, p.dval, 1, queue );                 // p = beta*p
        magma_caxpy( dofs, c_neg_one * omega * beta, v.dval, 1 , p.dval, 1, queue );
                                                        // p = p-omega*beta*v
        magma_caxpy( dofs, c_one, r.dval, 1, p.dval, 1, queue );      // p = p+r

        // preconditioner
        tempop1 = magma_sync_wtime( queue );
        CHECK( magma_c_applyprecond_left( MagmaNoTrans, A, p, &mt, precond_par, queue ));
        CHECK( magma_c_applyprecond_right( MagmaNoTrans, A, mt, &y, precond_par, queue ));
        tempop2 = magma_sync_wtime( queue );
        precond_par->runtime += tempop2-tempop1;
        
        CHECK( magma_c_spmv( c_one, A, y, c_zero, v, queue ));      // v = Ap
        solver_par->spmv_count++;
        alpha = rho_new / magma_cdotc( dofs, rr.dval, 1, v.dval, 1, queue );
        if( magma_c_isnan_inf( alpha ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        magma_ccopy( dofs, r.dval, 1 , s.dval, 1, queue );            // s=r
        magma_caxpy( dofs, c_neg_one * alpha, v.dval, 1 , s.dval, 1, queue ); // s=s-alpha*v

        // preconditioner
        tempop1 = magma_sync_wtime( queue );
        CHECK( magma_c_applyprecond_left( MagmaNoTrans, A, s, &ms, precond_par, queue ));
        CHECK( magma_c_applyprecond_right( MagmaNoTrans, A, ms, &z, precond_par, queue ));
        tempop2 = magma_sync_wtime( queue );
        precond_par->runtime += tempop2-tempop1;
        
        CHECK( magma_c_spmv( c_one, A, z, c_zero, t, queue ));       // t=As
        solver_par->spmv_count++;                  
       // omega = <s,t>/<t,t>
        omega = magma_cdotc( dofs, t.dval, 1, s.dval, 1, queue )
                   / magma_cdotc( dofs, t.dval, 1, t.dval, 1, queue );

        magma_caxpy( dofs, alpha, y.dval, 1 , x->dval, 1, queue );     // x=x+alpha*p
        magma_caxpy( dofs, omega, z.dval, 1 , x->dval, 1, queue );     // x=x+omega*s

        magma_ccopy( dofs, s.dval, 1 , r.dval, 1, queue );             // r=s
        magma_caxpy( dofs, c_neg_one * omega, t.dval, 1 , r.dval, 1, queue ); // r=r-omega*t
        res = betanom = magma_scnrm2( dofs, r.dval, 1, queue );

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
    float residual;
    CHECK(  magma_cresidualvec( A, b, *x, &r, &residual, queue));
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
    magma_cmfree(&r, queue );
    magma_cmfree(&rr, queue );
    magma_cmfree(&p, queue );
    magma_cmfree(&v, queue );
    magma_cmfree(&s, queue );
    magma_cmfree(&t, queue );
    magma_cmfree(&ms, queue );
    magma_cmfree(&mt, queue );
    magma_cmfree(&y, queue );
    magma_cmfree(&z, queue );

    solver_par->info = info;
    return info;
}   /* magma_cbicgstab */
