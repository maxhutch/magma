/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zptfqmr_merge.cpp normal z -> c, Mon May  2 23:30:58 2016
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex matrix A.
    This is a GPU implementation of the preconditioned 
    transpose-free Quasi-Minimal Residual method (TFQMR).

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
                
    @param[in,out]
    precond_par magma_c_preconditioner*
                preconditioner
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgesv
    ********************************************************************/

extern "C" magma_int_t
magma_cptfqmr_merge(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_TFQMRMERGE;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    // local variables
    magmaFloatComplex c_zero = MAGMA_C_ZERO, c_one = MAGMA_C_ONE;
    // solver variables
    float nom0, r0,  res, nomb;
    magmaFloatComplex rho = c_one, rho_l = c_one, eta = c_zero , c = c_zero , 
                        theta = c_zero , tau = c_zero, alpha = c_one, beta = c_zero,
                        sigma = c_zero;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // GPU workspace
    magma_c_matrix r={Magma_CSR}, r_tld={Magma_CSR}, pu_m={Magma_CSR},
                    d={Magma_CSR}, w={Magma_CSR}, v={Magma_CSR}, t={Magma_CSR},
                    u_mp1={Magma_CSR}, u_m={Magma_CSR}, Au={Magma_CSR}, 
                    Ad={Magma_CSR}, Au_new={Magma_CSR};
    CHECK( magma_cvinit( &t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &u_mp1,Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_cvinit( &r_tld,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &u_m, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_cvinit( &pu_m, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_cvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &w, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_cvinit( &Ad, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &Au_new, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &Au, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    
    // solver setup
    CHECK(  magma_cresidualvec( A, b, *x, &r, &nom0, queue));
    solver_par->init_res = nom0;
    magma_ccopy( dofs, r.dval, 1, r_tld.dval, 1, queue );   
    magma_ccopy( dofs, r.dval, 1, w.dval, 1, queue );   
    magma_ccopy( dofs, r.dval, 1, u_m.dval, 1, queue );  
    
    // preconditioner
    CHECK( magma_c_applyprecond_left( MagmaNoTrans, A, u_m, &t, precond_par, queue ));
    CHECK( magma_c_applyprecond_right( MagmaNoTrans, A, t, &pu_m, precond_par, queue ));
    
    CHECK( magma_c_spmv( c_one, A, pu_m, c_zero, v, queue ));   // v = A u
    magma_ccopy( dofs, v.dval, 1, Au.dval, 1, queue );  
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
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom0 < r0 ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }

    tau = magma_csqrt( magma_cdotc( dofs, r.dval, 1, r_tld.dval, 1, queue) );
    rho = magma_cdotc( dofs, r.dval, 1, r_tld.dval, 1, queue );
    rho_l = rho;
    
    //Chronometry
    real_Double_t tempo1, tempo2, tempop1, tempop2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        
        // do this every iteration as unrolled
        alpha = rho / magma_cdotc( dofs, v.dval, 1, r_tld.dval, 1, queue );
        sigma = theta * theta / alpha * eta; 
        
        magma_ctfqmr_1(  
        r.num_rows, 
        r.num_cols, 
        alpha,
        sigma,
        v.dval, 
        Au.dval,
        u_m.dval,
        pu_m.dval,
        u_mp1.dval,
        w.dval, 
        d.dval,
        Ad.dval,
        queue );
        
        theta = magma_csqrt( magma_cdotc(dofs, w.dval, 1, w.dval, 1, queue) ) / tau;
        c = c_one / magma_csqrt( c_one + theta*theta );
        tau = tau * theta *c;
        eta = c * c * alpha;
        sigma = theta * theta / alpha * eta;  
        
        magma_ctfqmr_2(  
        r.num_rows, 
        r.num_cols, 
        eta,
        d.dval,
        Ad.dval,
        x->dval, 
        r.dval, 
        queue );
        
        res = magma_scnrm2( dofs, r.dval, 1, queue );
        
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
            info = MAGMA_SUCCESS;
            break;
        }

        // preconditioner
        tempop1 = magma_sync_wtime( queue );
        CHECK( magma_c_applyprecond_left( MagmaNoTrans, A, u_mp1, &t, precond_par, queue ));
        CHECK( magma_c_applyprecond_right( MagmaNoTrans, A, t, &pu_m, precond_par, queue ));
        tempop2 = magma_sync_wtime( queue );
        precond_par->runtime += tempop2-tempop1;
        
        CHECK( magma_c_spmv( c_one, A, pu_m, c_zero, Au_new, queue )); // Au_new = A u_mp1
        solver_par->spmv_count++;
        magma_ccopy( dofs, Au_new.dval, 1, Au.dval, 1, queue );  
        magma_ccopy( dofs, u_mp1.dval, 1, u_m.dval, 1, queue );  

        // here starts the second part of the loop #################################
        magma_ctfqmr_5(  
        r.num_rows, 
        r.num_cols, 
        alpha,
        sigma,
        v.dval, 
        Au.dval,
        pu_m.dval,
        w.dval, 
        d.dval,
        Ad.dval,
        queue ); 
        
        sigma = theta * theta / alpha * eta;  
        
        theta = magma_csqrt( magma_cdotc(dofs, w.dval, 1, w.dval, 1, queue) ) / tau;
        c = c_one / magma_csqrt( c_one + theta*theta );
        tau = tau * theta *c;
        eta = c * c * alpha;

        magma_ctfqmr_2(  
        r.num_rows, 
        r.num_cols, 
        eta,
        d.dval,
        Ad.dval,
        x->dval, 
        r.dval, 
        queue );
        
        res = magma_scnrm2( dofs, r.dval, 1, queue );
        
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
            info = MAGMA_SUCCESS;
            break;
        }
        
        rho = magma_cdotc( dofs, w.dval, 1, r_tld.dval, 1, queue );
        beta = rho / rho_l;
        rho_l = rho;
        
        magma_ctfqmr_3(  
        r.num_rows, 
        r.num_cols, 
        beta,
        w.dval,
        u_m.dval,
        u_mp1.dval, 
        queue );
              
        // preconditioner
        tempop1 = magma_sync_wtime( queue );
        CHECK( magma_c_applyprecond_left( MagmaNoTrans, A, u_mp1, &t, precond_par, queue ));
        CHECK( magma_c_applyprecond_right( MagmaNoTrans, A, t, &pu_m, precond_par, queue ));
        tempop2 = magma_sync_wtime( queue );
        precond_par->runtime += tempop2-tempop1;
        
        CHECK( magma_c_spmv( c_one, A, pu_m, c_zero, Au_new, queue )); // Au_new = A pu_m

        solver_par->spmv_count++;
        magma_ctfqmr_4(  
        r.num_rows, 
        r.num_cols, 
        beta,
        Au_new.dval,
        v.dval,
        Au.dval, 
        queue );
        
        magma_ccopy( dofs, u_mp1.dval, 1, u_m.dval, 1, queue ); 
    
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    float residual;
    CHECK(  magma_cresidualvec( A, b, *x, &r, &residual, queue));
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
    magma_cmfree(&r, queue );
    magma_cmfree(&r_tld, queue );
    magma_cmfree(&d, queue );
    magma_cmfree(&w, queue );
    magma_cmfree(&v, queue );
    magma_cmfree(&u_m, queue );
    magma_cmfree(&u_mp1, queue );
    magma_cmfree(&pu_m, queue );
    magma_cmfree(&d, queue );
    magma_cmfree(&t, queue );
    magma_cmfree(&Au, queue );
    magma_cmfree(&Au_new, queue );
    magma_cmfree(&Ad, queue );
    
    solver_par->info = info;
    return info;
}   /* magma_cptfqmr_merge */
