/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/ztfqmr_unrolled.cpp normal z -> d, Mon May  2 23:30:57 2016
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a real matrix A.
    This is a GPU implementation of the transpose-free Quasi-Minimal Residual 
    method (TFQMR).

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
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgesv
    ********************************************************************/

extern "C" magma_int_t
magma_dtfqmr_unrolled(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    

    // prepare solver feedback
    solver_par->solver = Magma_TFQMR;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    solver_par->spmv_count = 0;
    
    // local variables
    double c_zero = MAGMA_D_ZERO, c_one = MAGMA_D_ONE;
    // solver variables
    double nom0, r0,  res, nomb;
    double rho = c_one, rho_l = c_one, eta = c_zero , c = c_zero , 
                        theta = c_zero , tau = c_zero, alpha = c_one, beta = c_zero,
                        sigma = c_zero;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // GPU workspace
    magma_d_matrix r={Magma_CSR}, r_tld={Magma_CSR},
                    d={Magma_CSR}, w={Magma_CSR}, v={Magma_CSR},
                    u_mp1={Magma_CSR}, u_m={Magma_CSR}, Au={Magma_CSR}, 
                    Ad={Magma_CSR}, Au_new={Magma_CSR};
    CHECK( magma_dvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &u_mp1,Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_dvinit( &r_tld,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &u_m, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_dvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &w, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    CHECK( magma_dvinit( &Ad, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Au_new, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &Au, Magma_DEV, A.num_rows, b.num_cols, c_one, queue ));
    
    // solver setup
    CHECK(  magma_dresidualvec( A, b, *x, &r, &nom0, queue));
    solver_par->init_res = nom0;
    magma_dcopy( dofs, r.dval, 1, r_tld.dval, 1, queue );   
    magma_dcopy( dofs, r.dval, 1, w.dval, 1, queue );   
    magma_dcopy( dofs, r.dval, 1, u_mp1.dval, 1, queue );   
    CHECK( magma_d_spmv( c_one, A, u_mp1, c_zero, v, queue ));   // v = A u
    magma_dcopy( dofs, v.dval, 1, Au.dval, 1, queue );  
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
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom0 < r0 ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }

    tau = magma_dsqrt( magma_ddot( dofs, r.dval, 1, r_tld.dval, 1, queue ) );
    rho = magma_ddot( dofs, r.dval, 1, r_tld.dval, 1, queue );
    rho_l = rho;
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        
        // do this every iteration as unrolled
        alpha = rho / magma_ddot( dofs, v.dval, 1, r_tld.dval, 1, queue );
        sigma = theta * theta / alpha * eta; 
        
        magma_daxpy( dofs,  -alpha, v.dval, 1, u_mp1.dval, 1, queue );     // u_mp1 = u_mp_1 - alpha*v;
        magma_daxpy( dofs,  -alpha, Au.dval, 1, w.dval, 1, queue );     // w = w - alpha*Au;
        magma_dscal( dofs, sigma, d.dval, 1, queue );    
        magma_daxpy( dofs, c_one, u_mp1.dval, 1, d.dval, 1, queue );     // d = u_mp1 + sigma*d;
        //magma_dscal( dofs, sigma, Ad.dval, 1, queue );         
        //magma_daxpy( dofs, c_one, Au.dval, 1, Ad.dval, 1, queue );     // Ad = Au + sigma*Ad;
        
        theta = magma_dsqrt( magma_ddot(dofs, w.dval, 1, w.dval, 1, queue ) ) / tau;
        c = c_one / magma_dsqrt( c_one + theta*theta );
        tau = tau * theta *c;
        eta = c * c * alpha;
        sigma = theta * theta / alpha * eta;  
        printf("sigma: %f+%fi\n", MAGMA_D_REAL(sigma), MAGMA_D_IMAG(sigma) );
        CHECK( magma_d_spmv( c_one, A, d, c_zero, Ad, queue )); // Au_new = A u_mp1
        solver_par->spmv_count++;
      
        magma_daxpy( dofs, eta, d.dval, 1, x->dval, 1, queue );     // x = x + eta * d
        magma_daxpy( dofs, -eta, Ad.dval, 1, r.dval, 1, queue );     // r = r - eta * Ad

    
        // here starts the second part of the loop #################################
        

        magma_daxpy( dofs,  -alpha, Au.dval, 1, w.dval, 1, queue );     // w = w - alpha*Au;
        magma_dscal( dofs, sigma, d.dval, 1, queue );    
        magma_daxpy( dofs, c_one, u_mp1.dval, 1, d.dval, 1, queue );     // d = u_mp1 + sigma*d;
        magma_dscal( dofs, sigma, Ad.dval, 1, queue );         
        magma_daxpy( dofs, c_one, Au.dval, 1, Ad.dval, 1, queue );     // Ad = Au + sigma*Ad;

        
        theta = magma_dsqrt( magma_ddot(dofs, w.dval, 1, w.dval, 1, queue ) ) / tau;
        c = c_one / magma_dsqrt( c_one + theta*theta );
        tau = tau * theta *c;
        eta = c * c * alpha;

        magma_daxpy( dofs, eta, d.dval, 1, x->dval, 1, queue );     // x = x + eta * d
        magma_daxpy( dofs, -eta, Ad.dval, 1, r.dval, 1, queue );     // r = r - eta * Ad
        
        res = magma_dnrm2( dofs, r.dval, 1, queue );
        
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
            break;
        }
        // do this every loop as unrolled
        rho_l = rho;
        rho = magma_ddot( dofs, w.dval, 1, r_tld.dval, 1, queue );
        beta = rho / rho_l;
        magma_dscal( dofs, beta, u_mp1.dval, 1, queue ); 
        magma_daxpy( dofs, c_one, w.dval, 1, u_mp1.dval, 1, queue );         // u_mp1 = w + beta*u_mp1;
              
        CHECK( magma_d_spmv( c_one, A, u_mp1, c_zero, Au_new, queue )); // Au_new = A u_mp1
        solver_par->spmv_count++;
        // do this every loop as unrolled
        magma_dscal( dofs, beta*beta, v.dval, 1, queue );                    
        magma_daxpy( dofs, beta, Au.dval, 1, v.dval, 1, queue );              
        magma_daxpy( dofs, c_one, Au_new.dval, 1, v.dval, 1, queue );      // v = Au_new + beta*(Au+beta*v);
        
        magma_dcopy( dofs, Au_new.dval, 1, Au.dval, 1, queue );  
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_dresidualvec( A, b, *x, &r, &residual, queue));
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
    magma_dmfree(&r, queue );
    magma_dmfree(&r_tld, queue );
    magma_dmfree(&d, queue );
    magma_dmfree(&w, queue );
    magma_dmfree(&v, queue );
    magma_dmfree(&u_m, queue );
    magma_dmfree(&u_mp1, queue );
    magma_dmfree(&d, queue );
    magma_dmfree(&Au, queue );
    magma_dmfree(&Au_new, queue );
    magma_dmfree(&Ad, queue );
    
    solver_par->info = info;
    return info;
}   /* magma_dfqmr_unrolled */
