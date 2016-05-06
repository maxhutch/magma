/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zqmr.cpp normal z -> s, Mon May  2 23:30:56 2016
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a general real matrix A.
    This is a GPU implementation of the Quasi-Minimal Residual method (QMR).

    Arguments
    ---------

    @param[in]
    A           magma_s_matrix
                input matrix A

    @param[in]
    b           magma_s_matrix
                RHS b

    @param[in,out]
    x           magma_s_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_s_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup    magmasparse_sgesv
    ********************************************************************/

extern "C" magma_int_t
magma_sqmr(
    magma_s_matrix A, magma_s_matrix b, magma_s_matrix *x,
    magma_s_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_QMR;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    
    // local variables
    float c_zero = MAGMA_S_ZERO, c_one = MAGMA_S_ONE;
    // solver variables
    float nom0, r0, res=0, nomb;
    float rho = c_one, rho1 = c_one, eta = -c_one , pds = c_one, 
                        thet = c_one, thet1 = c_one, epsilon = c_one, 
                        beta = c_one, delta = c_one, pde = c_one, rde = c_one,
                        gamm = c_one, gamm1 = c_one, psi = c_one;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // need to transpose the matrix
    magma_s_matrix AT={Magma_CSR}, Ah1={Magma_CSR}, Ah2={Magma_CSR};
    
    // GPU workspace
    magma_s_matrix r={Magma_CSR}, r_tld={Magma_CSR},
                    v={Magma_CSR}, w={Magma_CSR}, wt={Magma_CSR},
                    d={Magma_CSR}, s={Magma_CSR}, z={Magma_CSR}, q={Magma_CSR}, 
                    p={Magma_CSR}, pt={Magma_CSR}, y={Magma_CSR};
    CHECK( magma_svinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &r_tld, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &w, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &wt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &pt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &y, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // solver setup
    CHECK(  magma_sresidualvec( A, b, *x, &r, &nom0, queue));
    solver_par->init_res = nom0;
    magma_scopy( dofs, r.dval, 1, r_tld.dval, 1, queue );   
    magma_scopy( dofs, r.dval, 1, y.dval, 1, queue );   
    magma_scopy( dofs, r.dval, 1, v.dval, 1, queue );  
    magma_scopy( dofs, r.dval, 1, wt.dval, 1, queue );   
    magma_scopy( dofs, r.dval, 1, z.dval, 1, queue );  
    
    // transpose the matrix
    magma_smtransfer( A, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_smconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_smfree(&Ah1, queue );
    magma_smtransposeconjugate( Ah2, &Ah1, queue );
    magma_smfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_smconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_smfree(&Ah1, queue );
    magma_smtransfer( Ah2, &AT, Magma_CPU, Magma_DEV, queue );
    magma_smfree(&Ah2, queue );
    
    nomb = magma_snrm2( dofs, b.dval, 1, queue );
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

    psi = magma_ssqrt( magma_sdot( dofs, z.dval, 1, z.dval, 1, queue ));
    rho = magma_ssqrt( magma_sdot( dofs, y.dval, 1, y.dval, 1, queue ));
        // v = y / rho
        // y = y / rho
        // w = wt / psi
        // z = z / psi
    magma_sscal( dofs, c_one / rho, v.dval, 1, queue ); 
    magma_sscal( dofs, c_one / rho, y.dval, 1, queue ); 
    magma_sscal( dofs, c_one / psi, w.dval, 1, queue ); 
    magma_sscal( dofs, c_one / psi, z.dval, 1, queue ); 
    
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        if( magma_s_isnan_inf( rho ) || magma_s_isnan_inf( psi ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
 
            // delta = z' * y;
        delta = magma_sdot( dofs, z.dval, 1, y.dval, 1, queue );
        
        if( magma_s_isnan_inf( delta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        // no precond: yt = y, zt = z
        //magma_scopy( dofs, y.dval, 1, yt.dval, 1 );
        //magma_scopy( dofs, z.dval, 1, zt.dval, 1 );
        
        if( solver_par->numiter == 1 ){
                // p = y;
                // q = z;
            magma_scopy( dofs, y.dval, 1, p.dval, 1, queue );
            magma_scopy( dofs, z.dval, 1, q.dval, 1, queue );
        }
        else{
            pde = psi * delta / epsilon;
            rde = rho * MAGMA_S_CONJ(delta/epsilon);
                // p = y - pde * p;
            magma_sscal( dofs, -pde, p.dval, 1, queue );    
            magma_saxpy( dofs, c_one, y.dval, 1, p.dval, 1, queue );
                // q = z - rde * q;
            magma_sscal( dofs, -rde, q.dval, 1, queue );    
            magma_saxpy( dofs, c_one, z.dval, 1, q.dval, 1, queue );
        }
        if( magma_s_isnan_inf( rho ) || magma_s_isnan_inf( psi ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        CHECK( magma_s_spmv( c_one, A, p, c_zero, pt, queue ));
        solver_par->spmv_count++;
            // epsilon = q' * pt;
        epsilon = magma_sdot( dofs, q.dval, 1, pt.dval, 1, queue );
        beta = epsilon / delta;

        if( magma_s_isnan_inf( epsilon ) || magma_s_isnan_inf( beta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
            // v = pt - beta * v;
        magma_sscal( dofs, -beta, v.dval, 1, queue ); 
        magma_saxpy( dofs, c_one, pt.dval, 1, v.dval, 1, queue ); 
            // no precond: y = v
        magma_scopy( dofs, v.dval, 1, y.dval, 1, queue );
        
        rho1 = rho;      
            // rho = norm(y);
        rho = magma_ssqrt( magma_sdot( dofs, y.dval, 1, y.dval, 1, queue ));
        
            // wt = A' * q - beta' * w;
        CHECK( magma_s_spmv( c_one, AT, q, c_zero, wt, queue ));
        solver_par->spmv_count++;
        magma_saxpy( dofs, - MAGMA_S_CONJ( beta ), w.dval, 1, wt.dval, 1, queue );  
        
                    // no precond: z = wt
        magma_scopy( dofs, wt.dval, 1, z.dval, 1, queue );
        


        thet1 = thet;        
        thet = rho / (gamm * MAGMA_S_MAKE( MAGMA_S_ABS(beta), 0.0 ));
        gamm1 = gamm;        
        
        gamm = c_one / magma_ssqrt(c_one + thet*thet);        
        eta = - eta * rho1 * gamm * gamm / (beta * gamm1 * gamm1);        

        if( magma_s_isnan_inf( thet ) || magma_s_isnan_inf( gamm ) || magma_s_isnan_inf( eta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        if( solver_par->numiter == 1 ){
                // d = eta * p;
                // s = eta * pt;
            magma_scopy( dofs, p.dval, 1, d.dval, 1, queue );
            magma_sscal( dofs, eta, d.dval, 1, queue );
            magma_scopy( dofs, pt.dval, 1, s.dval, 1, queue );
            magma_sscal( dofs, eta, s.dval, 1, queue );
                // x = x + d;                    
            magma_saxpy( dofs, c_one, d.dval, 1, x->dval, 1, queue );
                // r = r - s;
            magma_saxpy( dofs, -c_one, s.dval, 1, r.dval, 1, queue );
        }
        else{
                // d = eta * p + (thet1 * gamm)^2 * d;
                // s = eta * pt + (thet1 * gamm)^2 * s;
            pds = (thet1 * gamm) * (thet1 * gamm);
            magma_sscal( dofs, pds, d.dval, 1, queue );    
            magma_saxpy( dofs, eta, p.dval, 1, d.dval, 1, queue );
            magma_sscal( dofs, pds, s.dval, 1, queue );    
            magma_saxpy( dofs, eta, pt.dval, 1, s.dval, 1, queue );
                // x = x + d;                    
            magma_saxpy( dofs, c_one, d.dval, 1, x->dval, 1, queue );
                // r = r - s;
            magma_saxpy( dofs, -c_one, s.dval, 1, r.dval, 1, queue );
        }
            // psi = norm(z);
        psi = magma_ssqrt( magma_sdot( dofs, z.dval, 1, z.dval, 1, queue ) );
        
        res = magma_snrm2( dofs, r.dval, 1, queue );
        
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        
        // v = y / rho
        // y = y / rho
        // w = wt / psi
        // z = z / psi
        magma_sqmr_1(  
        r.num_rows, 
        r.num_cols, 
        rho,
        psi,
        y.dval, 
        z.dval,
        v.dval,
        w.dval,
        queue );

        if ( res/nomb <= solver_par->rtol || res <= solver_par->atol ){
            break;
        }
 
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    float residual;
    CHECK(  magma_sresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter && info == MAGMA_SUCCESS ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
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
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_smfree(&r, queue );
    magma_smfree(&r_tld, queue );
    magma_smfree(&v,  queue );
    magma_smfree(&w,  queue );
    magma_smfree(&wt, queue );
    magma_smfree(&d,  queue );
    magma_smfree(&s,  queue );
    magma_smfree(&z,  queue );
    magma_smfree(&q,  queue );
    magma_smfree(&p,  queue );
    magma_smfree(&pt, queue );
    magma_smfree(&y,  queue );
    magma_smfree(&AT, queue );
    magma_smfree(&Ah1, queue );
    magma_smfree(&Ah2, queue );

    
    solver_par->info = info;
    return info;
}   /* magma_sqmr */
