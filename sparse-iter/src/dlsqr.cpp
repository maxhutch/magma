/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zlsqr.cpp normal z -> d, Mon May  2 23:31:02 2016
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations A*X=B
    for X if A is consistent, otherwise it attempts to solve the least
    squares solution X that minimizes norm(B-A*X). The N-by-P coefficient
    matrix A need not be square but the right hand side column vector B
    must have length N.

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
                
    @param[in,out]
    precond_par magma_d_preconditioner*
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup    magmasparse_dgesv
    ********************************************************************/

extern "C" magma_int_t
magma_dlsqr(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_LSQR;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    magma_int_t m = A.num_rows * b.num_cols;
    magma_int_t n = A.num_cols * b.num_cols;
    
    // local variables
    double c_zero = MAGMA_D_ZERO, c_one = MAGMA_D_ONE;
    // solver variables
    double s, nom0, r0, res=0, nomb, phibar, beta, alpha, c, rho, rhot, phi, thet, normr, normar, norma, sumnormd2, normd;

    // need to transpose the matrix
    magma_d_matrix AT={Magma_CSR}, Ah1={Magma_CSR}, Ah2={Magma_CSR};
    
    // GPU workspace
    magma_d_matrix r={Magma_CSR},
                    v={Magma_CSR}, z={Magma_CSR}, zt={Magma_CSR},
                    d={Magma_CSR}, vt={Magma_CSR}, q={Magma_CSR}, 
                    w={Magma_CSR}, u={Magma_CSR};
    CHECK( magma_dvinit( &r, Magma_DEV, A.num_cols, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &v, Magma_DEV, A.num_cols, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &z, Magma_DEV, A.num_cols, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &d, Magma_DEV, A.num_cols, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &vt,Magma_DEV, A.num_cols, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &q, Magma_DEV, A.num_cols, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &w, Magma_DEV, A.num_cols, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &u, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &zt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // transpose the matrix
    magma_dmtransfer( A, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_dmconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_dmfree(&Ah1, queue );
    magma_dmtransposeconjugate( Ah2, &Ah1, queue );
    magma_dmfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_dmconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_dmfree(&Ah1, queue );
    magma_dmtransfer( Ah2, &AT, Magma_CPU, Magma_DEV, queue );
    magma_dmfree(&Ah2, queue );
    

    
    // solver setup
    CHECK(  magma_dresidualvec( A, b, *x, &r, &nom0, queue));
    solver_par->init_res = nom0;
    nomb = magma_dnrm2( m, b.dval, 1, queue );
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
    magma_dcopy( m, b.dval, 1, u.dval, 1, queue );  
    beta = magma_dnrm2( m, u.dval, 1, queue );
    magma_dscal( m, MAGMA_D_MAKE(1./beta, 0.0 ), u.dval, 1, queue );
    normr = beta;
    c = 1.0;
    s = 0.0;
    phibar = beta;
    CHECK( magma_d_spmv( c_one, AT, u, c_zero, v, queue ));
    
    if( precond_par->solver == Magma_NONE ){
        ;
    } else {
      CHECK( magma_d_applyprecond_right( MagmaTrans, A, v, &zt, precond_par, queue ));
      CHECK( magma_d_applyprecond_left( MagmaTrans, A, zt, &v, precond_par, queue ));
    }
    alpha = magma_dnrm2( n, v.dval, 1, queue );
    magma_dscal( n, MAGMA_D_MAKE(1./alpha, 0.0 ), v.dval, 1, queue );
    normar = alpha * beta;
    norma = 0;
    sumnormd2 = 0;
        
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        if( precond_par->solver == Magma_NONE || A.num_rows != A.num_cols ) {
            magma_dcopy( n, v.dval, 1 , z.dval, 1, queue );    
        } else {
            CHECK( magma_d_applyprecond_left( MagmaNoTrans, A, v, &zt, precond_par, queue ));
            CHECK( magma_d_applyprecond_right( MagmaNoTrans, A, zt, &z, precond_par, queue ));
        }
        //CHECK( magma_d_spmv( c_one, A, z, MAGMA_D_MAKE(-alpha,0.0), u, queue ));
        CHECK( magma_d_spmv( c_one, A, z, c_zero, zt, queue ));
        magma_dscal( m, MAGMA_D_MAKE(-alpha, 0.0 ), u.dval, 1, queue ); 
        magma_daxpy( m, c_one, zt.dval, 1, u.dval, 1, queue );
        
        solver_par->spmv_count++;
        beta = magma_dnrm2( m, u.dval, 1, queue );
        magma_dscal( m, MAGMA_D_MAKE(1./beta, 0.0 ), u.dval, 1, queue ); 
        // norma = norm([norma alpha beta]);
        norma = sqrt(norma*norma + alpha*alpha + beta*beta );
        
        //lsvec( solver_par->numiter-1 ) = normar / norma;
        
        thet = -s * alpha;
        rhot = c * alpha;
        rho = sqrt( rhot * rhot + beta * beta );
        c = rhot / rho;
        s = - beta / rho;
        phi = c * phibar;
        phibar = s * phibar;
        
        // d = (z - thet * d) / rho;
        magma_dscal( n, MAGMA_D_MAKE(-thet, 0.0 ), d.dval, 1, queue ); 
        magma_daxpy( n, c_one, z.dval, 1, d.dval, 1, queue );
        magma_dscal( n, MAGMA_D_MAKE(1./rho, 0.0 ), d.dval, 1, queue );
        normd = magma_dnrm2( n, d.dval, 1, queue );
        sumnormd2 = sumnormd2 + normd*normd;
        
        // convergence check
        res = normr;        
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        // check for convergence in A*x=b
        if ( res/nomb <= solver_par->rtol || res <= solver_par->atol ){
            info = MAGMA_SUCCESS;
            break;
        }
        // check for convergence in min{|b-A*x|}
        if ( A.num_rows != A.num_cols &&
               ( normar/(norma*normr) <= solver_par->rtol || normar <= solver_par->atol ) ){
            printf("%% warning: quit from minimization convergence check.\n");
            info = MAGMA_SUCCESS;
            break;
        }
        
        magma_daxpy( n, MAGMA_D_MAKE( phi, 0.0 ), d.dval, 1, x->dval, 1, queue );
        normr = fabs(s) * normr;
        CHECK( magma_d_spmv( c_one, AT, u, c_zero, vt, queue ));
        solver_par->spmv_count++;
        if( precond_par->solver == Magma_NONE ){
            ;    
        } else {
            CHECK( magma_d_applyprecond_right( MagmaTrans, A, vt, &zt, precond_par, queue ));
            CHECK( magma_d_applyprecond_left( MagmaTrans, A, zt, &vt, precond_par, queue ));
        }

        magma_dscal( n, MAGMA_D_MAKE(-beta, 0.0 ), v.dval, 1, queue ); 
        magma_daxpy( n, c_one, vt.dval, 1, v.dval, 1, queue );
        alpha = magma_dnrm2( n, v.dval, 1, queue );
        magma_dscal( n, MAGMA_D_MAKE(1./alpha, 0.0 ), v.dval, 1, queue ); 
        normar = alpha * fabs(s*phi);
         
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_dresidualvec( A, b, *x, &r, &residual, queue));
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
    magma_dmfree(&r, queue );
    magma_dmfree(&v,  queue );
    magma_dmfree(&z,  queue );
    magma_dmfree(&zt, queue );
    magma_dmfree(&d,  queue );
    magma_dmfree(&vt,  queue );
    magma_dmfree(&q,  queue );
    magma_dmfree(&u,  queue );
    magma_dmfree(&w,  queue );
    magma_dmfree(&AT, queue );
    magma_dmfree(&Ah1, queue );
    magma_dmfree(&Ah2, queue );

    
    solver_par->info = info;
    return info;
}   /* magma_dqmr */
