/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/src/zpbicg.cpp normal z -> s, Mon May  2 23:30:56 2016
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
    where A is a general matrix.
    This is a GPU implementation of the preconditioned Biconjugate Gradient method.

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
                
    @param[in,out]
    precond_par magma_s_preconditioner*
                preconditioner
    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgesv
    ********************************************************************/

extern "C" magma_int_t
magma_spbicg(
    magma_s_matrix A, magma_s_matrix b, magma_s_matrix *x,
    magma_s_solver_par *solver_par,
    magma_s_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_PBICG;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;

    // some useful variables
    float c_zero = MAGMA_S_ZERO;
    float c_one  = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;
    
    magma_int_t dofs = A.num_rows * b.num_cols;

    // workspace
    magma_s_matrix r={Magma_CSR}, rt={Magma_CSR}, p={Magma_CSR}, pt={Magma_CSR}, 
                z={Magma_CSR}, zt={Magma_CSR}, q={Magma_CSR}, y={Magma_CSR}, 
                yt={Magma_CSR},  qt={Magma_CSR};
                
    // need to transpose the matrix
    magma_s_matrix AT={Magma_CSR}, Ah1={Magma_CSR}, Ah2={Magma_CSR};
    
    CHECK( magma_svinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &rt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &pt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &qt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &y, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &yt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_svinit( &zt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // solver variables
    float alpha, rho, beta, rho_new, ptq;
    float res, nomb, nom0, r0;

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
    
    // solver setup
    CHECK(  magma_sresidualvec( A, b, *x, &r, &nom0, queue));
    res = nom0;
    solver_par->init_res = nom0;
    magma_scopy( dofs, r.dval, 1, rt.dval, 1, queue );                  // rr = r
    rho_new = magma_sdot( dofs, rt.dval, 1, r.dval, 1, queue );             // rho=<rr,r>
    rho = alpha = MAGMA_S_MAKE( 1.0, 0. );

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
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom0 < r0 ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }

    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );


    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;

        CHECK( magma_s_applyprecond_left( MagmaNoTrans, A, r, &y, precond_par, queue ));
        CHECK( magma_s_applyprecond_right( MagmaNoTrans, A, y, &z, precond_par, queue ));
        CHECK( magma_s_applyprecond_right( MagmaTrans, A, rt, &yt, precond_par, queue ));
        CHECK( magma_s_applyprecond_left( MagmaTrans, A, yt, &zt, precond_par, queue ));
        //magma_scopy( dofs, r.dval, 1 , y.dval, 1, queue );             // y=r
        //magma_scopy( dofs, y.dval, 1 , z.dval, 1, queue );             // z=y
        //magma_scopy( dofs, rt.dval, 1 , yt.dval, 1, queue );           // yt=rt
        //magma_scopy( dofs, yt.dval, 1 , zt.dval, 1, queue );           // yt=rt
        
        rho= rho_new;
        rho_new = magma_sdot( dofs, rt.dval, 1, z.dval, 1, queue );  // rho=<rt,z>
        if( magma_s_isnan_inf( rho_new ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        if( solver_par->numiter==1 ){
            magma_scopy( dofs, z.dval, 1 , p.dval, 1, queue );           // yt=rt
            magma_scopy( dofs, zt.dval, 1 , pt.dval, 1, queue );           // zt=yt
        } else {
            beta = rho_new/rho;
            magma_sscal( dofs, beta, p.dval, 1, queue );                 // p = beta*p
            magma_saxpy( dofs, c_one , z.dval, 1 , p.dval, 1, queue );   // p = z+beta*p
            magma_sscal( dofs, MAGMA_S_CONJ(beta), pt.dval, 1, queue );   // pt = beta*pt
            magma_saxpy( dofs, c_one , zt.dval, 1 , pt.dval, 1, queue );  // pt = zt+beta*pt
        }
        CHECK( magma_s_spmv( c_one, A, p, c_zero, q, queue ));      // v = Ap
        CHECK( magma_s_spmv( c_one, AT, pt, c_zero, qt, queue ));   // v = Ap
        solver_par->spmv_count++;
        solver_par->spmv_count++;
        ptq = magma_sdot( dofs, pt.dval, 1, q.dval, 1, queue );
        alpha = rho_new /ptq;
        
        
        magma_saxpy( dofs, alpha, p.dval, 1 , x->dval, 1, queue );                // x=x+alpha*p
        magma_saxpy( dofs, c_neg_one * alpha, q.dval, 1 , r.dval, 1, queue );     // r=r+alpha*q
        magma_saxpy( dofs, c_neg_one * MAGMA_S_CONJ(alpha), qt.dval, 1 , rt.dval, 1, queue );     // r=r+alpha*q

        res = magma_snrm2( dofs, r.dval, 1, queue );

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
    CHECK(  magma_sresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
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
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
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
    magma_smfree(&rt, queue );
    magma_smfree(&p, queue );
    magma_smfree(&pt, queue );
    magma_smfree(&q, queue );
    magma_smfree(&qt, queue );
    magma_smfree(&y, queue );
    magma_smfree(&yt, queue );
    magma_smfree(&z, queue );
    magma_smfree(&zt, queue );
    magma_smfree(&AT, queue );
    magma_smfree(&Ah1, queue );
    magma_smfree(&Ah2, queue );

    solver_par->info = info;
    return info;
}   /* magma_spbicg */
