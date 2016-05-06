/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from sparse-iter/src/zbicgstab_merge2.cpp normal z -> c, Mon May  2 23:30:56 2016
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )

#define  q(i)     (q.dval + (i)*dofs)

/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a general matrix.
    This is a GPU implementation of the Biconjugate Gradient Stabilized method.
    The difference to magma_cbicgstab is that we use specifically designed kernels
    merging multiple operations into one kernel.

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
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgesv
    ********************************************************************/

extern "C" magma_int_t
magma_cbicgstab_merge2(
    magma_c_matrix A, magma_c_matrix b,
    magma_c_matrix *x, magma_c_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_BICGSTABMERGE2;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    // solver variables
    magmaFloatComplex alpha, beta, omega, rho_old, rho_new, *skp_h={0};
    float nom, nom0, betanom, nomb;
    //float den;

    // some useful variables
    magmaFloatComplex c_zero = MAGMA_C_ZERO, c_one = MAGMA_C_ONE;
    
    magma_int_t dofs = A.num_rows;


    // workspace
    magma_c_matrix q={Magma_CSR}, r={Magma_CSR}, rr={Magma_CSR}, p={Magma_CSR}, v={Magma_CSR}, s={Magma_CSR}, t={Magma_CSR};
    magmaFloatComplex *d1=NULL, *d2=NULL, *skp=NULL;
    d1 = NULL;
    d2 = NULL;
    skp = NULL;
    
    CHECK( magma_cmalloc( &d1, dofs*(2) ));
    CHECK( magma_cmalloc( &d2, dofs*(2) ));

    // array for the parameters
    CHECK( magma_cmalloc( &skp, 8 ));
    // skp = [alpha|beta|omega|rho_old|rho|nom|tmp1|tmp2]
    CHECK( magma_cvinit( &q, Magma_DEV, dofs*6, 1, c_zero, queue ));

    // q = rr|r|p|v|s|t
    rr.memory_location = Magma_DEV; rr.dval = NULL; rr.num_rows = rr.nnz = dofs;
    r.memory_location = Magma_DEV; r.dval = NULL; r.num_rows = r.nnz = dofs;
    p.memory_location = Magma_DEV; p.dval = NULL; p.num_rows = p.nnz = dofs;
    v.memory_location = Magma_DEV; v.dval = NULL; v.num_rows = v.nnz = dofs;
    s.memory_location = Magma_DEV; s.dval = NULL; s.num_rows = s.nnz = dofs;
    t.memory_location = Magma_DEV; t.dval = NULL; t.num_rows = t.nnz = dofs;

    rr.dval = q(0);
    r.dval = q(1);
    p.dval = q(2);
    v.dval = q(3);
    s.dval = q(4);
    t.dval = q(5);

    // solver setup
    magma_cscal( dofs, c_zero, x->dval, 1, queue );                             // x = 0
    CHECK(  magma_cresidualvec( A, b, *x, &r, &nom0, queue));
    magma_ccopy( dofs, r.dval, 1, q(0), 1, queue );                            // rr = r
    magma_ccopy( dofs, r.dval, 1, q(1), 1, queue );                            // q = r
    betanom = nom0;
    nom = nom0*nom0;
    rho_new = magma_cdotc( dofs, r.dval, 1, r.dval, 1, queue );             // rho=<rr,r>
    rho_old = omega = alpha = MAGMA_C_MAKE( 1.0, 0. );
    beta = rho_new;
    solver_par->init_res = nom0;
    // array on host for the parameters
    CHECK( magma_cmalloc_cpu( &skp_h, 8 ));
    
    skp_h[0]=alpha;
    skp_h[1]=beta;
    skp_h[2]=omega;
    skp_h[3]=rho_old;
    skp_h[4]=rho_new;
    skp_h[5]=MAGMA_C_MAKE(nom, 0.0);
    magma_csetvector( 8, skp_h, 1, skp, 1, queue );

    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }
    nomb = magma_scnrm2( dofs, b.dval, 1, queue );
    if( nom0 < solver_par->atol ||
        nom0/nomb < solver_par->rtol ){
        info = MAGMA_SUCCESS;
        goto cleanup;
    }
    
    CHECK( magma_c_spmv( c_one, A, r, c_zero, v, queue ));             // z = A r

    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );


    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;

        // computes p=r+beta*(p-omega*v)
        CHECK( magma_cbicgmerge1( dofs, skp, v.dval, r.dval, p.dval, queue ));
        CHECK( magma_cbicgmerge_spmv1(  A, d1, d2, q(2), q(0), q(3), skp, queue ));
        solver_par->spmv_count++;
        CHECK( magma_cbicgmerge2( dofs, skp, r.dval, v.dval, s.dval, queue )); // s=r-alpha*v
        CHECK( magma_cbicgmerge_spmv2( A, d1, d2, q(4), q(5), skp, queue ));
        solver_par->spmv_count++;
        CHECK( magma_cbicgmerge_xrbeta( dofs, d1, d2, q(0), q(1), q(2),
                                                    q(4), q(5), x->dval, skp, queue ));

        // check stopping criterion (asynchronous copy)
        magma_cgetvector( 1 , skp+5, 1, skp_h+5, 1, queue );

        betanom = sqrt(MAGMA_C_REAL(skp_h[5]));

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if (  betanom  < solver_par->atol || 
              betanom/nomb < solver_par->rtol ) {
            break;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    float residual;
    CHECK( magma_cresidual( A, b, *x, &residual, NULL ));
    solver_par->iter_res = betanom;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
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
        if( solver_par->iter_res < solver_par->atol ||
            solver_par->iter_res/solver_par->init_res < solver_par->rtol ){
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
    magma_cmfree(&q, queue );  // frees all vectors
    magma_free(d1);
    magma_free(d2);
    magma_free( skp );
    magma_free_cpu( skp_h );

    solver_par->info = info;
    return info;
}   /* cbicgstab_merge2 */
