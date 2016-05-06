/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zpcg_merge.cpp normal z -> c, Mon May  2 23:30:55 2016
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
    This is a GPU implementation of the Conjugate Gradient method in variant,
    where multiple operations are merged into one compute kernel.

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
                preconditioner
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cposv
    ********************************************************************/

extern "C" magma_int_t
magma_cpcg_merge(
    magma_c_matrix A, magma_c_matrix b, magma_c_matrix *x,
    magma_c_solver_par *solver_par,
    magma_c_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_PCGMERGE;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    // solver variables
    magmaFloatComplex alpha, beta, gamma, rho, tmp1, *skp_h={0};
    float nom, nom0, r0,  res, nomb;
    magmaFloatComplex den;

    // some useful variables
    magmaFloatComplex c_zero = MAGMA_C_ZERO, c_one = MAGMA_C_ONE;
    magma_int_t dofs = A.num_rows*b.num_cols;

    magma_c_matrix r={Magma_CSR}, d={Magma_CSR}, z={Magma_CSR}, h={Magma_CSR},
                    rt={Magma_CSR};
    magmaFloatComplex *d1=NULL, *d2=NULL, *skp=NULL;

    // GPU workspace
    CHECK( magma_cvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &rt, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_cvinit( &h, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    
    CHECK( magma_cmalloc( &d1, dofs*(2) ));
    CHECK( magma_cmalloc( &d2, dofs*(2) ));
    // array for the parameters
    CHECK( magma_cmalloc( &skp, 7 ));
    // skp = [alpha|beta|gamma|rho|tmp1|tmp2|res]

    // solver setup
    CHECK(  magma_cresidualvec( A, b, *x, &r, &nom0, queue));
    
    // preconditioner
    CHECK( magma_c_applyprecond_left( MagmaNoTrans, A, r, &rt, precond_par, queue ));
    CHECK( magma_c_applyprecond_right( MagmaNoTrans, A, rt, &h, precond_par, queue ));
    
    magma_ccopy( dofs, h.dval, 1, d.dval, 1, queue );  
    nom = MAGMA_C_ABS( magma_cdotc( dofs, r.dval, 1, h.dval, 1, queue ));
    CHECK( magma_c_spmv( c_one, A, d, c_zero, z, queue ));              // z = A d
    den = magma_cdotc( dofs, d.dval, 1, z.dval, 1, queue ); // den = d'* z
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
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom < r0 ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }
    // check positive definite
    if ( MAGMA_C_ABS(den) <= 0.0 ) {
        info = MAGMA_NONSPD;
        goto cleanup;
    }    
    
    // array on host for the parameters
    CHECK( magma_cmalloc_cpu( &skp_h, 7 ));
    
    alpha = rho = gamma = tmp1 = c_one;
    beta =  magma_cdotc( dofs, h.dval, 1, r.dval, 1, queue );
    skp_h[0]=alpha;
    skp_h[1]=beta;
    skp_h[2]=gamma;
    skp_h[3]=rho;
    skp_h[4]=tmp1;
    skp_h[5]=MAGMA_C_MAKE(nom, 0.0);
    skp_h[6]=MAGMA_C_MAKE(nom, 0.0);

    magma_csetvector( 7, skp_h, 1, skp, 1, queue );

    //Chronometry
    real_Double_t tempo1, tempo2, tempop1, tempop2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        
        // computes SpMV and dot product
        CHECK( magma_ccgmerge_spmv1(  A, d1, d2, d.dval, z.dval, skp, queue ));            
        solver_par->spmv_count++;
            
        
        if( precond_par->solver == Magma_JACOBI ){
                CHECK( magma_cjcgmerge_xrbeta( dofs, d1, d2, precond_par->d.dval, x->dval, r.dval, d.dval, z.dval, h.dval, skp, queue ));
        }
        else if( precond_par->solver == Magma_NONE ){
            // updates x, r
            CHECK( magma_cpcgmerge_xrbeta1( dofs, x->dval, r.dval, d.dval, z.dval, skp, queue ));
            // computes scalars and updates d
            CHECK( magma_cpcgmerge_xrbeta2( dofs, d1, d2, r.dval, r.dval, d.dval, skp, queue ));
        } else {
        
            // updates x, r
            CHECK( magma_cpcgmerge_xrbeta1( dofs, x->dval, r.dval, d.dval, z.dval, skp, queue ));
            
            // preconditioner in between
            tempop1 = magma_sync_wtime( queue );
            CHECK( magma_c_applyprecond_left( MagmaNoTrans, A, r, &rt, precond_par, queue ));
            CHECK( magma_c_applyprecond_right( MagmaNoTrans, A, rt, &h, precond_par, queue ));
            //            magma_ccopy( dofs, r.dval, 1, h.dval, 1 );  
            tempop2 = magma_sync_wtime( queue );
            precond_par->runtime += tempop2-tempop1;
            
            // computes scalars and updates d
            CHECK( magma_cpcgmerge_xrbeta2( dofs, d1, d2, h.dval, r.dval, d.dval, skp, queue ));
        }
        
        //if( solver_par->numiter==1){
        //    magma_ccopy( dofs, h.dval, 1, d.dval, 1 );   
        //}
        
                // updates x, r, computes scalars and updates d
        //CHECK( magma_ccgmerge_xrbeta( dofs, d1, d2, x->dval, r.dval, d.dval, z.dval, skp, queue ));

        
        // check stopping criterion (asynchronous copy)
        magma_cgetvector( 1 , skp+6, 1, skp_h+6, 1, queue );
        res = sqrt(MAGMA_C_ABS(skp_h[6]));

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
        if( solver_par->iter_res < solver_par->atol ||
            solver_par->iter_res/solver_par->init_res < solver_par->rtol ){
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
        solver_par->info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_cmfree(&r, queue );
    magma_cmfree(&z, queue );
    magma_cmfree(&d, queue );
    magma_cmfree(&rt, queue );
    magma_cmfree(&h, queue );

    magma_free( d1 );
    magma_free( d2 );
    magma_free( skp );
    magma_free_cpu( skp_h );

    solver_par->info = info;
    return info;
}   /* magma_cpcg_merge */
