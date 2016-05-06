/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zpcg_merge.cpp normal z -> d, Mon May  2 23:30:55 2016
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
    This is a GPU implementation of the Conjugate Gradient method in variant,
    where multiple operations are merged into one compute kernel.

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
                preconditioner
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dposv
    ********************************************************************/

extern "C" magma_int_t
magma_dpcg_merge(
    magma_d_matrix A, magma_d_matrix b, magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_PCGMERGE;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    // solver variables
    double alpha, beta, gamma, rho, tmp1, *skp_h={0};
    double nom, nom0, r0,  res, nomb;
    double den;

    // some useful variables
    double c_zero = MAGMA_D_ZERO, c_one = MAGMA_D_ONE;
    magma_int_t dofs = A.num_rows*b.num_cols;

    magma_d_matrix r={Magma_CSR}, d={Magma_CSR}, z={Magma_CSR}, h={Magma_CSR},
                    rt={Magma_CSR};
    double *d1=NULL, *d2=NULL, *skp=NULL;

    // GPU workspace
    CHECK( magma_dvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &rt, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_dvinit( &h, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    
    CHECK( magma_dmalloc( &d1, dofs*(2) ));
    CHECK( magma_dmalloc( &d2, dofs*(2) ));
    // array for the parameters
    CHECK( magma_dmalloc( &skp, 7 ));
    // skp = [alpha|beta|gamma|rho|tmp1|tmp2|res]

    // solver setup
    CHECK(  magma_dresidualvec( A, b, *x, &r, &nom0, queue));
    
    // preconditioner
    CHECK( magma_d_applyprecond_left( MagmaNoTrans, A, r, &rt, precond_par, queue ));
    CHECK( magma_d_applyprecond_right( MagmaNoTrans, A, rt, &h, precond_par, queue ));
    
    magma_dcopy( dofs, h.dval, 1, d.dval, 1, queue );  
    nom = MAGMA_D_ABS( magma_ddot( dofs, r.dval, 1, h.dval, 1, queue ));
    CHECK( magma_d_spmv( c_one, A, d, c_zero, z, queue ));              // z = A d
    den = magma_ddot( dofs, d.dval, 1, z.dval, 1, queue ); // den = d'* z
    solver_par->init_res = nom0;
    
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
    if ( nom < r0 ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }
    // check positive definite
    if ( MAGMA_D_ABS(den) <= 0.0 ) {
        info = MAGMA_NONSPD;
        goto cleanup;
    }    
    
    // array on host for the parameters
    CHECK( magma_dmalloc_cpu( &skp_h, 7 ));
    
    alpha = rho = gamma = tmp1 = c_one;
    beta =  magma_ddot( dofs, h.dval, 1, r.dval, 1, queue );
    skp_h[0]=alpha;
    skp_h[1]=beta;
    skp_h[2]=gamma;
    skp_h[3]=rho;
    skp_h[4]=tmp1;
    skp_h[5]=MAGMA_D_MAKE(nom, 0.0);
    skp_h[6]=MAGMA_D_MAKE(nom, 0.0);

    magma_dsetvector( 7, skp_h, 1, skp, 1, queue );

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
        CHECK( magma_dcgmerge_spmv1(  A, d1, d2, d.dval, z.dval, skp, queue ));            
        solver_par->spmv_count++;
            
        
        if( precond_par->solver == Magma_JACOBI ){
                CHECK( magma_djcgmerge_xrbeta( dofs, d1, d2, precond_par->d.dval, x->dval, r.dval, d.dval, z.dval, h.dval, skp, queue ));
        }
        else if( precond_par->solver == Magma_NONE ){
            // updates x, r
            CHECK( magma_dpcgmerge_xrbeta1( dofs, x->dval, r.dval, d.dval, z.dval, skp, queue ));
            // computes scalars and updates d
            CHECK( magma_dpcgmerge_xrbeta2( dofs, d1, d2, r.dval, r.dval, d.dval, skp, queue ));
        } else {
        
            // updates x, r
            CHECK( magma_dpcgmerge_xrbeta1( dofs, x->dval, r.dval, d.dval, z.dval, skp, queue ));
            
            // preconditioner in between
            tempop1 = magma_sync_wtime( queue );
            CHECK( magma_d_applyprecond_left( MagmaNoTrans, A, r, &rt, precond_par, queue ));
            CHECK( magma_d_applyprecond_right( MagmaNoTrans, A, rt, &h, precond_par, queue ));
            //            magma_dcopy( dofs, r.dval, 1, h.dval, 1 );  
            tempop2 = magma_sync_wtime( queue );
            precond_par->runtime += tempop2-tempop1;
            
            // computes scalars and updates d
            CHECK( magma_dpcgmerge_xrbeta2( dofs, d1, d2, h.dval, r.dval, d.dval, skp, queue ));
        }
        
        //if( solver_par->numiter==1){
        //    magma_dcopy( dofs, h.dval, 1, d.dval, 1 );   
        //}
        
                // updates x, r, computes scalars and updates d
        //CHECK( magma_dcgmerge_xrbeta( dofs, d1, d2, x->dval, r.dval, d.dval, z.dval, skp, queue ));

        
        // check stopping criterion (asynchronous copy)
        magma_dgetvector( 1 , skp+6, 1, skp_h+6, 1, queue );
        res = sqrt(MAGMA_D_ABS(skp_h[6]));

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
    magma_dmfree(&r, queue );
    magma_dmfree(&z, queue );
    magma_dmfree(&d, queue );
    magma_dmfree(&rt, queue );
    magma_dmfree(&h, queue );

    magma_free( d1 );
    magma_free( d2 );
    magma_free( skp );
    magma_free_cpu( skp_h );

    solver_par->info = info;
    return info;
}   /* magma_dpcg_merge */
