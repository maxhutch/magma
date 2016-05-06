/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Hartwig Anzt

       @generated from sparse-iter/src/zbaiter.cpp normal z -> d, Mon May  2 23:30:59 2016
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * x = b
    via the block asynchronous iteration method on GPU.

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
magma_dbaiter(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_solver_par *solver_par,
    magma_d_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
        
    // prepare solver feedback
    solver_par->solver = Magma_BAITER;
    
    // some useful variables 
    double c_zero = MAGMA_D_ZERO;

    // initial residual
    real_Double_t tempo1, tempo2, runtime=0;
    double residual;
    magma_int_t localiter = precond_par->maxiter;
    
    magma_d_matrix Ah={Magma_CSR}, ACSR={Magma_CSR}, A_d={Magma_CSR}, D={Magma_CSR}, 
                    R={Magma_CSR}, D_d={Magma_CSR}, R_d={Magma_CSR}, r={Magma_CSR};

    CHECK( magma_dmtransfer( A, &Ah, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_dmconvert( Ah, &ACSR, Ah.storage_type, Magma_CSR, queue ));

    CHECK( magma_dmtransfer( ACSR, &A_d, Magma_CPU, Magma_DEV, queue ));
    
    CHECK( magma_dvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK(  magma_dresidualvec( A_d, b, *x, &r, &residual, queue));
    solver_par->init_res = residual;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t) residual;
    }
    // setup
    CHECK( magma_dcsrsplit( 0, 256, ACSR, &D, &R, queue ));
    CHECK( magma_dmtransfer( D, &D_d, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_dmtransfer( R, &R_d, Magma_CPU, Magma_DEV, queue ));
    
    magma_int_t iterinc;
    if( solver_par->verbose == 0 ){
        iterinc = solver_par->maxiter;
    }
    else{
        iterinc = solver_par->verbose;
    }
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // block-asynchronous iteration iterator
    do
    {
        tempo1 = magma_sync_wtime( queue );
        solver_par->numiter+= iterinc;
        for( int z=0; z<iterinc; z++){
            CHECK( magma_dbajac_csr( localiter, D_d, R_d, b, x, queue ));
        }
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2-tempo1;
        if ( solver_par->verbose > 0 ) {
        CHECK(  magma_dresidualvec( A_d, b, *x, &r, &residual, queue));
            solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) residual;
            solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) runtime;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );

    solver_par->runtime = runtime;
    CHECK(  magma_dresidual( A_d, b, *x, &residual, queue));
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res ){
        info = MAGMA_SUCCESS;
    }
    else {
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_dmfree(&r, queue );
    magma_dmfree(&D, queue );
    magma_dmfree(&R, queue );
    magma_dmfree(&D_d, queue );
    magma_dmfree(&R_d, queue );
    magma_dmfree(&A_d, queue );
    magma_dmfree(&ACSR, queue );
    magma_dmfree(&Ah, queue );

    solver_par->info = info;
    return info;
}   /* magma_dbaiter */
