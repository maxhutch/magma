/*
    -- MAGMA (version 1.6.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2015

       @author Hartwig Anzt

       @generated from zbaiter.cpp normal z -> d, Sun May  3 11:22:59 2015
*/

#include "common_magmasparse.h"

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
    magma_queue_t queue )
{
    magma_int_t info = 0;
        
    // prepare solver feedback
    solver_par->solver = Magma_BAITER;
    solver_par->info = MAGMA_SUCCESS;

    // initial residual
    real_Double_t tempo1, tempo2;
    double residual;
    magma_int_t localiter = 1;
    
    magma_d_matrix Ah={Magma_CSR}, ACSR={Magma_CSR}, A_d={Magma_CSR}, D={Magma_CSR}, 
                    R={Magma_CSR}, D_d={Magma_CSR}, R_d={Magma_CSR};
    
    CHECK( magma_dresidual( A, b, *x, &residual, queue ));
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;



    CHECK( magma_dmtransfer( A, &Ah, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_dmconvert( Ah, &ACSR, Ah.storage_type, Magma_CSR, queue ));

    CHECK( magma_dmtransfer( ACSR, &A_d, Magma_CPU, Magma_DEV, queue ));

    // setup
    CHECK( magma_dcsrsplit( 256, ACSR, &D, &R, queue ));
    CHECK( magma_dmtransfer( D, &D_d, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_dmtransfer( R, &R_d, Magma_CPU, Magma_DEV, queue ));


    tempo1 = magma_sync_wtime( queue );

    // block-asynchronous iteration iterator
    for( int iter=0; iter<solver_par->maxiter; iter++)
        CHECK( magma_dbajac_csr( localiter, D_d, R_d, b, x, queue ));

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    CHECK(  magma_dresidual( A, b, *x, &residual, queue));
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res ){
        info = MAGMA_SUCCESS;
    }
    else{
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
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

