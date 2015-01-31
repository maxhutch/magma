/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Hartwig Anzt 

       @generated from zbaiter.cpp normal z -> s, Fri Jan 30 19:00:30 2015
*/

#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * x = b
    via the block asynchronous iteration method on GPU.

    Arguments
    ---------

    @param[in]
    A           magma_s_sparse_matrix
                input matrix A

    @param[in]
    b           magma_s_vector
                RHS b

    @param[in,out]
    x           magma_s_vector*
                solution approximation

    @param[in,out]
    solver_par  magma_s_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_sgesv
    ********************************************************************/

extern "C" magma_int_t
magma_sbaiter(
    magma_s_sparse_matrix A, 
    magma_s_vector b, 
    magma_s_vector *x,  
    magma_s_solver_par *solver_par,
    magma_queue_t queue )
{
    // prepare solver feedback
    solver_par->solver = Magma_BAITER;
    solver_par->info = MAGMA_SUCCESS;



    magma_s_sparse_matrix Ah, ACSR, A_d, D, R, D_d, R_d;

    magma_s_mtransfer( A, &Ah, A.memory_location, Magma_CPU, queue );
    magma_s_mconvert( Ah, &ACSR, Ah.storage_type, Magma_CSR, queue );

    magma_s_mtransfer( ACSR, &A_d, Magma_CPU, Magma_DEV, queue );

    // initial residual
    real_Double_t tempo1, tempo2;
    float residual;
    magma_sresidual( A_d, b, *x, &residual, queue );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;


    // setup
    magma_scsrsplit( 256, ACSR, &D, &R, queue );
    magma_s_mtransfer( D, &D_d, Magma_CPU, Magma_DEV, queue );
    magma_s_mtransfer( R, &R_d, Magma_CPU, Magma_DEV, queue );

    magma_int_t localiter = 1;

    tempo1 = magma_sync_wtime( queue );

    // block-asynchronous iteration iterator
    for( int iter=0; iter<solver_par->maxiter; iter++)
        magma_sbajac_csr( localiter, D_d, R_d, b, x, queue );

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_sresidual( A_d, b, *x, &residual, queue );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res )
        solver_par->info = MAGMA_SUCCESS;
    else
        solver_par->info = MAGMA_DIVERGENCE;

    magma_s_mfree(&D, queue );
    magma_s_mfree(&R, queue );
    magma_s_mfree(&D_d, queue );
    magma_s_mfree(&R_d, queue );
    magma_s_mfree(&A_d, queue );
    magma_s_mfree(&ACSR, queue );
    magma_s_mfree(&Ah, queue );

    return MAGMA_SUCCESS;
}   /* magma_sbaiter */

