/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @author Hartwig Anzt 

       @generated from zbaiter.cpp normal z -> s, Fri May 30 10:41:41 2014
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    Solves a system of linear equations
       A * x = b
    via the block asynchronous iteration method on GPU.

    Arguments
    =========

    magma_s_sparse_matrix A                   input matrix A
    magma_s_vector b                          RHS b
    magma_s_vector *x                         solution approximation
    magma_s_solver_par *solver_par       solver parameters

    ========================================================================  */


magma_int_t
magma_sbaiter( magma_s_sparse_matrix A, magma_s_vector b, magma_s_vector *x,  
           magma_s_solver_par *solver_par )
{
    // prepare solver feedback
    solver_par->solver = Magma_BAITER;
    solver_par->info = 0;

    magma_s_sparse_matrix A_d, D, R, D_d, R_d;
    magma_s_mtransfer( A, &A_d, Magma_CPU, Magma_DEV );

    // initial residual
    real_Double_t tempo1, tempo2;
    float residual;
    magma_sresidual( A_d, b, *x, &residual );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;

    // setup
    magma_scsrsplit( 256, A, &D, &R );
    magma_s_mtransfer( D, &D_d, Magma_CPU, Magma_DEV );
    magma_s_mtransfer( R, &R_d, Magma_CPU, Magma_DEV );

    magma_int_t localiter = 1;

    magma_device_sync(); tempo1=magma_wtime();

    // block-asynchronous iteration iterator
    for( int iter=0; iter<solver_par->maxiter; iter++)
        magma_sbajac_csr( localiter, D_d, R_d, b, x );

    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_sresidual( A_d, b, *x, &residual );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if( solver_par->init_res > solver_par->final_res )
        solver_par->info = 0;
    else
        solver_par->info = -1;

    magma_s_mfree(&D);
    magma_s_mfree(&R);
    magma_s_mfree(&D_d);
    magma_s_mfree(&R_d);
    magma_s_mfree(&A_d);

    return MAGMA_SUCCESS;
}   /* magma_sbaiter */

