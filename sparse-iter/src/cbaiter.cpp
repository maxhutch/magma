/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @author Hartwig Anzt 

       @generated from zbaiter.cpp normal z -> c, Fri Jul 18 17:34:29 2014
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

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

    @param
    A           magma_c_sparse_matrix
                input matrix A

    @param
    b           magma_c_vector
                RHS b

    @param
    x           magma_c_vector*
                solution approximation

    @param
    solver_par  magma_c_solver_par*
                solver parameters

    @ingroup magmasparse_cgesv
    ********************************************************************/

magma_int_t
magma_cbaiter( magma_c_sparse_matrix A, magma_c_vector b, magma_c_vector *x,  
           magma_c_solver_par *solver_par )
{
    // prepare solver feedback
    solver_par->solver = Magma_BAITER;
    solver_par->info = 0;

    magma_c_sparse_matrix A_d, D, R, D_d, R_d;
    magma_c_mtransfer( A, &A_d, Magma_CPU, Magma_DEV );

    // initial residual
    real_Double_t tempo1, tempo2;
    float residual;
    magma_cresidual( A_d, b, *x, &residual );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;

    // setup
    magma_ccsrsplit( 256, A, &D, &R );
    magma_c_mtransfer( D, &D_d, Magma_CPU, Magma_DEV );
    magma_c_mtransfer( R, &R_d, Magma_CPU, Magma_DEV );

    magma_int_t localiter = 1;

    magma_device_sync(); tempo1=magma_wtime();

    // block-asynchronous iteration iterator
    for( int iter=0; iter<solver_par->maxiter; iter++)
        magma_cbajac_csr( localiter, D_d, R_d, b, x );

    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_cresidual( A_d, b, *x, &residual );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if( solver_par->init_res > solver_par->final_res )
        solver_par->info = 0;
    else
        solver_par->info = -1;

    magma_c_mfree(&D);
    magma_c_mfree(&R);
    magma_c_mfree(&D_d);
    magma_c_mfree(&R_d);
    magma_c_mfree(&A_d);

    return MAGMA_SUCCESS;
}   /* magma_cbaiter */

