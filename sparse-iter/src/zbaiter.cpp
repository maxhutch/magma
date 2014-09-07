/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "magmasparse.h"

#include <assert.h>

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

    @param
    A           magma_z_sparse_matrix
                input matrix A

    @param
    b           magma_z_vector
                RHS b

    @param
    x           magma_z_vector*
                solution approximation

    @param
    solver_par  magma_z_solver_par*
                solver parameters

    @ingroup magmasparse_zgesv
    ********************************************************************/

magma_int_t
magma_zbaiter( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_z_solver_par *solver_par )
{
    // prepare solver feedback
    solver_par->solver = Magma_BAITER;
    solver_par->info = 0;

    magma_z_sparse_matrix A_d, D, R, D_d, R_d;
    magma_z_mtransfer( A, &A_d, Magma_CPU, Magma_DEV );

    // initial residual
    real_Double_t tempo1, tempo2;
    double residual;
    magma_zresidual( A_d, b, *x, &residual );
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;

    // setup
    magma_zcsrsplit( 256, A, &D, &R );
    magma_z_mtransfer( D, &D_d, Magma_CPU, Magma_DEV );
    magma_z_mtransfer( R, &R_d, Magma_CPU, Magma_DEV );

    magma_int_t localiter = 1;

    magma_device_sync(); tempo1=magma_wtime();

    // block-asynchronous iteration iterator
    for( int iter=0; iter<solver_par->maxiter; iter++)
        magma_zbajac_csr( localiter, D_d, R_d, b, x );

    magma_device_sync(); tempo2=magma_wtime();
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    magma_zresidual( A_d, b, *x, &residual );
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if( solver_par->init_res > solver_par->final_res )
        solver_par->info = 0;
    else
        solver_par->info = -1;

    magma_z_mfree(&D);
    magma_z_mfree(&R);
    magma_z_mfree(&D_d);
    magma_z_mfree(&R_d);
    magma_z_mfree(&A_d);

    return MAGMA_SUCCESS;
}   /* magma_zbaiter */

