/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from run_zalltests.cpp normal z -> s, Fri Jul 18 17:34:31 2014
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "../include/magmasparse.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Debugging file
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_s_solver_par solver_par;
    magma_s_preconditioner precond_par;
    solver_par.epsilon = 10e-16;
    solver_par.maxiter = 1000;
    solver_par.verbose = 0;
    solver_par.restart = 30;
    solver_par.num_eigenvalues = 0;
    solver_par.ortho = Magma_CGS;
    
    float one = MAGMA_S_MAKE(1.0, 0.0);
    float zero = MAGMA_S_MAKE(0.0, 0.0);

    magma_s_sparse_matrix A, B, B_d;
    magma_s_vector x, b;

    // generate matrix of desired structure and size
    magma_int_t n=100;   // size is n*n
    magma_int_t nn = n*n;
    magma_int_t offdiags = 2;
    magma_index_t *diag_offset;
    float *diag_vals;
    magma_smalloc_cpu( &diag_vals, offdiags+1 );
    magma_index_malloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n;
    diag_vals[0] = MAGMA_S_MAKE( 4.1, 0.0 );
    diag_vals[1] = MAGMA_S_MAKE( -1.0, 0.0 );
    diag_vals[2] = MAGMA_S_MAKE( -1.0, 0.0 );
    magma_smgenerator( nn, offdiags, diag_offset, diag_vals, &A );

    // convert marix into desired format
    B.storage_type = Magma_SELLC;
    B.blocksize = 8;
    B.alignment = 8;
    // scale matrix
    magma_smscale( &A, Magma_UNITDIAG );

    magma_s_mconvert( A, &B, Magma_CSR, B.storage_type );
    magma_s_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );


    // test CG ####################################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // solver
    magma_scg_res( B_d, b, &x, &solver_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    if( solver_par.numiter > 150 ){
        printf("error: test not passed!\n"); exit(-1);
    }
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);

    // test PCG Jacobi ############################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // Preconditioner
    precond_par.solver = Magma_JACOBI;
    magma_s_precondsetup( B_d, b, &precond_par );
    // solver
    magma_spcg( B_d, b, &x, &solver_par, &precond_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    if( solver_par.numiter > 150 ){
        printf("error: test not passed!\n"); exit(-1);
    }
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);

    // test PCG IC ################################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // Preconditioner
    precond_par.solver = Magma_ICC;
    magma_s_precondsetup( B_d, b, &precond_par );
    // solver
    magma_spcg( B_d, b, &x, &solver_par, &precond_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    if( solver_par.numiter > 150 ){
        printf("error: test not passed!\n"); exit(-1);
    }
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);


    // test PCG IC ################################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // Preconditioner
    precond_par.solver = Magma_ICC;
    magma_s_precondsetup( B_d, b, &precond_par );
    // solver
    magma_spcg( B_d, b, &x, &solver_par, &precond_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    if( solver_par.numiter > 150 ){
        printf("error: test not passed!\n"); exit(-1);
    }
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);

    // test BICGSTAB ####################################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // solver
    magma_sbicgstab( B_d, b, &x, &solver_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    if( solver_par.numiter > 150 ){
        printf("error: test not passed!\n"); exit(-1);
    }
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);

    // test PBICGSTAB Jacobi ############################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // Preconditioner
    precond_par.solver = Magma_JACOBI;
    magma_s_precondsetup( B_d, b, &precond_par );
    // solver
    magma_spbicgstab( B_d, b, &x, &solver_par, &precond_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    if( solver_par.numiter > 150 ){
        printf("error: test not passed!\n"); exit(-1);
    }
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);
/*
    // test PBICGSTAB ILU ###############################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // Preconditioner
    precond_par.solver = Magma_ILU;
    magma_s_precondsetup( B_d, b, &precond_par );
    // solver
    magma_spbicgstab( B_d, b, &x, &solver_par, &precond_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    if( solver_par.numiter > 150 ){
        printf("error: test not passed!\n"); exit(-1);
    }
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);

    // test PBICGSTAB ILU ###############################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);printf("here\n");
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // Preconditioner
    precond_par.solver = Magma_ILU;
    magma_s_precondsetup( B_d, b, &precond_par );
    // solver
    magma_spbicgstab( B_d, b, &x, &solver_par, &precond_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    if( solver_par.numiter > 150 ){
        printf("error: test not passed!\n"); exit(-1);
    }
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);

    // test GMRES ####################################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // solver
    magma_sgmres( B_d, b, &x, &solver_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);

    // test PGMRES Jacobi ############################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // Preconditioner
    precond_par.solver = Magma_JACOBI;
    magma_s_precondsetup( B_d, b, &precond_par );
    // solver
    magma_spgmres( B_d, b, &x, &solver_par, &precond_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);*/

    // test PGMRES ILU ###############################
    // vectors and initial guess
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
    magma_s_vfree(&x);
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );
    magma_ssolverinfo_init( &solver_par, &precond_par );
    // Preconditioner
    precond_par.solver = Magma_ILU;
    magma_s_precondsetup( B_d, b, &precond_par );
    // solver
    magma_spgmres( B_d, b, &x, &solver_par, &precond_par );
    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );
    if( solver_par.numiter > 150 ){
        printf("error: test not passed!\n"); exit(-1);
    }
    magma_ssolverinfo_free( &solver_par, &precond_par );
    magma_s_vfree(&x);
    magma_s_vfree(&b);


    printf("all tests passed.\n");




    magma_s_mfree(&B_d);
    magma_s_mfree(&B);
    magma_s_mfree(&A); 


    TESTING_FINALIZE();
    return 0;
}
