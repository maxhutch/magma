/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from run_zlaplacesolver.cpp normal z -> s, Fri May 30 10:41:49 2014
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
    precond_par.solver = Magma_JACOBI;

    magma_ssolverinfo_init( &solver_par, &precond_par );
    
    float one = MAGMA_S_MAKE(1.0, 0.0);
    float zero = MAGMA_S_MAKE(0.0, 0.0);

    magma_s_sparse_matrix A, B, B_d;
    magma_s_vector x, b;

    // generate matrix of desired structure and size
    magma_int_t n=10;   // size is n*n
    magma_int_t nn = n*n;
    magma_int_t offdiags = 2;
    magma_index_t *diag_offset;
    float *diag_vals;
    magma_smalloc_cpu( &diag_vals, offdiags+1 );
    magma_indexmalloc_cpu( &diag_offset, offdiags+1 );
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n;
    diag_vals[0] = MAGMA_S_MAKE( 4.0, 0.0 );
    diag_vals[1] = MAGMA_S_MAKE( -1.0, 0.0 );
    diag_vals[2] = MAGMA_S_MAKE( -1.0, 0.0 );
    magma_smgenerator( nn, offdiags, diag_offset, diag_vals, &A );

    // convert marix into desired format
    B.storage_type = Magma_SELLC;
    B.blocksize = 8;
    B.alignment = 8;
    magma_s_mconvert( A, &B, Magma_CSR, B.storage_type );
    magma_s_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );

    // vectors
    magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );

    // solver
    magma_spcg( B_d, b, &x, &solver_par, &precond_par );

    // solverinfo
    magma_ssolverinfo( &solver_par, &precond_par );

    magma_ssolverinfo_free( &solver_par, &precond_par );

    magma_s_mfree(&B_d);
    magma_s_mfree(&B);
    magma_s_mfree(&A); 
    magma_s_vfree(&x);
    magma_s_vfree(&b);

    TESTING_FINALIZE();
    return 0;
}
