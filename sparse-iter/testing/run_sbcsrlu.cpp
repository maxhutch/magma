/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from run_zbcsrlu.cpp normal z -> s, Fri May 30 10:41:49 2014
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "../include/magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- running magma_sbcsrlu
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_s_solver_par solver_par;
    magma_s_preconditioner precond_par;
    solver_par.maxiter = 1000;
    solver_par.verbose = 0;
    solver_par.version = 0;
    solver_par.num_eigenvalues = 0;
    int scale = 0;
    magma_scale_t scaling = Magma_NOSCALE;

    magma_s_sparse_matrix A, B;
    magma_s_vector x, b;
    
    float one = MAGMA_S_MAKE(1.0, 0.0);
    float zero = MAGMA_S_MAKE(0.0, 0.0);

    B.storage_type = Magma_CSR;
    int i;
    for( i = 1; i < argc; ++i ) {
      if ( strcmp("--version", argv[i]) == 0 ) {
            solver_par.version = atoi( argv[++i] );
        }else if ( strcmp("--mscale", argv[i]) == 0 ) {
            scale = atoi( argv[++i] );
            switch( scale ) {
                case 0: scaling = Magma_NOSCALE; break;
                case 1: scaling = Magma_UNITDIAG; break;
                case 2: scaling = Magma_UNITROW; break;
            }

        }
      else
        break;
    }
    printf( "\n#    usage: ./run_sbcsrlu"
            " [ --version %d (0=CUBLAS batched, 1=custom kernels) "
            " --mscale %d (0=no, 1=unitdiag, 2=unitrownrm) ]"
            " matrices \n\n", (int) solver_par.version, (int) scale );

    magma_ssolverinfo_init( &solver_par, &precond_par );

    while(  i < argc ){

        magma_s_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale initial guess
        magma_smscale( &A, scaling );

        magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
        magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );

        magma_sbcsrlu( A, b, &x, &solver_par );

        magma_ssolverinfo( &solver_par, &precond_par );

        magma_s_mfree(&A); 
        magma_s_vfree(&x);
        magma_s_vfree(&b);

        i++;
    }

    magma_ssolverinfo_free( &solver_par, &precond_par );

    TESTING_FINALIZE();
    return 0;
}
