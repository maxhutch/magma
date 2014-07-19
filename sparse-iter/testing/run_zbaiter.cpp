/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @precisions normal z -> c d s
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
   -- running magma_zbaiter
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_z_solver_par solver_par;
    magma_z_preconditioner precond_par;
    solver_par.maxiter = 1000;
    solver_par.verbose = 0;
    solver_par.num_eigenvalues = 0;
    int scale = 0;
    magma_scale_t scaling = Magma_NOSCALE;
    
    magma_z_sparse_matrix A;
    magma_z_vector x, b;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

    int i;
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("--maxiter", argv[i]) == 0 ){
            solver_par.maxiter = atoi( argv[++i] );
        }else if ( strcmp("--mscale", argv[i]) == 0 ) {
            scale = atoi( argv[++i] );
            switch( scale ) {
                case 0: scaling = Magma_NOSCALE; break;
                case 1: scaling = Magma_UNITDIAG; break;
                case 2: scaling = Magma_UNITROW; break;
            }

        }else
            break;
    }
    printf( "\n#    usage: ./run_zbaiter"
        " [ "
        " --mscale %d (0=no, 1=unitdiag, 2=unitrownrm)"
        " --maxiter %d ]"
        " matrices \n\n",
        (int) scale,
        (int) solver_par.maxiter);

    magma_zsolverinfo_init( &solver_par, &precond_par );

    while(  i < argc ){

        magma_z_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale initial guess
        magma_zmscale( &A, scaling );

        magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
        magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

        magma_zbaiter( A, b, &x, &solver_par );

        magma_zsolverinfo( &solver_par, &precond_par );


        magma_z_mfree(&A); 
        magma_z_vfree(&x);
        magma_z_vfree(&b);
        
        i++;
    }

    magma_zsolverinfo_free( &solver_par, &precond_par );

    TESTING_FINALIZE();

    return 0;
}
