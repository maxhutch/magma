/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @generated from run_zcg.cpp normal z -> d, Fri Jul 18 17:34:31 2014
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
   -- running magma_dcg magma_dcg_merge 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_d_solver_par solver_par;
    magma_d_preconditioner precond_par;
    solver_par.epsilon = 10e-16;
    solver_par.maxiter = 1000;
    solver_par.verbose = 0;
    solver_par.num_eigenvalues = 0;
    int format = 0;
    int version = 0;

    magma_d_sparse_matrix A, B, B_d;
    magma_d_vector x, b;
    B.blocksize = 8;
    B.alignment = 8;
    int scale = 0;
    magma_scale_t scaling = Magma_NOSCALE;
    
    double one = MAGMA_D_MAKE(1.0, 0.0);
    double zero = MAGMA_D_MAKE(0.0, 0.0);

    B.storage_type = Magma_CSR;
    int i;
    for( i = 1; i < argc; ++i ) {
     if ( strcmp("--format", argv[i]) == 0 ) {
            format = atoi( argv[++i] );
            switch( format ) {
                case 0: B.storage_type = Magma_CSR; break;
                case 1: B.storage_type = Magma_ELL; break;
                case 2: B.storage_type = Magma_ELLRT; break;
                case 3: B.storage_type = Magma_SELLP; break;
            }
        }else if ( strcmp("--mscale", argv[i]) == 0 ) {
            scale = atoi( argv[++i] );
            switch( scale ) {
                case 0: scaling = Magma_NOSCALE; break;
                case 1: scaling = Magma_UNITDIAG; break;
                case 2: scaling = Magma_UNITROW; break;
            }

        } else if ( strcmp("--version", argv[i]) == 0 ) {
            version = atoi( argv[++i] );
        }else if ( strcmp("--blocksize", argv[i]) == 0 ) {
            B.blocksize = atoi( argv[++i] );
        }else if ( strcmp("--alignment", argv[i]) == 0 ) {
            B.alignment = atoi( argv[++i] );
        }else if ( strcmp("--verbose", argv[i]) == 0 ) {
            solver_par.verbose = atoi( argv[++i] );
        }  else if ( strcmp("--maxiter", argv[i]) == 0 ) {
            solver_par.maxiter = atoi( argv[++i] );
        } else if ( strcmp("--tol", argv[i]) == 0 ) {
            sscanf( argv[++i], "%lf", &solver_par.epsilon );
        } else
            break;
    }
    printf( "\n#    usage: ./run_dcg"
        " [ --format %d (0=CSR, 1=ELL 2=ELLRT, 3=SELLP)"
        " [ --blocksize %d --alignment %d ]"
        " --mscale %d (0=no, 1=unitdiag, 2=unitrownrm)"
        " --verbose %d (0=summary, k=details every k iterations)"
        " --maxiter %d --tol %.2e"
        " --version %d (0=basic, 1=merged) ]"
        " matrices \n\n", format, (int) B.blocksize, (int) B.alignment,
        (int) scale,
        (int) solver_par.verbose,
        (int) solver_par.maxiter, solver_par.epsilon, version );

    magma_dsolverinfo_init( &solver_par, &precond_par );

    while(  i < argc ){

        magma_d_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale matrix
        magma_dmscale( &A, scaling );

        magma_d_mconvert( A, &B, Magma_CSR, B.storage_type );
        magma_d_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );

        // vectors and initial guess
        magma_d_vinit( &b, Magma_DEV, A.num_cols, one );
        magma_d_vinit( &x, Magma_DEV, A.num_cols, one );
        magma_d_spmv( one, B_d, x, zero, b );                 //  b = A x
        magma_d_vfree(&x);
        magma_d_vinit( &x, Magma_DEV, A.num_cols, zero );

        if( version == 0 )
            magma_dcg_res( B_d, b, &x, &solver_par );
        else if ( version == 1 )
            magma_dcg_merge( B_d, b, &x, &solver_par );

        magma_dsolverinfo( &solver_par, &precond_par );

        magma_d_mfree(&B_d);
        magma_d_mfree(&B);
        magma_d_mfree(&A); 
        magma_d_vfree(&x);
        magma_d_vfree(&b);

        i++;
    }
    
    magma_dsolverinfo_free( &solver_par, &precond_par );

    TESTING_FINALIZE();
    return 0;
}
