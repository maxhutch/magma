/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

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
   -- running magma_ziterref
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_z_solver_par solver_par;
    solver_par.epsilon = 10e-16;
    solver_par.maxiter = 1000;
    solver_par.verbose = 0;
    solver_par.num_eigenvalues = 0;
    int version = 0;
    int format = 0;

    magma_z_preconditioner precond_par;
    precond_par.solver = Magma_JACOBI;
    precond_par.epsilon = 1e-8;
    precond_par.maxiter = 100;
    precond_par.restart = 30;

    magma_z_sparse_matrix A, B, B_d;
    magma_z_vector x, b;
    B.blocksize = 8;
    B.alignment = 8;
    int scale = 0;
    magma_scale_t scaling = Magma_NOSCALE;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

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

        }else if ( strcmp("--blocksize", argv[i]) == 0 ) {
            B.blocksize = atoi( argv[++i] );
        }else if ( strcmp("--alignment", argv[i]) == 0 ) {
            B.alignment = atoi( argv[++i] );
        }else if ( strcmp("--maxiter", argv[i]) == 0 ) {
            solver_par.maxiter = atoi( argv[++i] );
        }else if ( strcmp("--verbose", argv[i]) == 0 ) {
            solver_par.verbose = atoi( argv[++i] );
        } else if ( strcmp("--tol", argv[i]) == 0 ) {
            sscanf( argv[++i], "%lf", &solver_par.epsilon );
        } else if ( strcmp("--preconditioner", argv[i]) == 0 ) {
            version = atoi( argv[++i] );
            switch( version ) {
                case 0: precond_par.solver = Magma_JACOBI; break;
                case 1: precond_par.solver = Magma_CG; break;
                case 2: precond_par.solver = Magma_BICGSTAB; break;
                case 3: precond_par.solver = Magma_GMRES; break;
            }
        } else if ( strcmp("--precond-maxiter", argv[i]) == 0 ) {
            precond_par.maxiter = atoi( argv[++i] );
        }else if ( strcmp("--precond-tol", argv[i]) == 0 ) {
            sscanf( argv[++i], "%lf", &precond_par.epsilon );
        }else if ( strcmp("--precond-restart", argv[i]) == 0 ) {
            precond_par.restart = atoi( argv[++i] );
        }else 
            break;
    }
    printf( "\n#    usage: ./run_ziterref"
        " [ --format %d (0=CSR, 1=ELL 2=ELLRT, 3=SELLP)"
        " [ --blocksize %d --alignment %d ]"
        " --mscale %d (0=no, 1=unitdiag, 2=unitrownrm)"
        " --verbose %d (0=summary, k=details every k iterations)"
        " --maxiter %d --tol %.2e"
        " --preconditioner %d (0=Jacobi, 1=CG, 2=BiCGStab, 3=GMRES)"
        " [ --precond-maxiter %d --precond-tol %.2e"
        " --precond-restart %d ] ]"
        " matrices \n\n", format, (int) B.blocksize, (int) B.alignment,
        (int) scale,
        (int) solver_par.verbose,
        (int) solver_par.maxiter, solver_par.epsilon, version,
        (int) precond_par.maxiter, precond_par.epsilon, 
        (int) precond_par.restart );

    magma_zsolverinfo_init( &solver_par, &precond_par );

    while(  i < argc ){

        magma_z_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale initial guess
        magma_zmscale( &A, scaling );

        magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
        magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

        magma_z_mconvert( A, &B, Magma_CSR, B.storage_type );
        magma_z_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );

        magma_ziterref( B_d, b, &x, &solver_par, &precond_par );

        magma_zsolverinfo( &solver_par, &precond_par );

        magma_z_mfree(&B_d);
        magma_z_mfree(&B);
        magma_z_mfree(&A); 
        magma_z_vfree(&x);
        magma_z_vfree(&b);
    
        i++;
    }

    magma_zsolverinfo_free( &solver_par, &precond_par );

    TESTING_FINALIZE();
    return 0;
}
