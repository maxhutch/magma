/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from testing_zsolver.cpp normal z -> s, Tue Sep  2 12:38:36 2014
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
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_sopts zopts;

    int i=1;
    magma_sparse_opts( argc, argv, &zopts, &i);


    float one = MAGMA_S_MAKE(1.0, 0.0);
    float zero = MAGMA_S_MAKE(0.0, 0.0);
    magma_s_sparse_matrix A, B, B_d;
    magma_s_vector x, b;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    if ( zopts.solver_par.solver != Magma_PCG &&
         zopts.solver_par.solver != Magma_PGMRES &&
         zopts.solver_par.solver != Magma_PBICGSTAB &&
         zopts.solver_par.solver != Magma_ITERREF )
    zopts.precond_par.solver = Magma_NONE;

    magma_ssolverinfo_init( &zopts.solver_par, &zopts.precond_par );

    while(  i < argc ){

        magma_s_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale matrix
        magma_smscale( &A, zopts.scaling );

        magma_s_mconvert( A, &B, Magma_CSR, zopts.output_format );
        magma_s_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );

        // vectors and initial guess
        magma_s_vinit( &b, Magma_DEV, A.num_cols, one );
        magma_s_vinit( &x, Magma_DEV, A.num_cols, one );
        magma_s_spmv( one, B_d, x, zero, b );                 //  b = A x
        magma_s_vfree(&x);
        magma_s_vinit( &x, Magma_DEV, A.num_cols, zero );

        magma_s_solver( B_d, b, &x, &zopts ); 

        magma_ssolverinfo( &zopts.solver_par, &zopts.precond_par );

        magma_s_mfree(&B_d);
        magma_s_mfree(&B);
        magma_s_mfree(&A); 
        magma_s_vfree(&x);
        magma_s_vfree(&b);

        i++;
    }

    magma_ssolverinfo_free( &zopts.solver_par, &zopts.precond_par );

    TESTING_FINALIZE();
    return 0;
}
