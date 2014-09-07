/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @generated from testing_zsolver.cpp normal z -> d, Tue Sep  2 12:38:36 2014
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

    magma_dopts zopts;

    int i=1;
    magma_dparse_opts( argc, argv, &zopts, &i);


    double one = MAGMA_D_MAKE(1.0, 0.0);
    double zero = MAGMA_D_MAKE(0.0, 0.0);
    magma_d_sparse_matrix A, B, B_d;
    magma_d_vector x, b;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    if ( zopts.solver_par.solver != Magma_PCG &&
         zopts.solver_par.solver != Magma_PGMRES &&
         zopts.solver_par.solver != Magma_PBICGSTAB &&
         zopts.solver_par.solver != Magma_ITERREF )
    zopts.precond_par.solver = Magma_NONE;

    magma_dsolverinfo_init( &zopts.solver_par, &zopts.precond_par );

    while(  i < argc ){

        magma_d_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale matrix
        magma_dmscale( &A, zopts.scaling );

        magma_d_mconvert( A, &B, Magma_CSR, zopts.output_format );
        magma_d_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );

        // vectors and initial guess
        magma_d_vinit( &b, Magma_DEV, A.num_cols, one );
        magma_d_vinit( &x, Magma_DEV, A.num_cols, one );
        magma_d_spmv( one, B_d, x, zero, b );                 //  b = A x
        magma_d_vfree(&x);
        magma_d_vinit( &x, Magma_DEV, A.num_cols, zero );

        magma_d_solver( B_d, b, &x, &zopts ); 

        magma_dsolverinfo( &zopts.solver_par, &zopts.precond_par );

        magma_d_mfree(&B_d);
        magma_d_mfree(&B);
        magma_d_mfree(&A); 
        magma_d_vfree(&x);
        magma_d_vfree(&b);

        i++;
    }

    magma_dsolverinfo_free( &zopts.solver_par, &zopts.precond_par );

    TESTING_FINALIZE();
    return 0;
}
