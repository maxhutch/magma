/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

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
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_zopts zopts;

    int i=1;
    magma_zparse_opts( argc, argv, &zopts, &i);


    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magma_z_sparse_matrix A, B, B_d;
    magma_z_vector x, b;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    if ( zopts.solver_par.solver != Magma_PCG &&
         zopts.solver_par.solver != Magma_PGMRES &&
         zopts.solver_par.solver != Magma_PBICGSTAB &&
         zopts.solver_par.solver != Magma_ITERREF )
    zopts.precond_par.solver = Magma_NONE;

    magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par );

    while(  i < argc ){

        magma_z_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale matrix
        magma_zmscale( &A, zopts.scaling );

        magma_z_mconvert( A, &B, Magma_CSR, zopts.output_format );
        magma_z_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );

        // vectors and initial guess
        magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
        magma_z_vinit( &x, Magma_DEV, A.num_cols, one );
        magma_z_spmv( one, B_d, x, zero, b );                 //  b = A x
        magma_z_vfree(&x);
        magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

        magma_z_solver( B_d, b, &x, &zopts ); 

        magma_zsolverinfo( &zopts.solver_par, &zopts.precond_par );

        magma_z_mfree(&B_d);
        magma_z_mfree(&B);
        magma_z_mfree(&A); 
        magma_z_vfree(&x);
        magma_z_vfree(&b);

        i++;
    }

    magma_zsolverinfo_free( &zopts.solver_par, &zopts.precond_par );

    TESTING_FINALIZE();
    return 0;
}
