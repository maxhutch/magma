/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

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
int main(  int argc, char** argv )
{
    TESTING_INIT();

    magma_zopts zopts;
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );
    
    int i=1;
    magma_zparse_opts( argc, argv, &zopts, &i, queue );


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

    magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue );

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_zm_5stencil(  laplace_size, &A, queue );
        } else {                        // file-matrix test
            magma_z_csr_mtx( &A,  argv[i], queue );
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );


        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_rows;
        magma_zeigensolverinfo_init( &zopts.solver_par, queue );

        // scale matrix
        magma_zmscale( &A, zopts.scaling, queue );

        magma_z_mconvert( A, &B, Magma_CSR, zopts.output_format, queue );
        magma_z_mtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue );

        // vectors and initial guess
        magma_z_vinit( &b, Magma_DEV, A.num_cols, one, queue );
        magma_z_vinit( &x, Magma_DEV, A.num_cols, one, queue );
        magma_z_spmv( one, B_d, x, zero, b, queue );                 //  b = A x
        magma_z_vfree(&x, queue );
        magma_z_vinit( &x, Magma_DEV, A.num_cols, zero, queue );

        magma_z_solver( B_d, b, &x, &zopts, queue );         

        magma_zsolverinfo( &zopts.solver_par, &zopts.precond_par, queue );

        magma_z_mfree(&B_d, queue );
        magma_z_mfree(&B, queue );
        magma_z_mfree(&A, queue ); 
        magma_z_vfree(&x, queue );
        magma_z_vfree(&b, queue );

        i++;
    }

    magma_zsolverinfo_free( &zopts.solver_par, &zopts.precond_par, queue );
    
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
